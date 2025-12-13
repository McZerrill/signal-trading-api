import time
from threading import RLock
from typing import Optional, Tuple, Dict

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Riduci il rumore di yfinance (tieni WARNING / ERROR)
logging.getLogger("yfinance").setLevel(logging.WARNING)

# ============================================================
#  Config cache
# ============================================================

YF_CACHE_TTL = 60  # secondi

# key: (symbol, interval, period) -> (df, timestamp)
_YF_CACHE: Dict[Tuple[str, str, str], Tuple[pd.DataFrame, float]] = {}
_YF_LOCK = RLock()


def _map_interval(interval: str) -> str:
    """
    Normalizza gli intervalli in formato accettato da yfinance.
    """
    interval = interval.lower().strip()
    allowed = {
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h",
        "1d", "5d", "1wk", "1mo", "3mo",
    }
    if interval in allowed:
        return interval
    if interval == "1h":
        return "60m"
    if interval == "4h":
        return "240m"
    return "1d"


def _default_period(interval: str) -> str:
    """
    Sceglie un periodo di default compatibile con l'intervallo.
    (equivalente al vecchio range_str)
    """
    interval = interval.lower()
    if interval.endswith("m") or interval in ("60m", "90m", "1h"):
        return "7d"
    if interval in ("1d", "5d"):
        return "1y"
    return "1y"


def _cache_get(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    key = (symbol, interval, period)
    now = time.time()
    with _YF_LOCK:
        entry = _YF_CACHE.get(key)
        if not entry:
            return None
        df, ts = entry
        if now - ts > YF_CACHE_TTL:
            _YF_CACHE.pop(key, None)
            return None
        return df.copy(deep=True)


def _cache_set(symbol: str, interval: str, period: str, df: pd.DataFrame) -> None:
    key = (symbol, interval, period)
    with _YF_LOCK:
        _YF_CACHE[key] = (df.copy(deep=True), time.time())


def _normalize_ohlc_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Porta il DataFrame di yfinance a:
    index: datetime (UTC)
    columns: open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Caso MultiIndex (es. ('Close','GC=F'))
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        # tipico: level 0 = Open/High/Low/Close, level 1 = ticker
        if symbol in lvl1:
            df = df.xs(symbol, axis=1, level=1)
        elif symbol in lvl0:
            df = df.xs(symbol, axis=1, level=0)
        else:
            # fallback: usa solo il livello 0
            df.columns = lvl0

    # Ora le colonne dovrebbero essere singolo livello
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Assicura la presenza delle 5 colonne base
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Index a datetime UTC
    df.index = pd.to_datetime(df.index, utc=True)
    df.rename_axis("datetime", inplace=True)

    # Converte i numerici in float (per evitare Series/object strani)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    return df


# ============================================================
#  API principali usate dal backend
# ============================================================

def get_yahoo_df(
    symbol: str,
    interval: str = "15m",
    range_str: Optional[str] = None,
) -> pd.DataFrame:
    """
    Restituisce un DataFrame compatibile con trend_logic:
    colonne: open, high, low, close, volume
    index: datetime (UTC)
    """
    yf_interval = _map_interval(interval)
    period = range_str or _default_period(yf_interval)

    # 1) cache
    cached = _cache_get(symbol, yf_interval, period)
    if cached is not None:
        return cached

    # 2) chiamata yfinance
    try:
        raw = yf.download(
            tickers=symbol,
            period=period,
            interval=yf_interval,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
            group_by="column",  # evita MultiIndex, ma gestiamo anche il caso opposto
        )
    except Exception as e:
        logger.warning(f"[YF] Errore download per {symbol}: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = _normalize_ohlc_df(raw, symbol)

    _cache_set(symbol, yf_interval, period, df)
    return df


def get_yahoo_last_price(symbol: str) -> float:
    """
    Ultimo prezzo di chiusura recente.
    Usa un daily 1d/5d (pochi dati, quasi zero rischio 429).
    """
    df = get_yahoo_df(symbol, interval="1d", range_str="5d")
    if df.empty:
        return 0.0
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0


# ============================================================
#  Mappa simboli "logici" → ticker reali Yahoo
# ============================================================

YAHOO_SYMBOL_MAP = {
    # Macro / futures / indici
    "XAUUSD": "GC=F",     # Oro futures
    "XAGUSD": "SI=F",     # Argento futures
    "SP500":  "^GSPC",    # S&P 500
    "NAS100": "^NDX",     # Nasdaq 100
    "DAX40":  "^GDAXI",   # DAX tedesco

    # Crypto principali (ticker Yahoo)
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "ADAUSDT": "ADA-USD",

    # Alias senza USDT
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD",
    "ADAUSD": "ADA-USD",
}
