import time
from threading import RLock
from typing import Optional

import pandas as pd
import yfinance as yf

# ============================================================
#  Config
# ============================================================

# TTL cache per le chiamate Yahoo (in secondi)
YF_CACHE_TTL = 60  # 1 minuto; riduce tantissimo il rischio di 429

# Cache semplice in RAM
# key: (symbol, interval, period) -> (df, timestamp)
_YF_CACHE: dict[tuple[str, str, str], tuple[pd.DataFrame, float]] = {}
_YF_LOCK = RLock()


def _map_interval(interval: str) -> str:
    """
    Normalizza gli intervalli in formato accettato da yfinance.
    """
    interval = interval.lower().strip()
    # yfinance supporta direttamente questi
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
    # fallback prudente
    return "1d"


def _default_period(interval: str) -> str:
    """
    Sceglie un periodo di default compatibile con l'intervallo.
    (equivalente al vecchio range_str)
    """
    interval = interval.lower()
    if interval.endswith("m") or interval in ("60m", "90m", "1h"):
        # per intraday 15m/1h: 7 giorni bastano a trend_logic
        return "7d"
    if interval in ("1d", "5d"):
        return "1y"
    # fallback generico
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
            # scaduto
            _YF_CACHE.pop(key, None)
            return None
        # ritorno una copia per evitare side-effect
        return df.copy(deep=True)


def _cache_set(symbol: str, interval: str, period: str, df: pd.DataFrame) -> None:
    key = (symbol, interval, period)
    with _YF_LOCK:
        _YF_CACHE[key] = (df.copy(deep=True), time.time())


# ============================================================
#  API principali usate dal backend
# ============================================================

def get_yahoo_df(
    symbol: str,
    interval: str = "15m",
    range_str: Optional[str] = None,
) -> pd.DataFrame:
    """
    Restituisce un DataFrame con colonne: open, high, low, close, volume
    compatibile con trend_logic.

    - Usa yfinance (ticker.history / download)
    - Applica una cache interna (TTL = YF_CACHE_TTL sec)
    """
    yf_interval = _map_interval(interval)
    period = range_str or _default_period(yf_interval)

    # 1) tenta cache
    cached = _cache_get(symbol, yf_interval, period)
    if cached is not None:
        return cached

    # 2) chiamata a yfinance
    #    NB: usiamo download perché è semplice e robusto
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=yf_interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        # restituisco comunque un DF con le colonne giuste
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # yfinance in genere usa queste colonne: Open, High, Low, Close, Volume
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",  # fallback se manca Close
            "Volume": "volume",
        }
    )

    # Teniamo solo le colonne che ci servono
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["open", "high", "low", "close", "volume"]]

    # pulizia NaN sul close
    df = df.dropna(subset=["close"])

    # indicizza per datetime (già è un DatetimeIndex, ma normalizziamo)
    df.index = pd.to_datetime(df.index, utc=True)
    df.rename_axis("datetime", inplace=True)

    # 3) salva in cache
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

    # --- Crypto principali (ticker Yahoo) ---
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "ADAUSDT": "ADA-USD",

    # Alias "senza USDT" (se ti piace usarli così)
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD",
    "ADAUSD": "ADA-USD",
}

