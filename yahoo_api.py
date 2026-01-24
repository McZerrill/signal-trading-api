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

def _ttl_for_interval(interval: str) -> int:
    interval = (interval or "").lower().strip()
    if interval in ("1m", "2m", "5m"):
        return 15
    if interval in ("15m", "30m", "60m", "90m"):
        return 45
    return YF_CACHE_TTL

def _map_interval(interval: str) -> str:
    """
    Normalizza gli intervalli in formato accettato da yfinance.
    """
    interval = interval.lower().strip()

    # normalizzazioni esplicite
    if interval == "1h":
        return "60m"
    if interval == "4h":
        return "240m"

    allowed = {
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m",
        "1d", "5d", "1wk", "1mo", "3mo",
    }
    if interval in allowed:
        return interval

    return "1d"



def _default_period(interval: str) -> str:
    interval = (interval or "").lower().strip()

    # intraday: period più corti = meno errori e più "fresco"
    if interval == "1m":
        return "1d"
    if interval in ("2m", "5m"):
        return "5d"
    if interval in ("15m", "30m", "60m", "90m"):
        return "30d"

    # daily+
    if interval in ("1d", "5d"):
        return "1y"
    return "1y"

def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    interval = (interval or "").lower().strip()
    if interval.endswith("m"):
        try:
            return pd.Timedelta(minutes=int(interval[:-1]))
        except Exception:
            return pd.Timedelta(minutes=15)
    if interval.endswith("h"):
        try:
            return pd.Timedelta(hours=int(interval[:-1]))
        except Exception:
            return pd.Timedelta(hours=1)
    if interval in ("1d", "5d"):
        return pd.Timedelta(days=1)
    return pd.Timedelta(minutes=15)


def _allowed_lag(interval: str) -> pd.Timedelta:
    """
    Quanto ritardo massimo tolleriamo per considerare 'freschi' i dati Yahoo.
    Se oltre, li marchiamo come STALE (e in routes puoi evitare notifiche).
    """
    interval = (interval or "").lower().strip()
    # margini “larghi” per evitare falsi STALE, ma bloccano i 15m vecchi di 15m+
    if interval in ("1m", "2m", "5m"):
        return pd.Timedelta(minutes=6)
    if interval == "15m":
        return pd.Timedelta(minutes=12)
    if interval == "30m":
        return pd.Timedelta(minutes=22)
    if interval in ("60m", "1h"):
        return pd.Timedelta(minutes=50)
    if interval == "90m":
        return pd.Timedelta(minutes=75)
    return pd.Timedelta(minutes=25)


def _cache_get(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    key = (symbol, interval, period)
    now = time.time()
    with _YF_LOCK:
        entry = _YF_CACHE.get(key)
        if not entry:
            return None
        df, ts = entry
        ttl = _ttl_for_interval(interval)
        if now - ts > ttl:
            _YF_CACHE.pop(key, None)
            return None
        return df.copy(deep=True)


def _cache_set(symbol: str, interval: str, period: str, df: pd.DataFrame) -> None:
    key = (symbol, interval, period)
    with _YF_LOCK:
        _YF_CACHE[key] = (df.copy(deep=True), time.time())


def _normalize_ohlc_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Porta il DataFrame di yfinance (qualsiasi formato) a:
    index: datetime (UTC)
    columns: open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # --- Gestione MultiIndex o livelli strani ---
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        # Caso classico: ('Open','BTC-USD') o ('Open','GC=F')
        if symbol in lvl1:
            df = df.xs(symbol, axis=1, level=1)
        elif symbol in lvl0:
            df = df.xs(symbol, axis=1, level=0)
        else:
            # Se il livello 1 è vuoto o uguale per tutti, prendo solo il primo livello
            df.columns = lvl0
    else:
        # singolo livello: non serve toccare
        pass

    # --- Normalizza nomi colonne ---
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    # In caso di tuple residue, converti tutto a stringa
    df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.rename(columns=rename_map)

    # se close manca ma adj_close c'è, usa adj_close come close
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    # se close è tutto NaN ma adj_close ha valori, usa quello
    if "adj_close" in df.columns and "close" in df.columns:
        try:
            if df["close"].isna().all() and (not df["adj_close"].isna().all()):
                df["close"] = df["adj_close"]
        except Exception:
            pass



    # --- Aggiungi colonne mancanti ---
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # --- Pulisci e mantieni solo le 5 ---
    df = df[["open", "high", "low", "close", "volume"]]

    # --- Converti in float/num in modo sicuro ---
    for col in ["open", "high", "low", "close", "volume"]:
        try:
            # Se è una Series va bene; se è DataFrame, prendo la prima colonna
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            df[col] = pd.to_numeric(col_data, errors="coerce")
        except Exception:
            df[col] = pd.NA

    # --- Rimuovi righe senza close ---
    df = df.dropna(subset=["close"], how="any")

    # --- Index datetime ---
    df.index = pd.to_datetime(df.index, utc=True)
    df.rename_axis("datetime", inplace=True)

    return df

def _patch_last_bar_with_1m(symbol: str, df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Se Yahoo 15m/30m/60m è in ritardo, prova a “rinfrescare” l’ultima candela
    usando i dati 1m (close/high/low + volume se presente).
    """
    try:
        if df is None or df.empty:
            return df
        interval = (interval or "").lower().strip()
        if interval not in ("15m", "30m", "60m", "90m"):
            return df

        last_ts = df.index[-1]
        now_utc = pd.Timestamp.now(tz="UTC")

        # Se non siamo in ritardo, non fare nulla
        if (now_utc - last_ts) <= _allowed_lag(interval):
            return df

        # Prendi 1m (può fallire su alcuni ticker: ok, allora non patchiamo)
        df1 = get_yahoo_df(symbol, interval="1m", range_str="1d")
        if df1 is None or df1.empty:
            return df

        # Finestra 1m a partire dall’inizio dell’ultima candela "grossa"
        start = last_ts
        sub = df1[df1.index >= start]
        if sub.empty:
            return df

        # Patch OHLCV dell’ultima riga
        last_close = float(sub["close"].iloc[-1])
        hi = float(sub["high"].max()) if "high" in sub.columns else last_close
        lo = float(sub["low"].min())  if "low"  in sub.columns else last_close

        df = df.copy()
        df.loc[last_ts, "close"] = last_close
        if "high" in df.columns and pd.notna(df.loc[last_ts, "high"]):
            df.loc[last_ts, "high"] = max(float(df.loc[last_ts, "high"]), hi)
        else:
            df.loc[last_ts, "high"] = hi

        if "low" in df.columns and pd.notna(df.loc[last_ts, "low"]):
            df.loc[last_ts, "low"] = min(float(df.loc[last_ts, "low"]), lo)
        else:
            df.loc[last_ts, "low"] = lo

        # volume: solo se significativo (su indici/metalli spesso è 0)
        if "volume" in df.columns and "volume" in sub.columns:
            v = float(pd.to_numeric(sub["volume"], errors="coerce").fillna(0.0).sum())
            if v > 0:
                df.loc[last_ts, "volume"] = v

        # info diagnostica per routes/log
        df.attrs = dict(getattr(df, "attrs", {}) or {})
        df.attrs["patched_with_1m"] = True
        df.attrs["patched_at_utc"] = str(now_utc)
        df.attrs["patch_from_1m_last_ts"] = str(sub.index[-1])

        logger.warning(
            f"[YAHOO_API] PATCH 1m -> {symbol} {interval}: "
            f"last_ts={last_ts} now={now_utc} lag={(now_utc-last_ts)}"
        )

        return df

    except Exception as e:
        logger.warning(f"[YAHOO_API] patch_last_bar_with_1m error {symbol} {interval}: {e}")
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
    logging.debug(f"[YAHOO_API] FETCH REMOTO per {symbol} ({yf_interval}, period={period})")

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

    # --- RECENCY CHECK + METADATA (per evitare notifiche su dati vecchi) ---
    try:
        df.attrs = dict(getattr(df, "attrs", {}) or {})
        df.attrs["fetched_at_utc"] = str(pd.Timestamp.now(tz="UTC"))
        df.attrs["yf_interval"] = yf_interval
        df.attrs["period"] = period

        if not df.empty:
            last_ts = df.index[-1]
            now_utc = pd.Timestamp.now(tz="UTC")
            lag = now_utc - last_ts
            df.attrs["last_ts_utc"] = str(last_ts)
            df.attrs["lag_seconds"] = float(lag.total_seconds())

            lag_ok = lag <= _allowed_lag(yf_interval)
            df.attrs["is_stale"] = (not lag_ok)

            if not lag_ok:
                logger.warning(
                    f"[YAHOO_API] STALE {symbol} {yf_interval}: "
                    f"last_ts={last_ts} now={now_utc} lag={lag}"
                )

        # --- PATCH last bar con 1m se 15m/30m/60m è in ritardo ---
        df = _patch_last_bar_with_1m(symbol, df, yf_interval)

    except Exception as e:
        logger.debug(f"[YAHOO_API] recency/patch skip: {e}")

    _cache_set(symbol, yf_interval, period, df)

    if 'df' in locals() and isinstance(df, pd.DataFrame):
        logging.debug(f"[YAHOO_API] {symbol}: ricevuti {len(df)} dati ({interval})")
    else:
        logging.debug(f"[YAHOO_API] {symbol}: nessun dato ricevuto ({interval})")

    return df



def get_yahoo_last_price(symbol: str) -> float:
    """
    Ultimo close intraday quando disponibile (più vicino al "live"),
    fallback su daily se mercato chiuso o feed non disponibile.
    """
    # 1) prova intraday 1m (se disponibile)
    try:
        df1 = get_yahoo_df(symbol, interval="1m", range_str="1d")
        if df1 is not None and not df1.empty:
            return float(df1["close"].iloc[-1])
    except Exception:
        pass

    # 2) fallback: 5m
    try:
        df5 = get_yahoo_df(symbol, interval="5m", range_str="5d")
        if df5 is not None and not df5.empty:
            return float(df5["close"].iloc[-1])
    except Exception:
        pass

    # 3) fallback: 15m
    try:
        df15 = get_yahoo_df(symbol, interval="15m", range_str="30d")
        if df15 is not None and not df15.empty:
            return float(df15["close"].iloc[-1])
    except Exception:
        pass

    # 4) fallback finale: daily
    try:
        df = get_yahoo_df(symbol, interval="1d", range_str="5d")
        if df is None or df.empty:
            return 0.0
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0




# ============================================================
#  Mappa simboli "logici" → ticker reali Yahoo
# ============================================================

YAHOO_SYMBOL_MAP = {
    # -----------------------------
    # Macro / futures / indici
    # -----------------------------
    "XAUUSD":   "GC=F",       # Oro futures
    "XAGUSD":   "SI=F",       # Argento futures
    "SP500":    "^GSPC",      # S&P 500
    "NAS100":   "^NDX",       # Nasdaq 100
    "DAX40":    "^GDAXI",     # DAX tedesco
    "DJI":      "^DJI",       # Dow Jones
    "VIX":      "^VIX",       # Volatility Index
    "RUS2000":  "^RUT",       # Russell 2000

    "OILWTI":   "CL=F",       # Crude Oil WTI futures
    "OILB":     "BZ=F",       # Crude Oil Brent futures
    "RAME":     "HG=F",       # Rame futures
    "SOIA":     "ZS=F",       # Soia futures
    "GRANO":    "ZW=F",       # Grano futures
    "MAIS":     "ZC=F",       # Mais futures

    # -----------------------------
    # Forex (Yahoo)
    # -----------------------------
    "EURUSD":   "EURUSD=X",
    "GBPUSD":   "GBPUSD=X",
    "USDJPY":   "JPY=X",

    "NOKJPY":   "NOKJPY=X",
    "AUDDKK":   "AUDDKK=X",
    "NZDJPY":   "NZDJPY=X",
    "AUDJPY":   "AUDJPY=X",
    "USDDKK":   "USDDKK=X",
    "SGDJPY":   "SGDJPY=X",
    "NZDCHF":   "NZDCHF=X",
    "PLNJPY":   "PLNJPY=X",
    "AUDCHF":   "AUDCHF=X",
    "NZDUSD":   "NZDUSD=X",
    "CADCHF":   "CADCHF=X",
    "AUDUSD":   "AUDUSD=X",
    "NZDCAD":   "NZDCAD=X",
    "USDCHF":   "USDCHF=X",
    "EURCZK":   "EURCZK=X",
    "NZDMXN":   "NZDMXN=X",
    "NZDSGD":   "NZDSGD=X",
    "AUDCAD":   "AUDCAD=X",
    "CHFJPY":   "CHFJPY=X",
    "AUDSGD":   "AUDSGD=X",
    "USDTHB":   "USDTHB=X",
    "USDSEK":   "USDSEK=X",
    "USDHKD":   "USDHKD=X",
    "EURDKK":   "EURDKK=X",
    "USDPLN":   "USDPLN=X",
    "AUDNZD":   "AUDNZD=X",
    "USDCAD":   "USDCAD=X",
    "EURGBP":   "EURGBP=X",
    "USDZAR":   "USDZAR=X",
    "USDSGD":   "USDSGD=X",
    "USDMXN":   "USDMXN=X",
    "GBPJPY":   "GBPJPY=X",
    "EURJPY":   "EURJPY=X",
    "CHFSEK":   "CHFSEK=X",
    "GBPCHF":   "GBPCHF=X",
    "CHFSGD":   "CHFSGD=X",
    "EURCHF":   "EURCHF=X",
    "AUDNOK":   "AUDNOK=X",
    "GBPHKD":   "GBPHKD=X",
    "EURSEK":   "EURSEK=X",
    "EURHKD":   "EURHKD=X",
    "GBPCAD":   "GBPCAD=X",
    "GBPSGD":   "GBPSGD=X",
    "EURCAD":   "EURCAD=X",
    "GBPZAR":   "GBPZAR=X",
    "EURPLN":   "EURPLN=X",
    "EURSGD":   "EURSGD=X",
    "EURZAR":   "EURZAR=X",
    "EURCNH":   "EURCNH=X",
    "GBPAUD":   "GBPAUD=X",
    "GBPNZD":   "GBPNZD=X",
    "EURAUD":   "EURAUD=X",
    "EURNZD":   "EURNZD=X",
    "EURNOK":   "EURNOK=X",

    # -----------------------------
    # Basket / ETF (opzionali)
    # -----------------------------
    "MAG7":     "MAGS",       # Roundhill Magnificent Seven ETF (proxy)

    # -----------------------------
    # Titoli azionari (USA)
    # -----------------------------
    "AAPL":     "AAPL",
    "MSFT":     "MSFT",
    "NVDA":     "NVDA",
    "TSLA":     "TSLA",
    "META":     "META",
    "GOOGL":    "GOOGL",
    "AMZN":     "AMZN",
    "NFLX":     "NFLX",

    "ORCL":     "ORCL",
    "AMD":      "AMD",
    "AVGO":     "AVGO",
    "MU":       "MU",

    "KO":       "KO",
    "MCD":      "MCD",
    "COST":     "COST",
    "WMT":      "WMT",
    "V":        "V",
    "MA":       "MA",
    "UPS":      "UPS",

    "QCOM":     "QCOM",
    "ADBE":     "ADBE",
    "NKE":      "NKE",
    "INTC":     "INTC",

    "JNJ":      "JNJ",
    "LLY":      "LLY",
    "ABBV":     "ABBV",
    "ABT":      "ABT",

    "BRKB":     "BRK-B",
    "BABA":     "BABA",
    "MELI":     "MELI",

    "COIN":     "COIN",
    "RDDT":     "RDDT",
    "GME":      "GME",

    "CAT":      "CAT",
    "FCX":      "FCX",
    "AA":       "AA",
    "GOLD":     "GOLD",

    "BYND":     "BYND",
    "SPCE":     "SPCE",
    "LYFT":     "LYFT",

    "SEDG":     "SEDG",
    "FSLR":     "FSLR",
    "AAL":      "AAL",
    "LUV":      "LUV",
    "CCL":      "CCL",
    "DAL":      "DAL",
    "AMC":      "AMC",
    "ACB":      "ACB",

    "DIS":      "DIS",
    "JPM":      "JPM",
    "BAC":      "BAC",

    "FERRARI":  "RACE",       # Ferrari (NYSE)
}


