import requests
import pandas as pd
import time

# Cache semplice in memoria: (symbol, interval, range) -> (timestamp, df)
_YAHOO_DF_CACHE: dict[tuple[str, str, str], tuple[float, pd.DataFrame]] = {}


YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"


def _map_interval(interval: str) -> str:
    interval = interval.lower()
    if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d"):
        return interval
    if interval in ("1h", "1H"):
        return "60m"
    return "1d"


def _default_range(interval: str) -> str:
    if interval.endswith("m"):
        return "7d"
    if interval.endswith("d"):
        return "1y"
    return "1y"


def get_yahoo_df(
    symbol: str,
    interval: str = "15m",
    range_str: str | None = None,
    ttl_seconds: int = 60,
) -> pd.DataFrame:
    """
    Restituisce un DF con colonne: open, high, low, close, volume
    compatibile con trend_logic.

    Usa una cache in memoria con TTL per non stressare Yahoo
    (evita errori 429 Too Many Requests).
    """
    y_interval = _map_interval(interval)
    final_range = range_str or _default_range(y_interval)

    # ---- CACHE ----
    now = time.time()
    cache_key = (symbol, y_interval, final_range)
    cached = _YAHOO_DF_CACHE.get(cache_key)
    if cached is not None:
        ts_cached, df_cached = cached
        if now - ts_cached < ttl_seconds:
            # ritorna una copia per sicurezza
            return df_cached.copy()

    # ---- HTTP CALL A YAHOO ----
    params = {"interval": y_interval, "range": final_range}
    url = f"{YAHOO_BASE_URL}{symbol}"

    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    data = r.json()

    result_list = data.get("chart", {}).get("result")
    if not result_list:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        _YAHOO_DF_CACHE[cache_key] = (now, df)
        return df

    result = result_list[0]
    ts = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]

    if not ts or "close" not in quote:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        _YAHOO_DF_CACHE[cache_key] = (now, df)
        return df

    df = pd.DataFrame({
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    if len(df) != len(ts):
        m = min(len(df), len(ts))
        df = df.iloc[:m]
        ts = ts[:m]

    df.index = pd.to_datetime(ts, unit="s", utc=True)
    df = df.dropna(subset=["close"])
    df.rename_axis("datetime", inplace=True)

    # Salva in cache
    _YAHOO_DF_CACHE[cache_key] = (now, df.copy())
    return df


def get_yahoo_last_price(symbol: str) -> float:
    df = get_yahoo_df(symbol, interval="1d", range_str="5d")
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


# Mappa "nostri" simboli → ticker Yahoo
YAHOO_SYMBOL_MAP = {
    # ----- COMMODITIES / INDICI -----
    "XAUUSD": "GC=F",     # Oro futures
    "XAGUSD": "SI=F",     # Argento futures
    "SP500": "^GSPC",     # S&P 500
    "NAS100": "^NDX",     # Nasdaq 100
    "DAX40": "^GDAXI",    # DAX

    # ----- CRYPTO via Yahoo (USD) -----
    # (chiavi "logiche" che userai nella app /hotassets)
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
    "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD",
    "ADAUSD": "ADA-USD",
    "DOGEUSD": "DOGE-USD",
    "LTCUSD": "LTC-USD",

    # ----- Esempi azioni, se ti servono -----
    # "AAPL": "AAPL",
    # "TSLA": "TSLA",
}

