import os
import time
import requests
import pandas as pd
from typing import Optional
from binance.client import Client
from cachetools import TTLCache, cached, keys

# -----------------------------------------------------------------------------
# Inizializza il client Binance con chiavi da variabili d'ambiente
# -----------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# ✅ Modalità test attiva
MODALITA_TEST = True

# -----------------------------------------------------------------------------
# Caching (cachetools)
# -----------------------------------------------------------------------------
# Simboli (TTL 60s)
_symbols_cache = TTLCache(maxsize=1, ttl=60)

# Tick/step per simbolo (TTL 30 min)
_tick_cache = TTLCache(maxsize=256, ttl=1800)

# OHLCV per (symbol, interval, limit, end_time) (TTL 5s)
_df_cache = TTLCache(maxsize=512, ttl=5)


# =========================
#   SYMBOLS BEST (24hr)
# =========================
@cached(cache=_symbols_cache)
def _fetch_best_symbols_full() -> list[str]:
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        filtered = [
            d for d in data
            if d.get("symbol", "").endswith("USDC")
            and not any(x in d.get("symbol", "") for x in ["UP", "DOWN", "BULL", "BEAR"])
            and float(d.get("quoteVolume", 0.0)) > (1_000_000 if MODALITA_TEST else 5_000_000)
        ]

        sorted_pairs = sorted(filtered, key=lambda x: float(x["quoteVolume"]), reverse=True)
        top_symbols = [d["symbol"] for d in sorted_pairs]
        print(f"✅ {len(top_symbols)} simboli trovati con volume > 1M USDC")
        return top_symbols

    except Exception as e:
        print("❌ Errore nel recupero simboli Binance:", e)
        return [
            "BTCUSDC", "ETHUSDC", "BNBUSDC", "XRPUSDC", "ADAUSDC",
            "SOLUSDC", "AVAXUSDC", "DOTUSDC", "DOGEUSDC", "MATICUSDC"
        ]


def get_best_symbols(limit: int = 80) -> list[str]:
    full_list = _fetch_best_symbols_full()
    return full_list[:max(0, int(limit))]


# =========================
#   OHLCV (15m/1h/1d/…)
# =========================
def _df_cache_key(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None):
    return keys.hashkey(symbol.upper(), interval, int(limit), int(end_time) if end_time is not None else None)


@cached(cache=_df_cache, key=_df_cache_key)
def _get_binance_df_cached(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time

    try:
        klines = client.get_klines(**params)
    except Exception as e:
        print(f"❌ Errore nel caricamento candela {symbol}-{interval}: {e}")
        return pd.DataFrame()

    if not klines:
        print(f"⚠️ Nessuna candela per {symbol} ({interval})")
        return pd.DataFrame()
    if len(klines) < 50:
        print(f"ℹ️ Dati parziali per {symbol} ({interval}): {len(klines)} candele (ok)")

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.dropna(inplace=True)
    return df


def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> pd.DataFrame:
    """
    Wrapper pubblico: usa la cache a TTL 5s e restituisce SEMPRE una copia profonda,
    così ogni chiamata può modificare il DataFrame senza intaccare la cache.
    """
    df = _get_binance_df_cached(symbol, interval, limit, end_time)
    return df.copy(deep=True)


# =========================
#   BID/ASK & SPREAD
# =========================
def get_bid_ask(symbol: str) -> dict:
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        data = requests.get(url, timeout=5).json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        spread = (ask - bid) / ((ask + bid) / 2) * 100.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}
    except Exception as e:
        print(f"❌ Errore get_bid_ask per {symbol}:", e)
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}


# =========================
#   TICK & STEP SIZE
# =========================
@cached(cache=_tick_cache, key=lambda symbol: keys.hashkey(symbol.upper()))
def get_symbol_tick_step(symbol: str) -> tuple[float, float]:
    """
    (tick_size, step_size) da /exchangeInfo, TTL 30 minuti. Fallback (0.0001, 0.0001).
    """
    try:
        s = symbol.upper()
        url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={s}"
        data = requests.get(url, timeout=5).json()
        info_list = data.get("symbols") or []
        if not info_list:
            raise ValueError("symbols vuoto in exchangeInfo")

        info = info_list[0]
        tick = 0.0001
        step = 0.0001
        for f in info.get("filters", []):
            ftype = f.get("filterType")
            if ftype == "PRICE_FILTER":
                tick = float(f.get("tickSize", tick))
            elif ftype == "LOT_SIZE":
                step = float(f.get("stepSize", step))
        return tick, step

    except Exception as e:
        print(f"⚠️ Errore get_symbol_tick_step({symbol}): {e}")
        return 0.0001, 0.0001
