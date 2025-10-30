import time
import os
from typing import Optional

import requests
import pandas as pd
from binance.client import Client


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Quote supportate (USDT + USDC per compatibilità con il resto del backend)
QUOTE_SUFFIXES = ("USDT", "USDC")

# Modalità test
MODALITA_TEST = True

# Inizializza client Binance con chiavi da env
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Cache simboli 60s
_symbol_cache = {"time": 0.0, "data": []}

# Cache tick/step 30 min
_tick_cache: dict[str, dict] = {}
TICK_TTL_SEC = 1800  # 30 minuti


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now() -> float:
    return time.time()


def _ok_symbol(sym: str) -> bool:
    """Filtra spot symbol:
       - quote in QUOTE_SUFFIXES
       - esclude token a leva (UP/DOWN/BULL/BEAR)
    """
    if not any(sym.endswith(q) for q in QUOTE_SUFFIXES):
        return False
    lev = ("UP", "DOWN", "BULL", "BEAR")
    return not any(x in sym for x in lev)


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
def get_best_symbols(limit: int = 80):
    """
    Restituisce i migliori simboli per volume 24h (quote USDT/USDC),
    con cache di 60 secondi.
    """
    now = _now()
    if now - _symbol_cache["time"] < 60 and _symbol_cache["data"]:
        return _symbol_cache["data"][:limit]

    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Filtro robusto
        filtered = []
        for d in data:
            sym = d.get("symbol", "")
            if not sym or not _ok_symbol(sym):
                continue
            try:
                qv = float(d.get("quoteVolume", 0.0))
            except Exception:
                qv = 0.0
            # soglia diversa tra test/prod
            min_vol = 1_000_000 if MODALITA_TEST else 5_000_000
            if qv >= min_vol:
                filtered.append((sym, qv))

        # Ordina per volume 24h decrescente
        filtered.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in filtered[:limit]]

        # Fallback se vuoto (evita app “senza dati”)
        if not symbols:
            symbols = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT",
                "BTCUSDC", "ETHUSDC", "BNBUSDC"
            ]

        _symbol_cache["time"] = now
        _symbol_cache["data"] = symbols
        print(f"✅ get_best_symbols: {len(symbols)} simboli candidati (quote {QUOTE_SUFFIXES})")
        return symbols

    except Exception as e:
        print("❌ Errore get_best_symbols:", e)
        # fallback statico
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "BTCUSDC", "ETHUSDC", "BNBUSDC",
            "SOLUSDT", "XRPUSDT", "ADAUSDT"
        ][:limit]


def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> pd.DataFrame:
    """
    Scarica klines e restituisce OHLCV come DataFrame float con DatetimeIndex.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time

    try:
        klines = client.get_klines(**params)
    except Exception as e:
        print(f"❌ Errore get_klines {symbol}-{interval}: {e}")
        return pd.DataFrame()

    if not klines:
        print(f"⚠️ Nessuna candela per {symbol} ({interval})")
        return pd.DataFrame()
    if len(klines) < 50:
        print(f"ℹ️ Dati parziali per {symbol} ({interval}): {len(klines)} candele")

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


def get_bid_ask(symbol: str) -> dict:
    """
    Recupera bid/ask reali dal book Binance e calcola lo spread percentuale.
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        data = requests.get(url, timeout=5).json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        spread = (ask - bid) / ((ask + bid) / 2.0) * 100.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}
    except Exception as e:
        print(f"❌ Errore get_bid_ask({symbol}):", e)
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}


def get_symbol_tick_step(symbol: str) -> tuple[float, float]:
    """
    Ritorna (tick_size, step_size) reali da Binance (exchangeInfo) con cache 30 minuti.
    Fallback: (0.0001, 0.0001).
    """
    s = (symbol or "").upper()
    if not s:
        return 0.0001, 0.0001

    now = _now()
    cache = _tick_cache.get(s)
    if cache and (now - cache.get("ts", 0.0) < TICK_TTL_SEC):
        return cache["tick"], cache["step"]

    try:
        url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={s}"
        data = requests.get(url, timeout=6).json()

        symbols = data.get("symbols") or []
        if not symbols:
            raise ValueError(f"exchangeInfo vuoto per {s}")

        info = symbols[0]
        tick = 0.0001
        step = 0.0001
        for f in info.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                tick = float(f.get("tickSize", tick))
            elif ft == "LOT_SIZE":
                step = float(f.get("stepSize", step))

        _tick_cache[s] = {"tick": tick, "step": step, "ts": now}
        return tick, step

    except Exception as e:
        print(f"⚠️ Errore get_symbol_tick_step({symbol}): {e}")
        return 0.0001, 0.0001
