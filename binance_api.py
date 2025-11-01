# binance_api.py
# -----------------------------------------------------------------------------
# Backend Binance helpers con cache corta per prezzi e timeout ridotti.
# Pensato per evitare timeouts lato app (3s) in giornate con latenza alta.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Dict, Any

import requests
import pandas as pd
from binance.client import Client

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Quote supportate (compatibili col resto del backend)
QUOTE_SUFFIXES: Tuple[str, str] = ("USDT", "USDC")

# Modalità test (volumi minimi più bassi su get_best_symbols)
MODALITA_TEST: bool = True

# Inizializza client Binance con chiavi da env
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Cache simboli (60s)
_symbol_cache: Dict[str, Any] = {"time": 0.0, "data": []}

# Cache tick/step (30 minuti)
_tick_cache: Dict[str, Dict[str, float]] = {}
TICK_TTL_SEC: float = 1800.0  # 30 min

# Mini-cache per prezzo/bid-ask (8s)
_price_cache: Dict[str, Dict[str, float]] = {}
PRICE_TTL: float = 8.0

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


def _get_json(url: str, params: Optional[dict] = None, timeout: float = 2.5, retries: int = 1) -> dict:
    """GET con timeout corto e 1 retry veloce (exponential-ish backoff)."""
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if i < retries:
                time.sleep(0.15 * (i + 1))
    # se arrivo qui, rilancio l'ultima eccezione
    raise last_exc  # type: ignore[misc]

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
def get_best_symbols(limit: int = 80):
    """Restituisce i migliori simboli per volume 24h (quote USDT/USDC), cache 60s."""
    now = _now()
    if now - _symbol_cache["time"] < 60 and _symbol_cache["data"]:
        return _symbol_cache["data"][:limit]

    try:
        data = _get_json("https://api.binance.com/api/v3/ticker/24hr", timeout=2.5, retries=1)

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
            min_vol = 1_000_000 if MODALITA_TEST else 5_000_000
            if qv >= min_vol:
                filtered.append((sym, qv))

        filtered.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in filtered[:limit]]

        # Fallback se vuoto
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
    """Scarica klines e restituisce OHLCV come DataFrame float con DatetimeIndex UTC."""
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
    """Recupera bid/ask reali con cache breve e timeout corto; calcola spread %."""
    s = (symbol or "").upper()
    if not s:
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}

    now = _now()

    # 1) Cache breve (PRICE_TTL secondi)
    c = _price_cache.get(s)
    if c and (now - float(c.get("ts", 0.0)) < float(PRICE_TTL)) and "bid" in c and "ask" in c:
        bid = float(c["bid"])
        ask = float(c["ask"])
        denom = (ask + bid) / 2.0 if (ask > 0.0 and bid > 0.0) else 0.0
        spread = ((ask - bid) / denom * 100.0) if denom > 0.0 else 0.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}

    # 2) Chiamata rete con timeout corto + 1 retry
    try:
        data = _get_json(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            {"symbol": s},
            timeout=1.8,   # corto per non far scadere il client mobile (3s)
            retries=1
        )
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])

        # aggiorna cache
        _price_cache[s] = {
            "ts": now,
            "bid": bid,
            "ask": ask,
            "price": (bid + ask) / 2.0
        }

        denom = (ask + bid) / 2.0 if (ask > 0.0 and bid > 0.0) else 0.0
        spread = ((ask - bid) / denom * 100.0) if denom > 0.0 else 0.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}

    except Exception as e:
        print(f"❌ Errore get_bid_ask({s}): {e}")

        # 3) Fallback su cache (anche se un po' stantia è meglio di zero)
        c = _price_cache.get(s)
        if c and ("bid" in c and "ask" in c):
            bid = float(c["bid"])
            ask = float(c["ask"])
            denom = (ask + bid) / 2.0 if (ask > 0.0 and bid > 0.0) else 0.0
            spread = ((ask - bid) / denom * 100.0) if denom > 0.0 else 0.0
            return {"bid": bid, "ask": ask, "spread": round(spread, 4)}

        # 4) Fallback finale neutro
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}



def get_price(symbol: str) -> float:
    """Prezzo medio (cache 8s). Preferisce (bid+ask)/2 se disponibile."""
    c = _price_cache.get(symbol)
    if c and _now() - c.get("ts", 0.0) < PRICE_TTL and "price" in c:
        return float(c["price"])
    try:
        # Tenta prima il bookTicker (coerente con get_bid_ask)
        data = _get_json(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            {"symbol": symbol}, timeout=2.2, retries=1
        )
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        p = (bid + ask) / 2.0
        _price_cache[symbol] = {"ts": _now(), "price": p, "bid": bid, "ask": ask}
        return p
    except Exception:
        # Fallback su ticker/price
        try:
            data = _get_json(
                "https://api.binance.com/api/v3/ticker/price",
                {"symbol": symbol}, timeout=2.2, retries=1
            )
            p = float(data["price"])
            _price_cache[symbol] = {"ts": _now(), "price": p, **_price_cache.get(symbol, {})}
            return p
        except Exception as e:
            print(f"❌ Errore get_price({symbol}):", e)
            return float(_price_cache.get(symbol, {}).get("price", 0.0))


def get_symbol_tick_step(symbol: str) -> Tuple[float, float]:
    """Ritorna (tick_size, step_size) da exchangeInfo con cache 30 minuti.
       Fallback: (0.0001, 0.0001).
    """
    s = (symbol or "").upper()
    if not s:
        return 0.0001, 0.0001

    now = _now()
    cache = _tick_cache.get(s)
    if cache and (now - cache.get("ts", 0.0) < TICK_TTL_SEC):
        return float(cache["tick"]), float(cache["step"])

    try:
        data = _get_json(
            "https://api.binance.com/api/v3/exchangeInfo",
            {"symbol": s}, timeout=2.5, retries=1
        )

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
