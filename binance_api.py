# binance_api.py
# -----------------------------------------------------------------------------
# Helper Binance con cache breve e timeout molto rapidi (pensati per client 3s)
# - Connection pooling via requests.Session()
# - Fallback immediato su cache se la rete è lenta
# - get_klines via python-binance (sincrono, stabile)
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
BASE_URL = "https://api.binance.com"
QUOTE_SUFFIXES: Tuple[str, str] = ("USDT", "USDC")
MODALITA_TEST: bool = True

# Binance client (per klines)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# HTTP session (keep-alive)
_session = requests.Session()
_session.headers.update({"User-Agent": "SegnaliDiBorsa/1.0"})

# Cache simboli (60s)
_symbol_cache: Dict[str, Any] = {"time": 0.0, "data": []}

# Cache tick/step (30 min)
_tick_cache: Dict[str, Dict[str, float]] = {}
TICK_TTL_SEC: float = 1800.0

# Cache prezzo/bid-ask (8s)
_price_cache: Dict[str, Dict[str, float]] = {}
PRICE_TTL: float = 8.0

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now() -> float:
    return time.time()

def _ok_symbol(sym: str) -> bool:
    if not any(sym.endswith(q) for q in QUOTE_SUFFIXES):
        return False
    return not any(x in sym for x in ("UP", "DOWN", "BULL", "BEAR"))

def _get_json(path: str, params: Optional[dict] = None, timeout: float = 1.25) -> dict:
    """GET con timeout cortissimo; nessun retry per rientrare nei 3s lato app."""
    url = f"{BASE_URL}{path}"
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
def get_best_symbols(limit: int = 80):
    """Top simboli per volume 24h (USDT/USDC), cache 60s."""
    now = _now()
    if now - _symbol_cache["time"] < 60 and _symbol_cache["data"]:
        return _symbol_cache["data"][:limit]

    try:
        data = _get_json("/api/v3/ticker/24hr", timeout=1.8)  # veloce ma sufficiente
        filtered = []
        min_vol = 1_000_000 if MODALITA_TEST else 5_000_000
        for d in data:
            sym = d.get("symbol", "")
            if not sym or not _ok_symbol(sym):
                continue
            try:
                qv = float(d.get("quoteVolume", 0.0))
            except Exception:
                qv = 0.0
            if qv >= min_vol:
                filtered.append((sym, qv))
        filtered.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in filtered[:limit]] or [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "BTCUSDC", "ETHUSDC", "BNBUSDC"
        ]
        _symbol_cache["time"] = now
        _symbol_cache["data"] = symbols
        return symbols
    except Exception as e:
        print("❌ Errore get_best_symbols:", e)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "BTCUSDC", "ETHUSDC", "BNBUSDC",
            "SOLUSDT", "XRPUSDT", "ADAUSDT"
        ][:limit]

def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> pd.DataFrame:
    """Klines → DataFrame OHLCV (float) con DatetimeIndex UTC."""
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
    """Bid/ask reali con cache 8s; timeout molto corto; fallback su cache."""
    s = (symbol or "").upper()
    if not s:
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}
    now = _now()

    c = _price_cache.get(s)
    if c and (now - float(c.get("ts", 0.0)) < PRICE_TTL) and "bid" in c and "ask" in c:
        bid, ask = float(c["bid"]), float(c["ask"])
        denom = (ask + bid) / 2.0 if (ask > 0.0 and bid > 0.0) else 0.0
        spread = ((ask - bid) / denom * 100.0) if denom > 0.0 else 0.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}

    try:
        data = _get_json("/api/v3/ticker/bookTicker", {"symbol": s}, timeout=1.25)
        bid = float(data["bidPrice"]); ask = float(data["askPrice"])
        _price_cache[s] = {"ts": now, "bid": bid, "ask": ask, "price": (bid + ask) / 2.0}
        denom = (ask + bid) / 2.0
        spread = ((ask - bid) / denom * 100.0) if denom > 0 else 0.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}
    except Exception as e:
        print(f"❌ Errore get_bid_ask({s}): {e}")
        # Fallback su cache se presente
        c = _price_cache.get(s)
        if c and "bid" in c and "ask" in c:
            bid, ask = float(c["bid"]), float(c["ask"])
            denom = (ask + bid) / 2.0 if (ask > 0.0 and bid > 0.0) else 0.0
            spread = ((ask - bid) / denom * 100.0) if denom > 0.0 else 0.0
            return {"bid": bid, "ask": ask, "spread": round(spread, 4)}
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}

def get_price(symbol: str) -> float:
    """Prezzo medio (cache 8s). Priorità: /ticker/price (rapidissima), poi bookTicker."""
    s = (symbol or "").upper()
    if not s:
        return 0.0
    c = _price_cache.get(s)
    if c and (_now() - c.get("ts", 0.0) < PRICE_TTL) and "price" in c:
        return float(c["price"])
    # 1) endpoint più veloce
    try:
        data = _get_json("/api/v3/ticker/price", {"symbol": s}, timeout=1.0)
        p = float(data["price"])
        # prova a riempire anche bid/ask se già in cache
        prev = _price_cache.get(s, {})
        _price_cache[s] = {"ts": _now(), "price": p, "bid": prev.get("bid", p), "ask": prev.get("ask", p)}
        return p
    except Exception:
        # 2) fallback su bookTicker
        try:
            data = _get_json("/api/v3/ticker/bookTicker", {"symbol": s}, timeout=1.25)
            bid = float(data["bidPrice"]); ask = float(data["askPrice"])
            p = (bid + ask) / 2.0
            _price_cache[s] = {"ts": _now(), "price": p, "bid": bid, "ask": ask}
            return p
        except Exception as e:
            print(f"❌ Errore get_price({s}): {e}")
            return float(_price_cache.get(s, {}).get("price", 0.0))

def get_symbol_tick_step(symbol: str) -> Tuple[float, float]:
    """(tick_size, step_size) con cache 30 min. Fallback (0.0001, 0.0001)."""
    s = (symbol or "").upper()
    if not s:
        return 0.0001, 0.0001
    now = _now()
    cache = _tick_cache.get(s)
    if cache and (now - cache.get("ts", 0.0) < TICK_TTL_SEC):
        return float(cache["tick"]), float(cache["step"])
    try:
        data = _get_json("/api/v3/exchangeInfo", {"symbol": s}, timeout=1.8)
        items = data.get("symbols") or []
        if not items:
            raise ValueError(f"exchangeInfo vuoto per {s}")
        info = items[0]
        tick = 0.0001
        step = 0.0001
        for f in info.get("filters", []):
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                tick = float(f.get("tickSize", tick))
            elif t == "LOT_SIZE":
                step = float(f.get("stepSize", step))
        _tick_cache[s] = {"tick": tick, "step": step, "ts": now}
        return tick, step
    except Exception as e:
        print(f"⚠️ Errore get_symbol_tick_step({s}): {e}")
        return 0.0001, 0.0001
