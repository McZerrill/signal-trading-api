import time 
import requests
import certifi
requests.adapters.DEFAULT_CA_BUNDLE_PATH = certifi.where()
import pandas as pd
from typing import Optional
from binance.client import Client
import os

# Inizializza il client Binance con chiavi da variabili d'ambiente
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Cache dei simboli
_symbol_cache = {"time": 0, "data": []}
_tick_cache: dict[str, float] = {}   # cache per tick per simbolo

# ✅ Modalità test attiva
MODALITA_TEST = True


def get_best_symbols(limit=80):
    now = time.time()
    if now - _symbol_cache["time"] < 60:  # cache breve: ogni 60s
        return _symbol_cache["data"]

    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Filtro su simboli USDC, no token a leva, volume > 1M
        filtered = [
            d for d in data
            if d["symbol"].endswith("USDC")
            and not any(x in d["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
            and float(d["quoteVolume"]) > (1_000_000 if MODALITA_TEST else 5_000_000)
        ]

        # Ordina per volume decrescente
        sorted_pairs = sorted(filtered, key=lambda x: float(x["quoteVolume"]), reverse=True)
        top_symbols = [d["symbol"] for d in sorted_pairs[:limit]]

        print(f"✅ {len(top_symbols)} simboli trovati con volume > 1M USDC")
        _symbol_cache["time"] = now
        _symbol_cache["data"] = top_symbols
        return top_symbols

    except Exception as e:
        print("❌ Errore nel recupero simboli Binance:", e)
        return [
            "BTCUSDC", "ETHUSDC", "BNBUSDC", "XRPUSDC", "ADAUSDC",
            "SOLUSDC", "AVAXUSDC", "DOTUSDC", "DOGEUSDC", "MATICUSDC"
        ]

def _safe_tick_from_precision(precision: int | None, default: float = 0.0001) -> float:
    try:
        if precision is None or precision < 0 or precision > 20:
            return default
        return float(10 ** (-precision))
    except Exception:
        return default

def get_symbol_tick_step(symbol: str) -> float:
    """
    Ritorna il tick (PRICE_FILTER.tickSize) di un simbolo spot Binance.
    Usa cache in-memoria; in caso di errore ritorna un fallback 0.0001.
    """
    # Cache hit
    if symbol in _tick_cache:
        return _tick_cache[symbol]

    try:
        # Preferiamo l'endpoint REST (evita dipendenze dalla versione del client)
        url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        syms = data.get("symbols") or []
        if not syms:
            # Fallback: prova senza parametro e filtra localmente (più pesante)
            url_all = "https://api.binance.com/api/v3/exchangeInfo"
            resp2 = requests.get(url_all, timeout=15)
            resp2.raise_for_status()
            data2 = resp2.json()
            syms = [s for s in (data2.get("symbols") or []) if s.get("symbol") == symbol]
            if not syms:
                _tick_cache[symbol] = 0.0001
                return _tick_cache[symbol]

        s0 = syms[0]
        # 1) Cerca PRICE_FILTER.tickSize
        for f in s0.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick_str = f.get("tickSize")
                try:
                    tick = float(tick_str)
                    if tick > 0:
                        _tick_cache[symbol] = tick
                        return tick
                except Exception:
                    pass

        # 2) Fallback: deduci dal "quotePrecision" o "baseAssetPrecision"
        qp = s0.get("quotePrecision")
        bp = s0.get("baseAssetPrecision")
        tick = _safe_tick_from_precision(qp if isinstance(qp, int) else bp)
        _tick_cache[symbol] = tick
        return tick

    except Exception:
        # Ultimo fallback robusto
        _tick_cache[symbol] = 0.0001
        return _tick_cache[symbol]

def get_symbol_lot_step(symbol: str) -> float:
    """
    Ritorna lo step di quantità (LOT_SIZE.stepSize). Utile per arrotondare le QTY.
    """
    cache_key = f"LOT:{symbol}"
    if cache_key in _tick_cache:
        return _tick_cache[cache_key]
    try:
        url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        syms = data.get("symbols") or []
        if syms:
            for f in syms[0].get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    step = float(f.get("stepSize", "0"))
                    if step > 0:
                        _tick_cache[cache_key] = step
                        return step
    except Exception:
        pass
    _tick_cache[cache_key] = 0.000001
    return _tick_cache[cache_key]


def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    
    df.dropna(inplace=True)

    return df


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def get_bid_ask(symbol: str) -> dict:
    """
    Recupera bid/ask reali dal book Binance e calcola lo spread percentuale.
    Ritorna spread=9999.0 in caso di dati non validi, così il chiamante filtra lo strumento.
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        data = requests.get(url, timeout=5).json()

        bid = _safe_float(data.get("bidPrice"))
        ask = _safe_float(data.get("askPrice"))

        # Se prezzi non validi o non positivi → segnala come "spread enorme"
        if bid <= 0.0 or ask <= 0.0:
            return {"bid": 0.0, "ask": 0.0, "spread": 9999.0}

        mid = (ask + bid) / 2.0
        if mid <= 0.0:
            return {"bid": bid, "ask": ask, "spread": 9999.0}

        spread = (ask - bid) / mid * 100.0
        return {"bid": bid, "ask": ask, "spread": round(spread, 4)}

    except Exception:
        # In qualunque errore di rete/parsing, evita crash del chiamante
        return {"bid": 0.0, "ask": 0.0, "spread": 9999.0}

