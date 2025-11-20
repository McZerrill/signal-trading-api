import time 
import requests
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


def get_bid_ask(symbol: str) -> dict:
    """
    Recupera bid/ask reali dal book Binance e calcola lo spread percentuale.
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        response = requests.get(url, timeout=5)
        data = response.json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        spread = (ask - bid) / ((ask + bid) / 2) * 100
        return {
            "bid": bid,
            "ask": ask,
            "spread": round(spread, 4)
        }
    except Exception as e:
        print(f"❌ Errore get_bid_ask per {symbol}:", e)
        return {
            "bid": 0.0,
            "ask": 0.0,
            "spread": 0.0
        }
def get_symbol_tick_step(symbol: str) -> dict:
    """
    Ritorna i valori reali di tickSize (prezzo) e stepSize (quantità)
    per il simbolo su Binance, usando il client ufficiale.
    """
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            print(f"⚠️ Nessuna symbol_info per {symbol}")
            return {"tickSize": 0.0, "stepSize": 0.0}

        tick_size = 0.0
        step_size = 0.0

        for f in info.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0.0))
            elif f.get("filterType") == "LOT_SIZE":
                step_size = float(f.get("stepSize", 0.0))

        return {
            "tickSize": tick_size,
            "stepSize": step_size,
        }

    except Exception as e:
        print(f"❌ Errore get_symbol_tick_step per {symbol}: {e}")
        return {"tickSize": 0.0, "stepSize": 0.0}
