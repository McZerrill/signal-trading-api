# binance_api.py

import os
import time
import requests
import pandas as pd
from typing import Optional
from binance.client import Client

# Inizializza client Binance dalle variabili d'ambiente
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Cache simboli (valida 15 minuti)
_symbol_cache = {"time": 0, "data": []}

def get_best_symbols(limit: int = 25) -> list[str]:
    """Restituisce i simboli USDT più liquidi ordinati per volume."""
    now = time.time()
    if now - _symbol_cache["time"] < 900:
        return _symbol_cache["data"]

    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        usdt_pairs = [
            d for d in data
            if d["symbol"].endswith("USDT") and not any(x in d["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
        ]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)
        top_symbols = [d["symbol"] for d in sorted_pairs[:limit]]

        _symbol_cache["time"] = now
        _symbol_cache["data"] = top_symbols
        return top_symbols

    except Exception as e:
        print("❌ Errore nel recupero simboli:", e)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT",
            "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "MATICUSDT"
        ]

def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> pd.DataFrame:
    """Restituisce un DataFrame OHLCV da Binance."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time

    try:
        klines = client.get_klines(**params)
    except Exception as e:
        print(f"❌ Errore caricamento {symbol}-{interval}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df
