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

def get_best_symbols(limit=50):
    now = time.time()
    if now - _symbol_cache["time"] < 900:  # 15 minuti cache
        return _symbol_cache["data"]

    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        # Filtro su simboli USDT, no token a leva, volume > 5M
        filtered = [
            d for d in data
            if d["symbol"].endswith("USDT")
            and not any(x in d["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
            and float(d["quoteVolume"]) > 5_000_000
        ]

        # Ordina per volume decrescente
        sorted_pairs = sorted(filtered, key=lambda x: float(x["quoteVolume"]), reverse=True)
        top_symbols = [d["symbol"] for d in sorted_pairs[:limit]]

        print(f"✅ {len(top_symbols)} simboli trovati con volume > 5M USDT")
        _symbol_cache["time"] = now
        _symbol_cache["data"] = top_symbols
        return top_symbols

    except Exception as e:
        print("❌ Errore nel recupero simboli Binance:", e)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "SOLUSDT", "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "MATICUSDT"
        ]

# Funzione per ottenere lo storico delle candele da Binance
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

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Filtro su volatilità (ATR %) e spread candle
    df["ATR"] = (df["high"] - df["low"]).rolling(window=14).mean()
    df.dropna(inplace=True)
    if len(df) < 15:
        return pd.DataFrame()

    atr_pct = df["ATR"].iloc[-1] / df["close"].iloc[-1]
    spread_pct = (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1]

    # Elimina asset troppo piatti o troppo volatili, oppure con spread eccessivo
    #if not (0.001 <= atr_pct <= 0.02):
    #    print(f"⛔ Scartato {symbol} per ATR {atr_pct:.2%}")
        #return pd.DataFrame()
    #if spread_pct > 0.01:
        #print(f"⛔ Scartato {symbol} per spread {spread_pct:.2%}")
        #return pd.DataFrame()

    #return df
