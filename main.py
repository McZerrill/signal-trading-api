from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import os
from dotenv import load_dotenv

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica chiave da .env
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Funzioni indicatori
def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcola_atr(df, periodi=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift())
    df['L-PC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df['TR'].rolling(window=periodi).mean()

@app.get("/analyze")
def analyze(symbol: str):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-timeseries"
    headers = {
        "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    params = {
        "symbol": symbol,
        "interval": "30m",
        "range": "7d",
        "region": "US"
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    try:
        timestamps = data["timeseries"]["timestamp"]
        valori = data["timeseries"]["close"]
        df = pd.DataFrame(valori)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
    except:
        return {"segnale": "ERROR", "commento": f"Errore nel recupero dati per {symbol.upper()}"}

    print(f"📊 Ricevute {len(df)} candele per {symbol.upper()}")

    if len(df) < 60:
        return {"segnale": "ERROR", "commento": f"Dati insufficienti per {symbol.upper()}"}

    # Indicatori
    df["MA_9"] = df["close"].rolling(window=9).mean()
    df["MA_21"] = df["close"].rolling(window=21).mean()
    df["MA_100"] = df["close"].rolling(window=100).mean()
    df["RSI"] = calcola_rsi(df["close"])
    df["ATR"] = calcola_atr(df)

    ultimo = df.iloc[-1]
    penultimo = df.iloc[-2]

    close = ultimo["close"]
    ma9, ma21, ma100 = ultimo["MA_9"], ultimo["MA_21"], ultimo["MA_100"]
    rsi, atr = ultimo["RSI"], ultimo["ATR"]
    spread = (ultimo["high"] - ultimo["low"]) * 0.1

    dist_ma9 = abs(ma9 - ma100)
    dist_ma21 = abs(ma21 - ma100)
    media_distanza = (dist_ma9 + dist_ma21) / 2
    soglia_distanza = close * 0.01

    segnale = "HOLD"
    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)
    commento = f"RSI: {round(rsi,2)} | MA9: {round(ma9,2)} | MA21: {round(ma21,2)} | MA100: {round(ma100,2)} | ATR: {round(atr,2)}"

    if (
        penultimo["MA_9"] < penultimo["MA_21"] and
        ma9 > ma21 and ma9 > ma100 and ma21 > ma100 and
        rsi < 40 and media_distanza > soglia_distanza
    ):
        segnale = "BUY"
        tp = round(close + (atr + spread), 2)
        sl = round(close - (atr + spread * 0.5), 2)
        commento += f"\n→ Incrocio rialzista + RSI basso\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    elif (
        penultimo["MA_9"] > penultimo["MA_21"] and
        ma9 < ma21 and ma9 < ma100 and ma21 < ma100 and
        rsi > 60 and media_distanza > soglia_distanza
    ):
        segnale = "SELL"
        tp = round(close - (atr + spread), 2)
        sl = round(close + (atr + spread * 0.5), 2)
        commento += f"\n→ Incrocio ribassista + RSI alto\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    return {
        "segnale": segnale,
        "commento": commento,
        "prezzo": round(close, 2),
        "take_profit": tp,
        "stop_loss": sl
    }
