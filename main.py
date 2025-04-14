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

# Carica API key da .env
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Funzione per calcolare RSI
def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Calcolo ATR classico
def calcola_atr(df, periodi=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift())
    df['L-PC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df['TR'].rolling(window=periodi).mean()

@app.get("/analyze")
def analyze(symbol: str):
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=30&count=100&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("s") != "ok":
        return {"segnale": "ERROR", "commento": f"Errore nel recupero dati per {symbol.upper()}"}

    df = pd.DataFrame({
        "close": data["c"],
        "high": data["h"],
        "low": data["l"],
        "open": data["o"]
    })

    if len(df) < 100:
        return {"segnale": "ERROR", "commento": f"Dati insufficienti per {symbol.upper()}"}

    # Indicatori
    df['MA_9'] = df['close'].rolling(window=9).mean()
    df['MA_21'] = df['close'].rolling(window=21).mean()
    df['MA_100'] = df['close'].rolling(window=100).mean()
    df['RSI'] = calcola_rsi(df['close'])
    df['ATR'] = calcola_atr(df)

    ultimo = df.iloc[-1]
    penultimo = df.iloc[-2]

    close = ultimo['close']
    ma9, ma21, ma100 = ultimo['MA_9'], ultimo['MA_21'], ultimo['MA_100']
    rsi, atr = ultimo['RSI'], ultimo['ATR']
    spread = (ultimo['high'] - ultimo['low']) * 0.1

    dist_ma9 = abs(ma9 - ma100)
    dist_ma21 = abs(ma21 - ma100)
    media_distanza = (dist_ma9 + dist_ma21) / 2
    soglia_distanza = close * 0.01

    segnale = "HOLD"
    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)
    commento = f"RSI: {round(rsi,2)} | MA9: {round(ma9,2)} | MA21: {round(ma21,2)} | MA100: {round(ma100,2)} | ATR: {round(atr,2)}"

    if (
        penultimo['MA_9'] < penultimo['MA_21'] and
        ma9 > ma21 and ma9 > ma100 and ma21 > ma100 and
        rsi < 40 and media_distanza > soglia_distanza
    ):
        segnale = "BUY"
        tp = round(close + (atr + spread), 2)
        sl = round(close - (atr + spread * 0.5), 2)
        commento += f"\n→ Incrocio rialzista + RSI basso\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    elif (
        penultimo['MA_9'] > penultimo['MA_21'] and
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
