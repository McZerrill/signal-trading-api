from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

# Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Calcolo RSI
def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    media_gain = gain.rolling(window=periodi).mean()
    media_loss = loss.rolling(window=periodi).mean()
    rs = media_gain / media_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@app.get("/analyze")
def analyze(symbol: str):
    data = yf.Ticker(symbol)
    hist = data.history(period="7d", interval="15m")  # ✅ Timeframe 15 minuti

    if hist.empty or len(hist) < 100:
        return {
            "segnale": "ERROR",
            "commento": f"Dati insufficienti per {symbol.upper()}"
        }

    # Indicatori tecnici
    hist['MA_9'] = hist['Close'].rolling(window=9).mean()
    hist['MA_21'] = hist['Close'].rolling(window=21).mean()
    hist['MA_100'] = hist['Close'].rolling(window=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = (hist['High'] - hist['Low']).rolling(window=14).mean()

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    close = ultimo['Close']
    ma9 = ultimo['MA_9']
    ma21 = ultimo['MA_21']
    ma100 = ultimo['MA_100']
    rsi = ultimo['RSI']
    atr = ultimo['ATR']
    spread = (ultimo['High'] - ultimo['Low']) * 0.1  # Spread fittizio

    segnale = "HOLD"
    commento = f"RSI: {round(rsi,2)} | MA9: {round(ma9,2)} | MA21: {round(ma21,2)} | MA100: {round(ma100,2)} | ATR: {round(atr,2)}"

    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)

    # BUY → incrocio rialzista sopra MA100 + RSI basso
    if (
        penultimo['MA_9'] < penultimo['MA_21'] and
        ma9 > ma21 and
        ma9 > ma100 and
        ma21 > ma100 and
        rsi < 30
    ):
        segnale = "BUY"
        tp = round(close + (atr + spread), 2)
        sl = round(close - (atr + spread * 0.5), 2)
        commento += f" → Incrocio rialzista + RSI basso\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    # SELL → incrocio ribassista sotto MA100 + RSI alto
    elif (
        penultimo['MA_9'] > penultimo['MA_21'] and
        ma9 < ma21 and
        ma9 < ma100 and
        ma21 < ma100 and
        rsi > 70
    ):
        segnale = "SELL"
        tp = round(close - (atr + spread), 2)
        sl = round(close + (atr + spread * 0.5), 2)
        commento += f" → Incrocio ribassista + RSI alto\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    return {
        "segnale": segnale,
        "commento": commento,
        "prezzo": round(close, 2),
        "take_profit": tp,
        "stop_loss": sl
    }
