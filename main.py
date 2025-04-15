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

# RSI
def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ATR classico
def calcola_atr(df, periodi=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = df['TR'].rolling(window=periodi).mean()
    return atr

@app.get("/analyze")
def analyze(symbol: str):
    data = yf.Ticker(symbol)
    hist = data.history(period="7d", interval="15m")

    if hist.empty or len(hist) < 100:
        return {
            "segnale": "ERROR",
            "commento": f"Dati insufficienti per {symbol.upper()}"
        }

    hist['MA_9'] = hist['Close'].rolling(window=9).mean()
    hist['MA_21'] = hist['Close'].rolling(window=21).mean()
    hist['MA_100'] = hist['Close'].rolling(window=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = calcola_atr(hist)

    ultimo = hist.iloc[-1]
    close = ultimo['Close']
    ma9 = ultimo['MA_9']
    ma21 = ultimo['MA_21']
    ma100 = ultimo['MA_100']
    rsi = ultimo['RSI']
    atr = ultimo['ATR']
    spread = (ultimo['High'] - ultimo['Low']) * 0.1

    segnale = "HOLD"
    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)

    commento = (
        f"RSI: {round(rsi, 2)} | MA9: {round(ma9, 2)} | "
        f"MA21: {round(ma21, 2)} | MA100: {round(ma100, 2)} | ATR: {round(atr, 2)}"
    )

    # BUY forte: incrocio MA9 sopra MA100 + divergenza crescente
    if (
        hist['MA_9'].iloc[-4] < hist['MA_100'].iloc[-4] and
        ma9 > ma100 and
        (ma9 - ma100) > (hist['MA_9'].iloc[-4] - hist['MA_100'].iloc[-4]) and
        rsi < 65
    ):
        segnale = "BUY"
        tp = round(close + (atr + spread), 2)
        sl = round(close - (atr * 0.8), 2)
        commento += f"\n✅ Incrocio MA + divergenza in espansione → BUY\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    # SELL forte: incrocio MA9 sotto MA100 + divergenza crescente
    elif (
        hist['MA_9'].iloc[-4] > hist['MA_100'].iloc[-4] and
        ma9 < ma100 and
        (ma100 - ma9) > (hist['MA_100'].iloc[-4] - hist['MA_9'].iloc[-4]) and
        rsi > 40
    ):
        segnale = "SELL"
        tp = round(close - (atr + spread), 2)
        sl = round(close + (atr * 0.8), 2)
        commento += f"\n⚠️ Incrocio ribassista + divergenza espansa → SELL\n🎯 TP: {tp} | 🛡️ SL: {sl}"

    return {
        "segnale": segnale,
        "commento": commento,
        "prezzo": round(close, 2),
        "take_profit": tp,
        "stop_loss": sl
    }