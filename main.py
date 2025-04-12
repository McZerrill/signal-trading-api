from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

# Abilita CORS per tutte le origini (puoi restringere a specifici domini)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    hist = data.history(period="6mo")

    if hist.empty or len(hist) < 200:
        return {"segnale": "ERROR", "commento": f"Dati insufficienti per {symbol.upper()}"}

    # Calcolo medie mobili
    hist['MA_9'] = hist['Close'].rolling(window=9).mean()
    hist['MA_21'] = hist['Close'].rolling(window=21).mean()
    hist['MA_200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    # Condizione incrocio: MA9 e MA21 passano da sotto a sopra la MA200
    incrocio_oggi = ultimo['MA_9'] > ultimo['MA_200'] and ultimo['MA_21'] > ultimo['MA_200']
    incrocio_ieri = penultimo['MA_9'] < penultimo['MA_200'] and penultimo['MA_21'] < penultimo['MA_200']
    incrocio = incrocio_oggi and incrocio_ieri

    rsi = ultimo['RSI']
    segnale = "HOLD"
    commento = f"RSI: {round(rsi, 2)} | MA9: {round(ultimo['MA_9'], 2)} | MA21: {round(ultimo['MA_21'], 2)} | MA200: {round(ultimo['MA_200'], 2)}"

    if incrocio and rsi < 30:
        segnale = "BUY"
        commento += " → Incrocio rialzista + RSI basso"
    elif rsi > 70:
        segnale = "SELL"
        commento += " → RSI alto (ipervenduto)"

    return {
        "segnale": segnale,
        "commento": commento
    }
