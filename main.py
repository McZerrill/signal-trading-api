from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd

app = FastAPI()

# Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inserisci qui la tua API key
API_KEY = "2685e9de941e427aaef496338cd43d8c"

# Funzione RSI
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
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&outputsize=5000&apikey={API_KEY}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return {"segnale": "ERROR", "commento": f"Dati non disponibili per {symbol.upper()}"}

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    if len(df) < 150:
        return {"segnale": "ERROR", "commento": f"Dati insufficienti per {symbol.upper()}"}

    df['MA_9'] = df['close'].rolling(window=9).mean()
    df['MA_21'] = df['close'].rolling(window=21).mean()
    df['MA_100'] = df['close'].rolling(window=100).mean()
    df['RSI'] = calcola_rsi(df['close'])
    df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()

    ultimo = df.iloc[-1]
    penultimo = df.iloc[-2]

    incrocio_buy = (penultimo['MA_9'] < penultimo['MA_21']) and (ultimo['MA_9'] > ultimo['MA_21'])
    sopra_ma100 = ultimo['MA_9'] > ultimo['MA_100'] and ultimo['MA_21'] > ultimo['MA_100']

    incrocio_sell = (penultimo['MA_9'] > penultimo['MA_21']) and (ultimo['MA_9'] < ultimo['MA_21'])
    sotto_ma100 = ultimo['MA_9'] < ultimo['MA_100'] and ultimo['MA_21'] < ultimo['MA_100']

    close = ultimo['close']
    atr = ultimo['ATR']
    rsi = ultimo['RSI']

    tp = round(close + atr * 2, 2)  # target profit
    sl = round(close - atr * 1.5, 2)  # stop loss

    segnale = "HOLD"
    commento = f"RSI: {round(rsi,2)} | MA9: {round(ultimo['MA_9'],2)} | MA21: {round(ultimo['MA_21'],2)} | MA100: {round(ultimo['MA_100'],2)} | ATR: {round(atr,2)}"

    if incrocio_buy and sopra_ma100 and rsi < 35:
        segnale = "BUY"
        commento += f"\n📈 Incrocio rialzista + sopra MA100 + RSI basso"
    elif incrocio_sell and sotto_ma100 and rsi > 65:
        segnale = "SELL"
        commento += f"\n📉 Incrocio ribassista + sotto MA100 + RSI alto"

    return {
        "segnale": segnale,
        "commento": commento,
        "prezzo": round(close, 2),
        "take_profit": tp,
        "stop_loss": sl
    }
