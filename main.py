from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SignalResponse(BaseModel):
    segnale: str
    commento: str
    prezzo: float
    take_profit: float
    stop_loss: float
    graficoBase64: str | None = None

class HotCrypto(BaseModel):
    symbol: str
    segnali_buy: int
    segnali_sell: int
    rsi: float
    macd: float

# === Indicatori ===
def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcola_macd(serie):
    ema12 = serie.ewm(span=12, adjust=False).mean()
    ema26 = serie.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line

def calcola_ema(df):
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_21'] = df['Close'].ewm(span=21).mean()
    df['EMA_100'] = df['Close'].ewm(span=100).mean()
    return df

# === Hot Assets Cripto ===
@app.get("/hotassets", response_model=list[HotCrypto])
def hot_assets():
    cripto_symbols = [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD",
        "DOGE-USD", "AVAX-USD", "MATIC-USD", "DOT-USD", "LINK-USD", "UNI1-USD",
        "XLM-USD", "ATOM-USD", "NEAR-USD", "APT-USD", "OP-USD", "LTC-USD"
    ]
    cutoff_time = datetime.utcnow() - timedelta(hours=2)
    risultati = []

    for symbol in cripto_symbols:
        try:
            df = yf.Ticker(symbol).history(period="2d", interval="15m")
            if df.empty or len(df) < 10:
                continue

            df = df.copy()
            df = calcola_ema(df)
            df['RSI'] = calcola_rsi(df['Close'])
            df['MACD'] = calcola_macd(df['Close'])
            df = df.dropna()

            segnali_buy = 0
            segnali_sell = 0

            for i in range(1, len(df)):
                if df.index[i] < cutoff_time:
                    continue
                # BUY: EMA9 > EMA21 > EMA100 e RSI > 50 e MACD > 0
                if (
                    df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] > df['EMA_100'].iloc[i]
                    and df['RSI'].iloc[i] > 50
                    and df['MACD'].iloc[i] > 0
                ):
                    segnali_buy += 1
                # SELL: EMA9 < EMA21 < EMA100 e RSI < 50 e MACD < 0
                elif (
                    df['EMA_9'].iloc[i] < df['EMA_21'].iloc[i] < df['EMA_100'].iloc[i]
                    and df['RSI'].iloc[i] < 50
                    and df['MACD'].iloc[i] < 0
                ):
                    segnali_sell += 1

            if segnali_buy > 0 or segnali_sell > 0:
                ultimi = df.iloc[-1]
                risultati.append(HotCrypto(
                    symbol=symbol,
                    segnali_buy=segnali_buy,
                    segnali_sell=segnali_sell,
                    rsi=round(ultimi['RSI'], 2),
                    macd=round(ultimi['MACD'], 4)
                ))

        except Exception as e:
            print(f"Errore su {symbol}: {e}")
            continue

    # Ordina per somma di segnali BUY/SELL
    risultati = sorted(risultati, key=lambda x: (x.segnali_buy + x.segnali_sell), reverse=True)
    return risultati[:10]
