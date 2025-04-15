from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Enable CORS
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


def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def genera_grafico_base64(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df['Close'], label='Close', linewidth=1.5)
    ax.plot(df['EMA_9'], label='EMA 9', linestyle='--')
    ax.plot(df['EMA_21'], label='EMA 21', linestyle='--')
    ax.plot(df['EMA_100'], label='EMA 100', linestyle='--')
    ax.legend()
    ax.set_title('Prezzo e Medie Mobili')
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    data = yf.Ticker(symbol)
    hist = data.history(period="7d", interval="15m")

    if hist.empty or len(hist) < 100:
        return SignalResponse(
            segnale="ERROR",
            commento=f"Dati insufficienti per {symbol.upper()}",
            prezzo=0.0,
            take_profit=0.0,
            stop_loss=0.0,
        )

    hist['EMA_9'] = hist['Close'].ewm(span=9, adjust=False).mean()
    hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()
    hist['EMA_100'] = hist['Close'].ewm(span=100, adjust=False).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    close = ultimo['Close']
    ema9 = ultimo['EMA_9']
    ema21 = ultimo['EMA_21']
    ema100 = ultimo['EMA_100']
    rsi = ultimo['RSI']
    spread = (ultimo['High'] - ultimo['Low']) * 0.1

    segnale = "HOLD"
    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)

    commento = (
        f"RSI: {round(rsi, 2)} | EMA9: {round(ema9, 2)} | EMA21: {round(ema21, 2)} | EMA100: {round(ema100, 2)}"
    )

    # BUY
    if ema9 > ema21 > ema100 and penultimo['EMA_21'] < penultimo['EMA_100']:
        segnale = "BUY"
        tp = round(close + (spread * 4), 2)
        sl = round(close - (spread * 3), 2)
        commento += "\n✅ Incrocio rialzista e allargamento medie"

    # SELL
    elif ema9 < ema21 < ema100 and penultimo['EMA_21'] > penultimo['EMA_100']:
        segnale = "SELL"
        tp = round(close - (spread * 4), 2)
        sl = round(close + (spread * 3), 2)
        commento += "\n⚠️ Incrocio ribassista e allargamento medie"

    grafico = genera_grafico_base64(hist)

    return SignalResponse(
        segnale=segnale,
        commento=commento,
        prezzo=round(close, 2),
        take_profit=tp,
        stop_loss=sl,
        graficoBase64=grafico
    )


@app.get("/hotassets")
def hot_assets():
    symbols = ["AAPL", "TSLA", "GOOG", "MSFT", "NVDA", "BTC-USD", "ETH-USD", "AMZN", "NFLX", "META"]
    hot = []

    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period="2d", interval="15m")
            if hist.empty or len(hist) < 100:
                continue
            hist['EMA_9'] = hist['Close'].ewm(span=9, adjust=False).mean()
            hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()
            hist['EMA_100'] = hist['Close'].ewm(span=100, adjust=False).mean()
            hist['RSI'] = calcola_rsi(hist['Close'])
            ultimo = hist.iloc[-1]

            rsi = ultimo['RSI']
            ema9 = ultimo['EMA_9']
            ema21 = ultimo['EMA_21']
            ema100 = ultimo['EMA_100']

            if rsi < 30 or rsi > 70 or (ema9 > ema21 > ema100 or ema9 < ema21 < ema100):
                hot.append({
                    "symbol": symbol,
                    "rsi": round(rsi, 2),
                    "ema9": round(ema9, 2),
                    "ema21": round(ema21, 2),
                    "ema100": round(ema100, 2)
                })
        except:
            continue

    return sorted(hot, key=lambda x: abs(x['rsi'] - 50), reverse=True)[:10]
