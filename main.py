from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

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
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcola_atr(df, periodi=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df['TR'].rolling(window=periodi).mean()

def calcola_tii(serie, periodo_ma=60, periodo_tii=30):
    ma = serie.rolling(window=periodo_ma).mean()
    deviazione = abs(serie - ma)
    media_deviazione = deviazione.rolling(window=periodo_tii).mean()
    return 100 - (100 / (1 + (serie - ma) / media_deviazione))

def analizza_trend(hist):
    hist['MA_9'] = hist['Close'].rolling(window=9).mean()
    hist['MA_21'] = hist['Close'].rolling(window=21).mean()
    hist['MA_100'] = hist['Close'].rolling(window=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = calcola_atr(hist)
    hist['TII'] = calcola_tii(hist['Close'])

    ultimo = hist.iloc[-1]
    ma9 = ultimo['MA_9']
    ma100 = ultimo['MA_100']
    rsi = ultimo['RSI']
    tii = ultimo['TII']

    incrocio_buy = (
        hist['MA_9'].iloc[-4] < hist['MA_100'].iloc[-4] and
        ma9 > ma100 and
        (ma9 - ma100) > (hist['MA_9'].iloc[-4] - hist['MA_100'].iloc[-4])
    )

    incrocio_sell = (
        hist['MA_9'].iloc[-4] > hist['MA_100'].iloc[-4] and
        ma9 < ma100 and
        (ma100 - ma9) > (hist['MA_100'].iloc[-4] - hist['MA_9'].iloc[-4])
    )

    if incrocio_buy:
        return "BUY", hist
    elif incrocio_sell:
        return "SELL", hist
    else:
        return "HOLD", hist

@app.get("/analyze")
def analyze(symbol: str):
    data = yf.Ticker(symbol)

    try:
        hist_15m = data.history(period="7d", interval="15m")
        if hist_15m.empty or len(hist_15m) < 100:
            raise Exception("Dati insufficienti 15m")
        segnale_15m, hist_15m = analizza_trend(hist_15m)

        hist_5m = data.history(period="1d", interval="5m")
        if hist_5m.empty or len(hist_5m) < 100:
            raise Exception("Dati insufficienti 5m")
        segnale_5m, hist_5m = analizza_trend(hist_5m)

        ultimo = hist_15m.iloc[-1]
        close = ultimo['Close']
        atr = ultimo['ATR']
        spread = (ultimo['High'] - ultimo['Low']) * 0.1
        tp = round(close + (atr + spread), 2) if segnale_15m == "BUY" else round(close - (atr + spread), 2)
        sl = round(close - (atr * 0.8), 2) if segnale_15m == "BUY" else round(close + (atr * 0.8), 2)

        rsi = round(ultimo['RSI'], 2)
        ma9 = round(ultimo['MA_9'], 2)
        ma21 = round(ultimo['MA_21'], 2)
        ma100 = round(ultimo['MA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        tii = round(ultimo['TII'], 2)

        if segnale_15m == segnale_5m and segnale_15m != "HOLD":
            commento = (
                f"✅ {segnale_15m} confermato su 5m e 15m\n"
                f"RSI: {rsi} | MA9: {ma9} | MA21: {ma21} | MA100: {ma100} | ATR: {atr} | TII: {tii}\n"
                f"🎯 TP: {tp} | 🛡️ SL: {sl}"
            )
            return {
                "segnale": segnale_15m,
                "commento": commento,
                "prezzo": round(close, 2),
                "take_profit": tp,
                "stop_loss": sl
            }

        # HOLD
        commento = (
            f"Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}\n"
            f"RSI: {rsi} | MA9: {ma9} | MA21: {ma21} | MA100: {ma100} | ATR: {atr} | TII: {tii}"
        )
        tp = round(close * 1.02, 2)
        sl = round(close * 0.98, 2)
        return {
            "segnale": "HOLD",
            "commento": commento,
            "prezzo": round(close, 2),
            "take_profit": tp,
            "stop_loss": sl
        }

    except:
        return {
            "segnale": "ERROR",
            "commento": f"Dati insufficienti per {symbol.upper()}",
            "prezzo": 0.0,
            "take_profit": 0.0,
            "stop_loss": 0.0
        }