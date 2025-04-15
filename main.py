from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Indicatori
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

# Analizza MA, RSI, TII per un timeframe
def analizza_trend(hist):
    hist['MA_9'] = hist['Close'].rolling(window=9).mean()
    hist['MA_100'] = hist['Close'].rolling(window=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
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

    conferma_rsi = rsi < 65 if incrocio_buy else rsi > 40
    conferma_tii = tii > 60 if incrocio_buy else tii < 40

    if incrocio_buy:
        return "BUY", rsi, tii, conferma_rsi, conferma_tii
    elif incrocio_sell:
        return "SELL", rsi, tii, conferma_rsi, conferma_tii
    else:
        return "HOLD", rsi, tii, False, False

@app.get("/analyze")
def analyze(symbol: str):
    data = yf.Ticker(symbol)

    # Dati 15m
    hist_15m = data.history(period="7d", interval="15m")
    if hist_15m.empty or len(hist_15m) < 100:
        return {
            "segnale": "ERROR",
            "commento": f"Dati insufficienti (15m) per {symbol.upper()}"
        }
    segnale_15m, rsi_15m, tii_15m, conf_rsi_15m, conf_tii_15m = analizza_trend(hist_15m)

    # Dati 5m
    hist_5m = data.history(period="1d", interval="5m")
    if hist_5m.empty or len(hist_5m) < 100:
        return {
            "segnale": "ERROR",
            "commento": f"Dati insufficienti (5m) per {symbol.upper()}"
        }
    segnale_5m, rsi_5m, tii_5m, conf_rsi_5m, conf_tii_5m = analizza_trend(hist_5m)

    # Verifica concordanza segnale tra timeframe
    if segnale_15m == segnale_5m and segnale_15m != "HOLD":
        ultimo = hist_15m.iloc[-1]
        close = ultimo['Close']
        atr = calcola_atr(hist_15m).iloc[-1]
        spread = (ultimo['High'] - ultimo['Low']) * 0.1
        tp = round(close + (atr + spread), 2) if segnale_15m == "BUY" else round(close - (atr + spread), 2)
        sl = round(close - (atr * 0.8), 2) if segnale_15m == "BUY" else round(close + (atr * 0.8), 2)

        rsi_ok = "ok" if conf_rsi_15m and conf_rsi_5m else "non confermato"
        tii_ok = "forte" if conf_tii_15m and conf_tii_5m else "non confermato"

        commento = (
            f"✅ Segnale {segnale_15m} confermato da 5m e 15m\n"
            f"RSI medio: {round((rsi_15m + rsi_5m)/2, 2)} ({rsi_ok}) | "
            f"TII medio: {round((tii_15m + tii_5m)/2, 2)} ({tii_ok})\n"
            f"🎯 TP: {tp} | 🛡️ SL: {sl}"
        )

        return {
            "segnale": segnale_15m,
            "commento": commento,
            "prezzo": round(close, 2),
            "take_profit": tp,
            "stop_loss": sl
        }

    # Altrimenti: HOLD
    ultimo = hist_15m.iloc[-1]
    close = ultimo['Close']
    tp = round(close * 1.02, 2)
    sl = round(close * 0.98, 2)

    return {
        "segnale": "HOLD",
        "commento": f"Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}",
        "prezzo": round(close, 2),
        "take_profit": tp,
        "stop_loss": sl
    }