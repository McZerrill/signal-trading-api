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
    ma = serie.ewm(span=periodo_ma).mean()
    deviazione = abs(serie - ma)
    media_deviazione = deviazione.ewm(span=periodo_tii).mean()
    return 100 - (100 / (1 + (serie - ma) / media_deviazione))

def valuta_distanza(distanza):
    if distanza < 1:
        return "bassa"
    elif distanza < 3:
        return "media"
    else:
        return "alta"

def analizza_trend(hist):
    hist['EMA_9'] = hist['Close'].ewm(span=9).mean()
    hist['EMA_21'] = hist['Close'].ewm(span=21).mean()
    hist['EMA_100'] = hist['Close'].ewm(span=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = calcola_atr(hist)
    hist['TII'] = calcola_tii(hist['Close'])

    ultimo = hist.iloc[-1]
    ema9_now = ultimo['EMA_9']
    ema21_now = ultimo['EMA_21']
    ema100_now = ultimo['EMA_100']

    ema9_past = hist['EMA_9'].iloc[-4]
    ema21_past = hist['EMA_21'].iloc[-4]
    ema100_past = hist['EMA_100'].iloc[-4]

    dist_now = abs(ema9_now - ema100_now) + abs(ema21_now - ema100_now)
    dist_past = abs(ema9_past - ema100_past) + abs(ema21_past - ema100_past)

    # BUY
    if (
        ema9_past < ema100_past and
        ema21_past < ema100_past and
        ema9_now > ema100_now and
        ema21_now > ema100_now and
        dist_now > dist_past
    ):
        return "BUY", hist, dist_now

    # SELL
    elif (
        ema9_past > ema100_past and
        ema21_past > ema100_past and
        ema9_now < ema100_now and
        ema21_now < ema100_now and
        dist_now > dist_past
    ):
        return "SELL", hist, dist_now

    return "HOLD", hist, dist_now

@app.get("/analyze")
def analyze(symbol: str):
    data = yf.Ticker(symbol)
    is_crypto = "-USD" in symbol.upper()

    try:
        hist_15m = data.history(period="7d", interval="15m")
        if hist_15m.empty or len(hist_15m) < 100:
            raise Exception("Dati insufficienti 15m")

        segnale_15m, hist_15m, distanza_15m = analizza_trend(hist_15m)

        try:
            hist_5m = data.history(period="1d", interval="5m")
            if hist_5m.empty or len(hist_5m) < 100:
                raise Exception("Dati 5m insufficienti")
            segnale_5m, _, _ = analizza_trend(hist_5m)
            conferma_due_timeframe = (segnale_15m == segnale_5m and segnale_15m != "HOLD")
            note_timeframe = ""
        except:
            segnale_5m = "NON DISPONIBILE"
            conferma_due_timeframe = False
            note_timeframe = "⚠️ Dati 5m non disponibili, analisi solo su 15m\n"

        ultimo = hist_15m.iloc[-1]
        close = ultimo['Close']
        atr = ultimo['ATR']
        spread = (ultimo['High'] - ultimo['Low']) * 0.1
        tp = round(close + (atr + spread), 2) if segnale_15m == "BUY" else round(close - (atr + spread), 2)
        sl = round(close - (atr * 0.8), 2) if segnale_15m == "BUY" else round(close + (atr * 0.8), 2)

        rsi = round(ultimo['RSI'], 2)
        ema9 = round(ultimo['EMA_9'], 2)
        ema21 = round(ultimo['EMA_21'], 2)
        ema100 = round(ultimo['EMA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        tii = round(ultimo['TII'], 2)
        dist_level = valuta_distanza(distanza_15m)

        ritardo = " | ⚠️ Ritardo stimato: ~15 minuti" if not is_crypto else ""

        if conferma_due_timeframe:
            commento = (
                f"✅ {segnale_15m} confermato su 5m e 15m{ritardo}\n"
                f"Distanza medie: {dist_level}\n"
                f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100} | ATR: {atr} | TII: {tii}\n"
                f"🎯 TP: {tp} | 🛡️ SL: {sl}"
            )
            return {
                "segnale": segnale_15m,
                "commento": commento,
                "prezzo": round(close, 2),
                "take_profit": tp,
                "stop_loss": sl
            }

        commento = (
            f"{note_timeframe}"
            f"Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}{ritardo}\n"
            f"Distanza medie: {dist_level}\n"
            f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100} | ATR: {atr} | TII: {tii}"
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
