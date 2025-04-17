from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time

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

def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcola_macd(serie):
    ema12 = serie.ewm(span=12).mean()
    ema26 = serie.ewm(span=26).mean()
    macd = ema12 - ema26
    segnale = macd.ewm(span=9).mean()
    return macd, segnale

def calcola_atr(df, periodi=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df['TR'].rolling(window=periodi).mean()

def valuta_distanza(distanza):
    if distanza < 1:
        return "bassa"
    elif distanza < 3:
        return "media"
    else:
        return "alta"

def genera_grafico_base64(df):
    try:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
        ax.plot(df['Close'], label='Close', linewidth=1.5)
        ax.plot(df['EMA_9'], label='EMA 9', linestyle='--')
        ax.plot(df['EMA_21'], label='EMA 21', linestyle='--')
        ax.plot(df['EMA_100'], label='EMA 100', linestyle='--')
        ax.legend()
        ax.set_title('Prezzo e Medie Mobili')
        ax.grid(True)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        print(f"✅ Lunghezza grafico base64: {len(encoded)}")
        return encoded
    except Exception as e:
        print(f"❌ Errore durante la generazione del grafico: {e}")
        return None

# Funzioni per individuare l'incrocio esatto
def ha_incrociato_sopra(serie_a, serie_b):
    return serie_a.iloc[-2] < serie_b.iloc[-2] and serie_a.iloc[-1] > serie_b.iloc[-1]

def ha_incrociato_sotto(serie_a, serie_b):
    return serie_a.iloc[-2] > serie_b.iloc[-2] and serie_a.iloc[-1] < serie_b.iloc[-1]

def analizza_trend(hist):
    hist['EMA_9'] = hist['Close'].ewm(span=9).mean()
    hist['EMA_21'] = hist['Close'].ewm(span=21).mean()
    hist['EMA_100'] = hist['Close'].ewm(span=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['Close'])
    
    # Se il dataset è troppo piccolo, restituisci HOLD
    if len(hist) < 5:
        return "HOLD", hist, 0.0

    ultimo = hist.iloc[-1]
    ema9_now = ultimo['EMA_9']
    ema21_now = ultimo['EMA_21']
    ema100_now = ultimo['EMA_100']
    rsi = ultimo['RSI']
    close = ultimo['Close']

    # Valori precedenti per il controllo incrocio
    ema9_past = hist['EMA_9'].iloc[-4]
    ema21_past = hist['EMA_21'].iloc[-4]
    ema100_past = hist['EMA_100'].iloc[-4]

    # Distanza attuale tra medie (somma delle differenze)
    dist_now = abs(ema9_now - ema100_now) + abs(ema21_now - ema100_now)
    # Calcolo della distanza relativa, in percentuale rispetto alla EMA 100
    distanza_relativa = abs((ema21_now - ema100_now) / ema100_now) * 100

    macd = ultimo['MACD']
    macd_signal = ultimo['MACD_SIGNAL']

    # Controllo volume: volume medio delle ultime 20 candele e volume corrente
    if 'Volume' in hist.columns and len(hist['Volume']) >= 20:
        volume_medio = hist['Volume'].rolling(window=20).mean().iloc[-1]
        volume_corrente = hist['Volume'].iloc[-1]
    else:
        volume_medio = 0
        volume_corrente = 0

    # Controllo preciso dell'incrocio
    incrocio_rialzista = (
        ha_incrociato_sopra(hist['EMA_9'], hist['EMA_100']) and
        ha_incrociato_sopra(hist['EMA_21'], hist['EMA_100'])
    )
    incrocio_ribassista = (
        ha_incrociato_sotto(hist['EMA_9'], hist['EMA_100']) and
        ha_incrociato_sotto(hist['EMA_21'], hist['EMA_100'])
    )

    # Filtro sul volume: si accetta il segnale solo se il volume corrente supera il volume medio
    filtro_volume = (volume_corrente > volume_medio) if volume_medio > 0 else True

    # Condizioni per BUY o SELL, includendo il controllo volume e la conferma degli indicatori
    if incrocio_rialzista and filtro_volume and macd > macd_signal and rsi > 55 and close > ema100_now:
        return "BUY", hist, dist_now
    elif incrocio_ribassista and filtro_volume and macd < macd_signal and rsi < 45 and close < ema100_now:
        return "SELL", hist, dist_now

    return "HOLD", hist, dist_now

@app.get("/analyze", response_model=SignalResponse)
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
        except Exception as e:
            segnale_5m = "NON DISPONIBILE"
            conferma_due_timeframe = False
            note_timeframe = "⚠️ Dati 5m non disponibili, analisi solo su 15m\n"

        ultimo = hist_15m.iloc[-1]
        close = ultimo['Close']
        atr = ultimo['ATR']
        # Calcolo dinamico dell'entry price, TP e SL
        if segnale_15m == "BUY":
            entry_price = round(close + 0.002, 4)
            tp = round(entry_price + (atr * 1.5), 4)
            sl = round(entry_price - (atr * 1.2), 4)
        elif segnale_15m == "SELL":
            entry_price = round(close - 0.002, 4)
            tp = round(entry_price - (atr * 1.5), 4)
            sl = round(entry_price + (atr * 1.2), 4)
        else:
            entry_price = round(close, 4)
            tp = round(close * 1.02, 4)
            sl = round(close * 0.98, 4)

        rsi = round(ultimo['RSI'], 2)
        ema9 = round(ultimo['EMA_9'], 2)
        ema21 = round(ultimo['EMA_21'], 2)
        ema100 = round(ultimo['EMA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        dist_level = valuta_distanza(distanza_15m)
        grafico = genera_grafico_base64(hist_15m)

        ritardo = " | ⚠️ Ritardo stimato: ~15 minuti" if not is_crypto else ""

        if conferma_due_timeframe:
            commento = (
                f"✅ {segnale_15m} confermato su 5m e 15m{ritardo}\n"
                f"Distanza medie: {dist_level}\n"
                f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100} | ATR: {atr}\n"
                f"🎯 Entry: {entry_price} | TP: {tp} | 🛡️ SL: {sl}"
            )
        else:
            commento = (
                f"{note_timeframe}"
                f"Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}{ritardo}\n"
                f"Distanza medie: {dist_level}\n"
                f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100} | ATR: {atr}\n"
                f"🎯 Entry stimato: {entry_price} | TP: {tp} | 🛡️ SL: {sl}"
            )
            # Se il segnale non è confermato, applica livelli standard (questo può essere ulteriormente personalizzato)
            segnale_15m = "HOLD"

        return SignalResponse(
            segnale=segnale_15m,
            commento=commento,
            prezzo=round(close, 4),
            take_profit=tp,
            stop_loss=sl,
            graficoBase64=grafico
        )

    except Exception as e:
        print(f"Errore in analyze: {e}")
        return SignalResponse(
            segnale="ERROR",
            commento=f"Dati insufficienti per {symbol.upper()}",
            prezzo=0.0,
            take_profit=0.0,
            stop_loss=0.0
        )

# 🔥 Titoli caldi basati su finestre mobili da 5 candele nelle ultime 12 ore
_hot_cache = {"time": 0, "data": []}

@app.get("/hotassets")
def hot_assets():
    now = time.time()
    if now - _hot_cache["time"] < 900:
        return _hot_cache["data"]

    symbols = [
        "BTC-USD", "ETH-USD", "BCH-USD", "LTC-USD", "XRP-USD",
        "EOS-USD", "TRX-USD", "ETC-USD", "DASH-USD", "ZEC-USD", "QTUM-USD"
    ]

    risultati = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="15m")
            if hist.empty or len(hist) < 100:
                continue

            segnali = 0
            for i in range(-48, -4):  # 12 ore = 48 candele da 15m
                sub_hist = hist.iloc[i - 4:i + 1].copy()
                segnale, _, _ = analizza_trend(sub_hist)
                if segnale in ["BUY", "SELL"]:
                    segnali += 1

            if segnali > 0:
                ultimo = hist.iloc[-1]
                risultati.append({
                    "symbol": symbol,
                    "segnali": segnali,
                    "rsi": round(ultimo['RSI'], 2),
                    "ema9": round(ultimo['EMA_9'], 2),
                    "ema21": round(ultimo['EMA_21'], 2),
                    "ema100": round(ultimo['EMA_100'], 2),
                })

        except Exception as e:
            print(f"Errore con {symbol}: {e}")
            continue

    risultati_ordinati = sorted(risultati, key=lambda x: x.get("segnali", 0), reverse=True)[:10]
    _hot_cache["time"] = now
    _hot_cache["data"] = risultati_ordinati
    return risultati_ordinati