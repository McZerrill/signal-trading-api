from fastapi import FastAPI from fastapi.middleware.cors import CORSMiddleware from pydantic import BaseModel import yfinance as yf import pandas as pd import matplotlib.pyplot as plt import io import base64

app = FastAPI()

app.add_middleware( CORSMiddleware, allow_origins=[""], allow_methods=[""], allow_headers=["*"], )

class SignalResponse(BaseModel): segnale: str commento: str prezzo: float take_profit: float stop_loss: float graficoBase64: str | None = None

def calcola_rsi(serie, periodi=14): delta = serie.diff() gain = delta.where(delta > 0, 0) loss = -delta.where(delta < 0, 0) avg_gain = gain.rolling(window=periodi).mean() avg_loss = loss.rolling(window=periodi).mean() rs = avg_gain / avg_loss return 100 - (100 / (1 + rs))

def calcola_atr(df, periodi=14): df['H-L'] = df['High'] - df['Low'] df['H-PC'] = abs(df['High'] - df['Close'].shift()) df['L-PC'] = abs(df['Low'] - df['Close'].shift()) df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1) return df['TR'].rolling(window=periodi).mean()

def calcola_macd(serie): ema12 = serie.ewm(span=12, adjust=False).mean() ema26 = serie.ewm(span=26, adjust=False).mean() macd = ema12 - ema26 signal = macd.ewm(span=9, adjust=False).mean() return macd, signal

def valuta_distanza(distanza): if distanza < 1: return "bassa" elif distanza < 3: return "media" else: return "alta"

def genera_grafico_base64(df): try: fig, ax = plt.subplots(figsize=(6, 3)) ax.plot(df['Close'], label='Close', linewidth=1.5) ax.plot(df['EMA_20'], label='EMA 20', linestyle='--') ax.plot(df['EMA_50'], label='EMA 50', linestyle='--') ax.legend() ax.set_title('Prezzo e Medie Mobili') ax.grid(True)

buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded
except Exception as e:
    print(f"❌ Errore durante la generazione del grafico: {e}")
    return None

def analizza_trend(hist): hist['EMA_20'] = hist['Close'].ewm(span=20).mean() hist['EMA_50'] = hist['Close'].ewm(span=50).mean() hist['RSI'] = calcola_rsi(hist['Close']) hist['ATR'] = calcola_atr(hist) hist['MACD'], hist['MACD_Signal'] = calcola_macd(hist['Close'])

if len(hist) < 5:
    return "HOLD", hist, 0.0

ultimo = hist.iloc[-1]
ema20_now = ultimo['EMA_20']
ema50_now = ultimo['EMA_50']
macd_now = ultimo['MACD']
macd_signal_now = ultimo['MACD_Signal']

ema20_past = hist['EMA_20'].iloc[-4]
ema50_past = hist['EMA_50'].iloc[-4]

dist_now = abs(ema20_now - ema50_now)
dist_past = abs(ema20_past - ema50_past)

if (
    ema20_past < ema50_past and
    ema20_now > ema50_now and
    macd_now > macd_signal_now and
    dist_now > dist_past
):
    return "BUY", hist, dist_now

elif (
    ema20_past > ema50_past and
    ema20_now < ema50_now and
    macd_now < macd_signal_now and
    dist_now > dist_past
):
    return "SELL", hist, dist_now

return "HOLD", hist, dist_now

@app.get("/analyze", response_model=SignalResponse) def analyze(symbol: str): data = yf.Ticker(symbol) is_crypto = "-USD" in symbol.upper()

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
    ema20 = round(ultimo['EMA_20'], 2)
    ema50 = round(ultimo['EMA_50'], 2)
    macd = round(ultimo['MACD'], 2)
    macd_signal = round(ultimo['MACD_Signal'], 2)
    atr = round(ultimo['ATR'], 2)
    dist_level = valuta_distanza(distanza_15m)
    grafico = genera_grafico_base64(hist_15m)

    ritardo = " | ⚠️ Ritardo stimato: ~15 minuti" if not is_crypto else ""

    if conferma_due_timeframe:
        commento = (
            f"✅ {segnale_15m} confermato su 5m e 15m{ritardo}\n"
            f"Distanza medie: {dist_level}\n"
            f"RSI: {rsi} | EMA20: {ema20} | EMA50: {ema50} | MACD: {macd} | Signal: {macd_signal} | ATR: {atr}\n"
            f"🎯 TP: {tp} | 🛡️ SL: {sl}"
        )
    else:
        commento = (
            f"{note_timeframe}"
            f"Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}{ritardo}\n"
            f"Distanza medie: {dist_level}\n"
            f"RSI: {rsi} | EMA20: {ema20} | EMA50: {ema50} | MACD: {macd} | Signal: {macd_signal} | ATR: {atr}"
        )
        tp = round(close * 1.02, 2)
        sl = round(close * 0.98, 2)
        segnale_15m = "HOLD"

    return SignalResponse(
        segnale=segnale_15m,
        commento=commento,
        prezzo=round(close, 2),
        take_profit=tp,
        stop_loss=sl,
        graficoBase64=grafico
    )

except:
    return SignalResponse(
        segnale="ERROR",
        commento=f"Dati insufficienti per {symbol.upper()}",
        prezzo=0.0,
        take_profit=0.0,
        stop_loss=0.0
    )

