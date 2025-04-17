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

app.add_middleware( CORSMiddleware, allow_origins=[""], allow_methods=[""], allow_headers=["*"], )

class SignalResponse(BaseModel): segnale: str commento: str prezzo: float take_profit: float stop_loss: float graficoBase64: str | None = None

def calcola_rsi(serie, periodi=14): delta = serie.diff() gain = delta.where(delta > 0, 0) loss = -delta.where(delta < 0, 0) avg_gain = gain.rolling(window=periodi).mean() avg_loss = loss.rolling(window=periodi).mean() rs = avg_gain / avg_loss return 100 - (100 / (1 + rs))

def calcola_macd(serie): ema12 = serie.ewm(span=12).mean() ema26 = serie.ewm(span=26).mean() macd = ema12 - ema26 segnale = macd.ewm(span=9).mean() return macd, segnale

def calcola_atr(df, periodi=14): df['H-L'] = df['High'] - df['Low'] df['H-PC'] = abs(df['High'] - df['Close'].shift()) df['L-PC'] = abs(df['Low'] - df['Close'].shift()) df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1) return df['TR'].rolling(window=periodi).mean()

def calcola_supporto(df, lookback=20): return round(df['Low'].tail(lookback).min(), 2)

def valuta_distanza(distanza): if distanza < 1: return "bassa" elif distanza < 3: return "media" else: return "alta"

def genera_grafico_base64(df): try: fig, ax = plt.subplots(figsize=(6, 3), dpi=200) ax.plot(df['Close'], label='Close', linewidth=1.5) ax.plot(df['EMA_9'], label='EMA 9', linestyle='--') ax.plot(df['EMA_21'], label='EMA 21', linestyle='--') ax.plot(df['EMA_100'], label='EMA 100', linestyle='--') ax.legend() ax.set_title('Prezzo e Medie Mobili') ax.grid(True) buf = io.BytesIO() plt.tight_layout() plt.savefig(buf, format='png', bbox_inches='tight') plt.close(fig) buf.seek(0) encoded = base64.b64encode(buf.read()).decode('utf-8') return encoded except: return None

def analizza_trend(hist): hist['EMA_9'] = hist['Close'].ewm(span=9).mean() hist['EMA_21'] = hist['Close'].ewm(span=21).mean() hist['EMA_100'] = hist['Close'].ewm(span=100).mean() hist['RSI'] = calcola_rsi(hist['Close']) hist['ATR'] = calcola_atr(hist) hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['Close'])

if len(hist) < 5:
    return "HOLD", hist, 0.0, "", 0.0, 0.0, 0.0

ultimo = hist.iloc[-1]
ema9 = ultimo['EMA_9']
ema21 = ultimo['EMA_21']
ema100 = ultimo['EMA_100']
rsi = ultimo['RSI']
macd = ultimo['MACD']
macd_signal = ultimo['MACD_SIGNAL']
close = ultimo['Close']
atr = ultimo['ATR']
supporto = calcola_supporto(hist)

dist = abs(ema9 - ema100) + abs(ema21 - ema100)
distanza_rel = abs((ema21 - ema100) / ema100) * 100

try:
    volume = ultimo['Volume']
    vol_media = hist['Volume'].rolling(window=20).mean().iloc[-1]
    filtro_volume = volume > vol_media
except:
    filtro_volume = True

note = ""
segnale = "HOLD"
tp = sl = 0.0

if ema9 > ema21 and (ema9 > ema100 or (ema21 - ema100) / ema100 < 0.01) and rsi > 50 and macd > macd_signal and filtro_volume:
    segnale = "BUY"
    tp = round(close + atr * 1.5, 4)
    sl = round(close - atr * 1.2, 4)
elif ema9 < ema21 and (ema9 < ema100 or (ema21 - ema100) / ema100 < 0.01) and rsi < 50 and macd < macd_signal and filtro_volume:
    segnale = "SELL"
    tp = round(close - atr * 1.5, 4)
    sl = round(close + atr * 1.2, 4)
elif macd < macd_signal and rsi < 45 and distanza_rel < 1.5:
    note = "⚠️ Segnale anticipato: MACD debole + RSI sotto 45"

return segnale, hist, dist, note, tp, sl, supporto

@app.get("/analyze", response_model=SignalResponse) def analyze(symbol: str): data = yf.Ticker(symbol) is_crypto = "-USD" in symbol.upper()

try:
    hist_15m = data.history(period="7d", interval="15m")
    if hist_15m.empty or len(hist_15m) < 100:
        raise Exception("Dati insufficienti 15m")

    segnale_15m, hist_15m, distanza_15m, note, tp, sl, supporto = analizza_trend(hist_15m)

    try:
        hist_5m = data.history(period="1d", interval="5m")
        if hist_5m.empty or len(hist_5m) < 100:
            raise Exception("Dati 5m insufficienti")
        segnale_5m, _, _, _, _, _, _ = analizza_trend(hist_5m)
        conferma_due_timeframe = (segnale_15m == segnale_5m and segnale_15m != "HOLD")
        note_timeframe = ""
    except:
        segnale_5m = "NON DISPONIBILE"
        conferma_due_timeframe = False
        note_timeframe = "⚠️ Dati 5m non disponibili, analisi solo su 15m\n"

    ultimo = hist_15m.iloc[-1]
    close = round(ultimo['Close'], 4)
    rsi = round(ultimo['RSI'], 2)
    ema9 = round(ultimo['EMA_9'], 2)
    ema21 = round(ultimo['EMA_21'], 2)
    ema100 = round(ultimo['EMA_100'], 2)
    atr = round(ultimo['ATR'], 2)
    macd = round(ultimo['MACD'], 4)
    macd_signal = round(ultimo['MACD_SIGNAL'], 4)
    dist_level = valuta_distanza(distanza_15m)
    grafico = genera_grafico_base64(hist_15m)

    ritardo = " | ⚠️ Ritardo stimato: ~15 minuti" if not is_crypto else ""

    if segnale_15m in ["BUY", "SELL"]:
        direzione = "LONG" if segnale_15m == "BUY" else "SHORT"
        tp_pct = round(((tp - close) / close) * 100, 1)
        sl_pct = round(((sl - close) / close) * 100, 1)
        commento = (
            f"✅ {segnale_15m} confermato su 15m{' e 5m' if conferma_due_timeframe else ''}{ritardo}\n"
            f"{note_timeframe}Segnale operativo: {direzione}\n"
            f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100}\n"
            f"MACD: {macd} | Signal: {macd_signal} | ATR: {atr}\n"
            f"Distanza medie: {dist_level}\n"
            f"🎯 Entry: {close} | TP: {tp_pct}% | 🛡️ SL: {sl_pct}%"
        )
        if segnale_15m == "SELL":
            commento += f"\nSupporto rilevante: {supporto}$"
    else:
        commento = (
            f"{note_timeframe}Segnale non confermato: 15m={segnale_15m}, 5m={segnale_5m}{ritardo}\n"
            f"RSI: {rsi} | EMA9: {ema9} | EMA21: {ema21} | EMA100: {ema100}\n"
            f"MACD: {macd} | Signal: {macd_signal} | ATR: {atr}\n"
            f"Distanza medie: {dist_level}\n"
            f"{note}\nSupporto rilevante: {supporto}$"
        )
        tp = sl = 0.0

    return SignalResponse(
        segnale=segnale_15m,
        commento=commento,
        prezzo=close,
        take_profit=tp,
        stop_loss=sl,
        graficoBase64=grafico
    )

except Exception as e:
    print(f"Errore: {e}")
    return SignalResponse(
        segnale="ERROR",
        commento=f"Errore durante l'analisi di {symbol.upper()}",
        prezzo=0.0,
        take_profit=0.0,
        stop_loss=0.0
    )



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
                print(f"{symbol}: Dati insufficienti.")
                continue

            buy_signals = 0
            sell_signals = 0

            for i in range(-48, -4):  # Ultime 12 ore con finestre mobili da 5 candele
                sub_hist = hist.iloc[i - 4:i + 1].copy()
                segnale, _, _, _, _, _, _ = analizza_trend(sub_hist)
                if segnale == "BUY":
                    buy_signals += 1
                elif segnale == "SELL":
                    sell_signals += 1

            total_signals = buy_signals + sell_signals
            if total_signals == 0:
                continue

            trend = "BUY" if buy_signals > sell_signals else "SELL" if sell_signals > buy_signals else "NEUTRO"

            ultimo = hist.iloc[-1]
            risultati.append({
                "symbol": symbol,
                "segnali": total_signals,
                "trend": trend,
                "rsi": round(ultimo['RSI'], 2),
                "ema9": round(ultimo['EMA_9'], 2),
                "ema21": round(ultimo['EMA_21'], 2),
                "ema100": round(ultimo['EMA_100'], 2),
            })

        except Exception as e:
            print(f"Errore con {symbol}: {e}")
            continue

    risultati_ordinati = sorted(risultati, key=lambda x: x.get("segnali", 0), reverse=True)[:10]

    if not risultati_ordinati:
        print("⚠️ Nessun hot asset rilevato nelle ultime 12 ore.")

    _hot_cache["time"] = now
    _hot_cache["data"] = risultati_ordinati
    return risultati_ordinati