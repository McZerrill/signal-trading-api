# FILE 1/5 - avvio e modelli

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API Segnali di Borsa attiva"}

class SignalResponse(BaseModel):
    segnale: str
    commento: str
    prezzo: float
    take_profit: float
    stop_loss: float
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    atr: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    ema100: float = 0.0
    timeframe: str = ""
# FILE 2/5 - Funzioni tecniche

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

def calcola_supporto(df, lookback=20):
    return round(df['Low'].tail(lookback).min(), 2)

def valuta_distanza(distanza):
    if distanza < 1:
        return "bassa"
    elif distanza < 3:
        return "media"
    else:
        return "alta"

def conta_candele_trend(hist, rialzista=True):
    count = 0
    for i in range(-1, -21, -1):
        ema9 = hist['EMA_9'].iloc[i]
        ema21 = hist['EMA_21'].iloc[i]
        ema100 = hist['EMA_100'].iloc[i]
        if rialzista:
            if ema9 > ema21 > ema100:
                count += 1
            else:
                break
        else:
            if ema9 < ema21 < ema100:
                count += 1
            else:
                break
    return count
# FILE 3/5 - Funzione principale di analisi del trend

def analizza_trend(hist):
    hist['EMA_9'] = hist['Close'].ewm(span=9).mean()
    hist['EMA_21'] = hist['Close'].ewm(span=21).mean()
    hist['EMA_100'] = hist['Close'].ewm(span=100).mean()
    hist['RSI'] = calcola_rsi(hist['Close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['Close'])

    if len(hist) < 5:
        return "HOLD", hist, 0.0, "", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema9, ema21, ema100 = ultimo['EMA_9'], ultimo['EMA_21'], ultimo['EMA_100']
    close = ultimo['Close']
    rsi = ultimo['RSI']
    atr = ultimo['ATR']
    macd = ultimo['MACD']
    macd_signal = ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    dist_attuale = abs(ema9 - ema21) + abs(ema21 - ema100)
    dist_precedente = abs(penultimo['EMA_9'] - penultimo['EMA_21']) + abs(penultimo['EMA_21'] - penultimo['EMA_100'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    note = ""
    segnale = "HOLD"
    tp = sl = 0.0

    if ema9 > ema21 > ema100:
        if dist_diff > 0:
            trend_strength = "\U0001F4C8 Trend forte in espansione"
        elif dist_diff < -0.1:
            trend_strength = "\u26A0\uFE0F Trend in indebolimento"
        else:
            trend_strength = "\u2796 Trend stabile"
    elif ema9 < ema21 and ema21 > ema100 and ema9 > ema100:
        trend_strength = "\u26D4\uFE0F Trend in esaurimento"
    elif ema9 < ema21 < ema100:
        trend_strength = "\u26D4\uFE0F Trend concluso o inversione in corso"
    elif ema9 > ema21 and penultimo['EMA_9'] < penultimo['EMA_21']:
        trend_strength = "\U0001F501 Trend ripreso"
    else:
        trend_strength = ""

    if ema9 > ema21 and (ema9 > ema100 or (ema21 - ema100) / ema100 < 0.01) and rsi > 50 and macd > macd_signal:
        segnale = "BUY"
        resistenza = hist['High'].tail(20).max()
        tp_raw = close + atr * 1.5
        sl_raw = close - atr * 1.2
        tp = round(min(tp_raw, resistenza), 4)
        sl = round(sl_raw, 4)

    elif ema9 < ema21 and (ema9 < ema100 or (ema21 - ema100) / ema100 < 0.01) and rsi < 50 and macd < macd_signal:
        segnale = "SELL"
        tp_raw = close - atr * 1.5
        sl_raw = close + atr * 1.2
        tp = round(max(tp_raw, supporto), 4)
        sl = round(sl_raw, 4)

    elif macd < macd_signal and rsi < 45 and dist_attuale < 1.5:
        note = "\u26A0\uFE0F Segnale anticipato: MACD debole + RSI sotto 45"

    if segnale == "HOLD":
        if penultimo['EMA_9'] < penultimo['EMA_21'] and ema9 > ema21:
            if (ema21 < ema100 and abs(ema9 - ema100) / ema100 < 0.01) and rsi > 50:
                note += "\n\U0001F4E1 Presegnale: EMA9 ha incrociato EMA21 e si avvicina a EMA100 (BUY anticipato)"
        elif penultimo['EMA_9'] > penultimo['EMA_21'] and ema9 < ema21:
            if (ema21 > ema100 and abs(ema9 - ema100) / ema100 < 0.01) and rsi < 50:
                note += "\n\U0001F4E1 Presegnale: EMA9 ha incrociato EMA21 al ribasso e si avvicina a EMA100 (SELL anticipato)"

    candele_trend = conta_candele_trend(hist, rialzista=(segnale == "BUY"))

    if segnale in ["BUY", "SELL"] and candele_trend >= 3:
        note += f"\n\U0001F4CA Attivo da {candele_trend} candele | {dist_level} distanza tra medie"
    elif segnale == "HOLD":
        if candele_trend <= 1 and not (ema9 > ema21 > ema100):
            note += "\n\u26D4\uFE0F Trend esaurito, considera chiusura posizione"
        elif ema9 > ema21 > ema100:
            note += "\n\u2796 Trend ancora attivo ma debole"

    note = trend_strength + ("\n" + note if note else "")

    return segnale, hist, dist_attuale, note, tp, sl, supporto


from datetime import datetime

from datetime import datetime

@app.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    ticker = yf.Ticker(symbol)
    is_crypto = "-USD" in symbol.upper()

    try:
        # ✅ Recupero variazione 24h
        try:
            info = ticker.info
            change_pct_24h = info.get("regularMarketChangePercent", 0.0)
        except:
            change_pct_24h = 0.0

        hist_15m = ticker.history(period="7d", interval="15m")
        hist_30m = ticker.history(period="14d", interval="30m")

        if hist_15m.empty or len(hist_15m) < 100:
            raise Exception("Dati insufficienti 15m")
        if hist_30m.empty or len(hist_30m) < 100:
            raise Exception("Dati insufficienti 30m")

        segnale_15m, h15, dist_15m, note15, tp15, sl15, supporto15 = analizza_trend(hist_15m)
        segnale_30m, h30, dist_30m, note30, tp30, sl30, supporto30 = analizza_trend(hist_30m)

        def conta_trend_attivo(hist):
            count = 0
            for i in range(-10, 0):
                sub = hist.iloc[i]
                if sub['EMA_9'] > sub['EMA_21'] > sub['EMA_100'] or sub['EMA_9'] < sub['EMA_21'] < sub['EMA_100']:
                    count += 1
            return count

        trend_15m = conta_trend_attivo(h15)
        trend_30m = conta_trend_attivo(h30)

        if trend_30m > trend_15m:
            timeframe = "30m"
            segnale, hist, distanza, note, tp, sl, supporto = segnale_30m, h30, dist_30m, note30, tp30, sl30, supporto30
        else:
            timeframe = "15m"
            segnale, hist, distanza, note, tp, sl, supporto = segnale_15m, h15, dist_15m, note15, tp15, sl15, supporto15

        try:
            hist_5m = ticker.history(period="1d", interval="5m")
            if hist_5m.empty or len(hist_5m) < 100:
                raise Exception("Dati 5m insufficienti")
            segnale_5m, _, _, _, _, _, _ = analizza_trend(hist_5m)
            conferma_due_timeframe = (segnale == segnale_5m and segnale != "HOLD")
            note_timeframe = ""
        except:
            segnale_5m = "NON DISPONIBILE"
            conferma_due_timeframe = False
            note_timeframe = f"⚠️ Dati 5m non disponibili, analisi solo su {timeframe}\n"

        # 🔧 Calcolo ritardo
        ultima_candela = hist.index[-1].to_pydatetime().replace(tzinfo=None)
        ora_corrente = datetime.utcnow()
        ritardo_minuti = int((ora_corrente - ultima_candela).total_seconds() / 60)
        ritardo_stimato = f"⏱️ Ritardo stimato: ~{ritardo_minuti} minuti"
        ritardo = f"\n{ritardo_stimato}"

        # 🔎 Estrazione dati
        ultimo = hist.iloc[-1]
        close = round(ultimo['Close'], 4)
        rsi = round(ultimo['RSI'], 2)
        ema9 = round(ultimo['EMA_9'], 2)
        ema21 = round(ultimo['EMA_21'], 2)
        ema100 = round(ultimo['EMA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)
        dist_level = valuta_distanza(distanza)
        change_pct = round(change_pct_24h, 2)

        # 🔔 Commento finale
        orario_candela = ultima_candela.strftime("%H:%M UTC (%d/%m)")

        if segnale in ["BUY", "SELL"]:
            tp_pct = round(((tp - close) / close) * 100, 1)
            sl_pct = round(((sl - close) / close) * 100, 1)
            trend_attivo = conta_trend_attivo(hist)
            trend_msg = f"Attivo da {trend_attivo} candele" if trend_attivo >= 2 else "Debole"

            commento = (
                f"{'🟢 BUY' if segnale == 'BUY' else '🔴 SELL'} | {symbol.upper()} @ {close}$\n"
                f"🎯 Target: {tp} ({tp_pct}%)\n"
                f"🛡 Stop: {sl} ({sl_pct}%)\n"
                f"RSI: {rsi} | EMA(9/21/100): {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal} | ATR: {atr}\n"
                f"Trend: {trend_msg} | {dist_level} distanza tra medie\n"
                f"{note_timeframe.strip()}{ritardo}\n"
                f"🕒 Prezzo rilevato alle: {orario_candela}"
            )
        else:
            commento = (
                f"{note_timeframe}Segnale non confermato: {timeframe}={segnale}, 5m={segnale_5m}{ritardo}\n"
                f"RSI: {rsi} | EMA(9/21/100): {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal} | ATR: {atr}\n"
                f"Distanza medie: {dist_level}\n"
                f"{note}\nSupporto: {supporto}$\n"
                f"🕒 Prezzo rilevato alle: {orario_candela}"
            )
            tp = sl = 0.0

        return SignalResponse(
            segnale=segnale,
            commento=commento,
            prezzo=close,
            take_profit=tp,
            stop_loss=sl,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            atr=atr,
            ema9=ema9,
            ema21=ema21,
            ema100=ema100,
            timeframe=timeframe
        )

    except Exception as e:
        print(f"Errore: {e}")
        return SignalResponse(
            segnale="ERROR",
            commento=f"Errore durante l'analisi di {symbol.upper()}: {e}",
            prezzo=0.0,
            take_profit=0.0,
            stop_loss=0.0
        )
# FILE 5/5 - Endpoint /hotassets

_hot_cache = {"time": 0, "data": []}

@app.get("/hotassets")
def hot_assets():
    now = time.time()
    if now - _hot_cache["time"] < 60:
        return _hot_cache["data"]

    symbols = [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "AVAX-USD", "DOT-USD",
        "DOGE-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "TRX-USD", "ETC-USD", "ATOM-USD",
        "NEAR-USD", "INJ-USD", "RNDR-USD", "FTM-USD", "ALGO-USD", "VET-USD", "SAND-USD",
        "EOS-USD", "CRO-USD", "HBAR-USD", "ZIL-USD", "OP-USD", "AR-USD"
    ]

    risultati = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty or len(hist) < 100:
                continue

            hist['EMA_9'] = hist['Close'].ewm(span=9).mean()
            hist['EMA_21'] = hist['Close'].ewm(span=21).mean()
            hist['EMA_100'] = hist['Close'].ewm(span=100).mean()
            hist['RSI'] = calcola_rsi(hist['Close'])

            hist['H-L'] = hist['High'] - hist['Low']
            hist['H-PC'] = abs(hist['High'] - hist['Close'].shift())
            hist['L-PC'] = abs(hist['Low'] - hist['Close'].shift())
            hist['TR'] = hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            hist['ATR'] = hist['TR'].rolling(window=14).mean()

            atr = hist['ATR'].iloc[-1]
            dist_medie = abs(hist['EMA_9'].iloc[-1] - hist['EMA_21'].iloc[-1]) + abs(hist['EMA_21'].iloc[-1] - hist['EMA_100'].iloc[-1])

            buy_signals = 0
            sell_signals = 0

            for i in range(-40, -4):
                sub_hist = hist.iloc[i - 4:i + 1].copy()
                sub_hist['EMA_9'] = sub_hist['Close'].ewm(span=9).mean()
                sub_hist['EMA_21'] = sub_hist['Close'].ewm(span=21).mean()
                sub_hist['EMA_100'] = sub_hist['Close'].ewm(span=100).mean()
                sub_hist['RSI'] = calcola_rsi(sub_hist['Close'])

                e9 = sub_hist['EMA_9'].iloc[-1]
                e21 = sub_hist['EMA_21'].iloc[-1]
                e100 = sub_hist['EMA_100'].iloc[-1]
                rsi = sub_hist['RSI'].iloc[-1]

                if e9 > e21 > e100 and rsi > 50:
                    buy_signals += 1
                elif e9 < e21 < e100 and rsi < 50:
                    sell_signals += 1

            total_signals = buy_signals + sell_signals
            if total_signals >= 1 or (atr > 0.5 and dist_medie > 1):
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

    risultati_ordinati = sorted(risultati, key=lambda x: x['segnali'], reverse=True)[:10]
    if not risultati_ordinati:
        print("⚠️ Nessun asset hot rilevato.")

    _hot_cache["time"] = now
    _hot_cache["data"] = risultati_ordinati
    return risultati_ordinati
