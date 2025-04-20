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
        close = round(ultimo['Close'], 4)
        rsi = round(ultimo['RSI'], 2)
        ema9 = round(ultimo['EMA_9'], 2)
        ema21 = round(ultimo['EMA_21'], 2)
        ema100 = round(ultimo['EMA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)
        dist_level = valuta_distanza(distanza)

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
                f"{note_timeframe.strip()}{ritardo}"
            )
        else:
            commento = (
                f"{note_timeframe}Segnale non confermato: {timeframe}={segnale}, 5m={segnale_5m}{ritardo}\n"
                f"RSI: {rsi} | EMA(9/21/100): {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal} | ATR: {atr}\n"
                f"Distanza medie: {dist_level}\n"
                f"{note}\nSupporto: {supporto}$"
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
            commento=f"Errore durante l'analisi di {symbol.upper()}",
            prezzo=0.0,
            take_profit=0.0,
            stop_loss=0.0
        )
# Cache per hot assets
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

    def calcola_rsi(serie, periodi=14):
        delta = serie.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=periodi).mean()
        avg_loss = loss.rolling(window=periodi).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

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

            # ATR (volatilità)
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
