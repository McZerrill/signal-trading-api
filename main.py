import requests
from pytz import timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from binance.client import Client
import pandas as pd
import time
from datetime import datetime
import os
from datetime import timezone as dt_timezone
utc = dt_timezone.utc


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

# Classe di risposta identica per compatibilità con l'app
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

# Inizializza il client Binance con variabili ambiente
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Funzione helper per ottenere un DataFrame da Binance
def get_binance_df(symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time is not None:
        params["endTime"] = end_time

    try:
        klines = client.get_klines(**params)
    except Exception as e:
        print(f"❌ Errore nel caricamento candela {symbol}-{interval}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# --- Funzioni tecniche ---

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
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift())
    df['L-PC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df['TR'].rolling(window=periodi).mean()

def calcola_supporto(df, lookback=20):
    return round(df['low'].tail(lookback).min(), 2)

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

    
def riconosci_pattern_candela(df):
    c = df.iloc[-1]  # ultima candela
    o, h, l, close = c['open'], c['high'], c['low'], c['close']
    corpo = abs(close - o)
    ombra_sup = h - max(o, close)
    ombra_inf = min(o, close) - l

    if corpo == 0:
        return ""

    # Hammer (BUY)
    if corpo > 0 and ombra_inf >= 2 * corpo and ombra_sup <= corpo * 0.3:
        return "🪓 Hammer rilevato (BUY)"
    
    # Shooting Star (SELL)
    if corpo > 0 and ombra_sup >= 2 * corpo and ombra_inf <= corpo * 0.3:
        return "🌠 Shooting Star rilevato (SELL)"
    
    return ""

# --- Analisi trend principale ---

def analizza_trend(hist):
    hist['EMA_9'] = hist['close'].ewm(span=9).mean()
    hist['EMA_21'] = hist['close'].ewm(span=21).mean()
    hist['EMA_100'] = hist['close'].ewm(span=100).mean()
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    if len(hist) < 5:
        return "HOLD", hist, 0.0, "", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema9, ema21, ema100 = ultimo['EMA_9'], ultimo['EMA_21'], ultimo['EMA_100']
    close = ultimo['close']
    rsi = ultimo['RSI']
    atr = ultimo['ATR']
    macd = ultimo['MACD']
    macd_signal = ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    dist_attuale = abs(ema9 - ema21) + abs(ema21 - ema100)
    dist_precedente = abs(penultimo['EMA_9'] - penultimo['EMA_21']) + abs(penultimo['EMA_21'] - penultimo['EMA_100'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    segnale = "HOLD"
    tp = sl = 0.0
    note = []

    # Forza del trend
    if ema9 > ema21 > ema100:
        if dist_diff > 0:
            forza_trend = "📈 Trend forte in espansione"
        elif dist_diff < -0.1:
            forza_trend = "⚠️ Trend in indebolimento"
        else:
            forza_trend = "➖ Trend stabile"
    elif ema9 < ema21 and ema21 > ema100 and ema9 > ema100:
        forza_trend = "⛔️ Trend in esaurimento"
    elif ema9 < ema21 < ema100:
        forza_trend = "⛔️ Trend concluso o inversione in corso"
    elif ema9 > ema21 and penultimo['EMA_9'] < penultimo['EMA_21']:
        forza_trend = "🔁 Trend ripreso"
    else:
        forza_trend = ""

    # Verifica coerenza ultime 3 candele
    recent_trend_up = all(hist['EMA_9'].iloc[-i] > hist['EMA_21'].iloc[-i] for i in range(1, 4))
    recent_trend_down = all(hist['EMA_9'].iloc[-i] < hist['EMA_21'].iloc[-i] for i in range(1, 4))

    # BUY
    if ema9 > ema21 > ema100 and rsi > 50 and macd > macd_signal and recent_trend_up:
        segnale = "BUY"
        resistenza = hist['high'].tail(20).max()
        tp = round(min(close + atr * 1.5, resistenza), 4)
        sl = round(close - atr * 1.2, 4)

    # SELL
    elif ema9 < ema21 < ema100 and rsi < 50 and macd < macd_signal and recent_trend_down:
        segnale = "SELL"
        tp = round(max(close - atr * 1.5, supporto), 4)
        sl = round(close + atr * 1.2, 4)

    elif macd < macd_signal and rsi < 45 and dist_attuale < 1.5:
        note.append("⚠️ Segnale anticipato: MACD debole + RSI sotto 45")

    # Presegnali in caso di HOLD
    presegnale = ""
    if segnale == "HOLD":
        if penultimo['EMA_9'] < penultimo['EMA_21'] and ema9 > ema21:
            if (ema21 < ema100 and abs(ema9 - ema100) / ema100 < 0.01) and rsi > 50 and macd > macd_signal:
                presegnale = "📡 Presegnale: incrocio EMA9↑EMA21 vicino a EMA100 (BUY anticipato)"
        elif penultimo['EMA_9'] > penultimo['EMA_21'] and ema9 < ema21:
            if (ema21 > ema100 and abs(ema9 - ema100) / ema100 < 0.01) and rsi < 50 and macd < macd_signal:
                presegnale = "📡 Presegnale: incrocio EMA9↓EMA21 vicino a EMA100 (SELL anticipato)"
        if presegnale:
            note.append(presegnale)

    # Conteggio trend attivo
    candele_trend = conta_candele_trend(hist, rialzista=(segnale == "BUY"))

    if segnale in ["BUY", "SELL"]:
        trend_msg = f"📊 Trend: Attivo da {candele_trend} candele | Distanza tra medie: {dist_level}"
        note.insert(0, trend_msg)
        if forza_trend:
            note.insert(1, forza_trend)

        # Verifica figura candlestick
        pattern = riconosci_pattern_candela(hist)
        if candele_trend >= 3 and pattern:
            note.append(f"✅ {pattern} + trend confermato da 3+ candele, possibile ingresso.")
        elif candele_trend == 2:
            note.append("🔄 Trend in formazione, attendere conferma.")

    elif segnale == "HOLD":
        if forza_trend:
            note.insert(0, forza_trend)

        if not presegnale:
            if candele_trend <= 1 and not (ema9 > ema21 > ema100):
                note.append("⛔️ Trend esaurito, considera chiusura posizione")
            elif ema9 > ema21 > ema100 and candele_trend <= 2 and "Trend in indebolimento" not in forza_trend:
                note.append("➖ Trend ancora attivo ma debole")

    commento = "\n".join(note).strip()
    return segnale, hist, dist_attuale, commento, tp, sl, supporto

    
@app.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    try:
        now = int(time.time() * 1000)

        # Calcola end_time SOLO per 15m per analisi stabile
        end_time_15m = now - (now % (15 * 60 * 1000)) - 1

        df_1m = get_binance_df(symbol, "1m", 300)  # aggiornamento live
        df_5m = get_binance_df(symbol, "5m", 300)  # aggiornamento live
        df_15m = get_binance_df(symbol, "15m", 100)

        # Analisi principale su 1m e 5m con EMA 7, 25, 99
        def analizza_breve(df):
            df['EMA_7'] = df['close'].ewm(span=7).mean()
            df['EMA_25'] = df['close'].ewm(span=25).mean()
            df['EMA_99'] = df['close'].ewm(span=99).mean()
            e7 = df['EMA_7'].iloc[-1]
            e25 = df['EMA_25'].iloc[-1]
            e99 = df['EMA_99'].iloc[-1]
            if e7 > e25 > e99:
                return "BUY"
            elif e7 < e25 < e99:
                return "SELL"
            return "HOLD"

        segnale_1m = analizza_breve(df_1m)
        segnale_5m = analizza_breve(df_5m)

        # Analisi tecnica dettagliata su 15m (EMA 9, 21, 100)
        segnale_15m, hist_15m, distanza, note15, tp, sl, supporto = analizza_trend(df_15m)

        # Conferma se tutti allineati
        conferma = (segnale_1m == segnale_5m == segnale_15m and segnale_15m != "HOLD")

        ultima = hist_15m.index[-1].to_pydatetime().replace(second=0, microsecond=0, tzinfo=utc)
        close = round(hist_15m['close'].iloc[-1], 4)
        rsi = round(hist_15m['RSI'].iloc[-1], 2)
        ema9 = round(hist_15m['EMA_9'].iloc[-1], 2)
        ema21 = round(hist_15m['EMA_21'].iloc[-1], 2)
        ema100 = round(hist_15m['EMA_100'].iloc[-1], 2)
        atr = round(hist_15m['ATR'].iloc[-1], 2)
        macd = round(hist_15m['MACD'].iloc[-1], 4)
        macd_signal = round(hist_15m['MACD_SIGNAL'].iloc[-1], 4)

        orario_utc = ultima.strftime("%H:%M UTC")
        orario_roma = ultima.astimezone(timezone("Europe/Rome")).strftime("%H:%M ora italiana")
        data = ultima.strftime("(%d/%m)")
        ritardo = f"🕒 Dati riferiti alla candela chiusa alle {orario_utc} / {orario_roma} {data}"

        if segnale_15m in ["BUY", "SELL"]:
            tp_pct = round(((tp - close) / close) * 100, 1)
            sl_pct = round(((sl - close) / close) * 100, 1)
            commento = (
                f"{'🟢 BUY' if segnale_15m == 'BUY' else '🔴 SELL'} | {symbol.upper()} @ {close}$\n"
                f"🎯 {tp} ({tp_pct}%)   🛡 {sl} ({sl_pct}%)\n"
                f"RSI: {rsi}  |  EMA: {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal}  |  ATR: {atr}\n"
                f"{note15}\n"
                f"{'✅ Conferma su 1m e 5m' if conferma else '⚠️ Trend 15m, ma 1m/5m non confermano'}\n"
                f"{ritardo}"
            )
        else:
            commento = (
                f"⚠️ Nessun segnale confermato su 15m\n"
                f"RSI: {rsi}  |  EMA: {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal}  |  ATR: {atr}\n"
                f"📉 Supporto: {supporto}$\n"
                f"{note15}\n"
                f"{ritardo}"
            )

        commento = "\n".join([r.strip() for r in commento.splitlines() if r.strip()])

        return SignalResponse(
            segnale=segnale_15m,
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
            timeframe="15m"
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
_symbol_cache = {"time": 0, "data": []}

def get_best_symbols(limit=25):
    now = time.time()
    if now - _symbol_cache["time"] < 900:  # 15 minuti di cache
        return _symbol_cache["data"]

    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        usdt_pairs = [d for d in data if d["symbol"].endswith("USDT") and not any(x in d["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)

        top_symbols = [d["symbol"] for d in sorted_pairs[:limit]]
        _symbol_cache["time"] = now
        _symbol_cache["data"] = top_symbols
        return top_symbols

    except Exception as e:
        print("Errore nel recupero dei simboli dinamici:", e)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT",
            "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "MATICUSDT"
        ]

_hot_cache = {"time": 0, "data": []}

@app.get("/hotassets")
def hot_assets():
    now = time.time()
    if now - _hot_cache["time"] < 30:
        return _hot_cache["data"]

    symbols = get_best_symbols(limit=25)
    risultati = []

    for symbol in symbols:
        try:
            df = get_binance_df(symbol, "1m", 100)
            if df.empty or len(df) < 30:
                continue

            df['EMA_9'] = df['close'].ewm(span=9).mean()
            df['EMA_21'] = df['close'].ewm(span=21).mean()
            df['EMA_100'] = df['close'].ewm(span=100).mean()
            df['RSI'] = calcola_rsi(df['close'])

            df['H-L'] = df['high'] - df['low']
            df['H-PC'] = abs(df['high'] - df['close'].shift())
            df['L-PC'] = abs(df['low'] - df['close'].shift())
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            atr = df['ATR'].iloc[-1]
            dist_medie = abs(df['EMA_9'].iloc[-1] - df['EMA_21'].iloc[-1]) + abs(df['EMA_21'].iloc[-1] - df['EMA_100'].iloc[-1])

            # Nuova logica segnali
            sub_signals = []
            for i in range(-40, -4):
                sub_df = df.iloc[i - 4:i + 1].copy()
                sub_df['EMA_9'] = sub_df['close'].ewm(span=9).mean()
                sub_df['EMA_21'] = sub_df['close'].ewm(span=21).mean()
                sub_df['EMA_100'] = sub_df['close'].ewm(span=100).mean()
                sub_df['RSI'] = calcola_rsi(sub_df['close'])

                e9 = sub_df['EMA_9'].iloc[-1]
                e21 = sub_df['EMA_21'].iloc[-1]
                e100 = sub_df['EMA_100'].iloc[-1]
                rsi = sub_df['RSI'].iloc[-1]

                if e9 > e21 > e100 and rsi > 50:
                    sub_signals.append("BUY")
                elif e9 < e21 < e100 and rsi < 50:
                    sub_signals.append("SELL")

            buy_signals = sub_signals.count("BUY")
            sell_signals = sub_signals.count("SELL")

            candele_attive = conta_candele_trend(df, rialzista=(buy_signals > sell_signals))

            # Determinazione del trend con messaggio unico
            if buy_signals > sell_signals:
                trend = "BUY"
            elif sell_signals > buy_signals:
                trend = "SELL"
            else:
                trend = "NEUTRO"

            # Salva risultato se almeno 1 segnale significativo
            if (buy_signals + sell_signals >= 1) or (atr > 0.5 and dist_medie > 1):
                ultimo = df.iloc[-1]
                risultati.append({
                    "symbol": symbol,
                    "segnali": buy_signals + sell_signals,
                    "trend": trend,
                    "rsi": round(ultimo['RSI'], 2),
                    "ema9": round(ultimo['EMA_9'], 2),
                    "ema21": round(ultimo['EMA_21'], 2),
                    "ema100": round(ultimo['EMA_100'], 2),
                    "candele_trend": candele_attive
                })

        except Exception as e:
            print(f"Errore con {symbol}: {e}")
            continue

    risultati_ordinati = sorted(risultati, key=lambda x: x['segnali'], reverse=True)[:10]
    _hot_cache["time"] = now
    _hot_cache["data"] = risultati_ordinati
    return risultati_ordinati
