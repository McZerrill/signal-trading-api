# trend_logic.py

import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto

def valuta_distanza(distanza: float) -> str:
    if distanza < 1:
        return "bassa"
    elif distanza < 3:
        return "media"
    else:
        return "alta"

def conta_candele_trend(hist: pd.DataFrame, rialzista: bool = True, max_candele: int = 20) -> int:
    count = 0
    for i in range(-1, -max_candele - 1, -1):
        e9, e21, e100 = hist['EMA_9'].iloc[i], hist['EMA_21'].iloc[i], hist['EMA_100'].iloc[i]
        if rialzista and (e9 > e21 > e100):
            count += 1
        elif not rialzista and (e9 < e21 < e100):
            count += 1
        else:
            break
    return count

def riconosci_pattern_candela(df: pd.DataFrame) -> str:
    c = df.iloc[-1]
    o, h, l, close = c['open'], c['high'], c['low'], c['close']
    corpo = abs(close - o)
    ombra_sup = h - max(o, close)
    ombra_inf = min(o, close) - l

    if corpo == 0:
        return ""

    if corpo > 0 and ombra_inf >= 2 * corpo and ombra_sup <= corpo * 0.3:
        return "🪓 Hammer rilevato (BUY)"
    if corpo > 0 and ombra_sup >= 2 * corpo and ombra_inf <= corpo * 0.3:
        return "🌠 Shooting Star rilevato (SELL)"
    return ""

def analizza_trend(hist: pd.DataFrame):
    hist['EMA_9'] = hist['close'].ewm(span=9).mean()
    hist['EMA_21'] = hist['close'].ewm(span=21).mean()
    hist['EMA_100'] = hist['close'].ewm(span=100).mean()
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    if len(hist) < 5:
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema9, ema21, ema100 = ultimo['EMA_9'], ultimo['EMA_21'], ultimo['EMA_100']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    dist_attuale = abs(ema9 - ema21) + abs(ema21 - ema100)
    dist_precedente = abs(penultimo['EMA_9'] - penultimo['EMA_21']) + abs(penultimo['EMA_21'] - penultimo['EMA_100'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    note = []
    segnale = "HOLD"
    tp = sl = 0.0

    # Forza del trend
    forza_trend = ""
    if ema9 > ema21 > ema100:
        forza_trend = "📈 Trend forte in espansione" if dist_diff > 0 else "➖ Trend stabile"
    elif ema9 < ema21 < ema100:
        forza_trend = "⛔️ Trend ribassista"
    elif ema9 < ema21 and ema21 > ema100 and ema9 > ema100:
        forza_trend = "⛔️ Trend in esaurimento"
    elif ema9 > ema21 and penultimo['EMA_9'] < penultimo['EMA_21']:
        forza_trend = "🔁 Trend ripreso"

    # Verifica coerenza ultime candele
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

    # Presegnali
    if segnale == "HOLD":
        if penultimo['EMA_9'] < penultimo['EMA_21'] and ema9 > ema21:
            if ema21 < ema100 and abs(ema9 - ema100) / ema100 < 0.01 and rsi > 50 and macd > macd_signal:
                note.append("📡 Presegnale: incrocio EMA9↑EMA21 vicino EMA100 (BUY anticipato)")
        elif penultimo['EMA_9'] > penultimo['EMA_21'] and ema9 < ema21:
            if ema21 > ema100 and abs(ema9 - ema100) / ema100 < 0.01 and rsi < 50 and macd < macd_signal:
                note.append("📡 Presegnale: incrocio EMA9↓EMA21 vicino EMA100 (SELL anticipato)")

    # Conteggio trend attivo
    candele_trend = conta_candele_trend(hist, rialzista=(segnale == "BUY"))
    pattern = riconosci_pattern_candela(hist)

    if segnale in ["BUY", "SELL"]:
        note.insert(0, f"📊 Trend: Attivo da {candele_trend} candele | Distanza: {dist_level}")
        if forza_trend:
            note.insert(1, forza_trend)
        if candele_trend >= 3 and pattern:
            note.append(f"✅ {pattern} + trend confermato")
        elif candele_trend == 2:
            note.append("🔄 Trend in formazione")
    else:
        if forza_trend:
            note.insert(0, forza_trend)
        if not note and candele_trend <= 1 and not (ema9 > ema21 > ema100):
            note.append("⛔
