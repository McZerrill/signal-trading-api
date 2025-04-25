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
        e7, e25, e99 = hist['EMA_7'].iloc[i], hist['EMA_25'].iloc[i], hist['EMA_99'].iloc[i]
        if rialzista and (e7 > e25 > e99):
            count += 1
        elif not rialzista and (e7 < e25 < e99):
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
    hist['EMA_7'] = hist['close'].ewm(span=7).mean()
    hist['EMA_25'] = hist['close'].ewm(span=25).mean()
    hist['EMA_99'] = hist['close'].ewm(span=99).mean()
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    if len(hist) < 5:
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    dist_attuale = abs(ema7 - ema25) + abs(ema25 - ema99)
    dist_precedente = abs(penultimo['EMA_7'] - penultimo['EMA_25']) + abs(penultimo['EMA_25'] - penultimo['EMA_99'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    note = []
    segnale = "HOLD"
    tp = sl = 0.0
    condizioni_verificate = 0

    # Forza del trend
    forza_trend = ""
    if ema7 > ema25 > ema99:
        forza_trend = "📈 Trend forte in espansione" if dist_diff > 0 else "➖ Trend stabile"
    elif ema7 < ema25 < ema99:
        forza_trend = "⛔️ Trend ribassista"
    elif ema7 < ema25 and ema25 > ema99 and ema7 > ema99:
        forza_trend = "⛔️ Trend in esaurimento"
    elif ema7 > ema25 and penultimo['EMA_7'] < penultimo['EMA_25']:
        forza_trend = "🔁 Trend ripreso"

    # Ultime 3 candele coerenti
    recent_trend_up = all(hist['EMA_7'].iloc[-i] > hist['EMA_25'].iloc[-i] for i in range(1, 4))
    recent_trend_down = all(hist['EMA_7'].iloc[-i] < hist['EMA_25'].iloc[-i] for i in range(1, 4))

    # BUY
    if (
        penultimo['EMA_7'] < penultimo['EMA_25'] < penultimo['EMA_99']
        and ema7 > ema25 > ema99
        and rsi > 50 and macd > macd_signal
        and recent_trend_up and dist_diff > 0
    ):
        segnale = "BUY"
        condizioni_verificate = 5
        resistenza = hist['high'].tail(20).max()
        tp = round(min(close + atr * 1.5, resistenza), 4)
        sl = round(close - atr * 1.2, 4)

    # SELL
    elif (
        penultimo['EMA_7'] > penultimo['EMA_25'] > penultimo['EMA_99']
        and ema7 < ema25 < ema99
        and rsi < 50 and macd < macd_signal
        and recent_trend_down and dist_diff > 0
    ):
        segnale = "SELL"
        condizioni_verificate = 5
        tp = round(max(close - atr * 1.5, supporto), 4)
        sl = round(close + atr * 1.2, 4)

    # Presegnali
    if segnale == "HOLD":
        if penultimo['EMA_7'] < penultimo['EMA_25'] and ema7 > ema25:
            if ema25 < ema99 and abs(ema7 - ema99) / ema99 < 0.01:
                condizioni_verificate += 1
                if rsi > 50: condizioni_verificate += 1
                if macd > macd_signal: condizioni_verificate += 1
                note.append("🟢🗣 Presegnale: EMA7↑EMA25 vicino EMA99 (BUY)")
        elif penultimo['EMA_7'] > penultimo['EMA_25'] and ema7 < ema25:
            if ema25 > ema99 and abs(ema7 - ema99) / ema99 < 0.01:
                condizioni_verificate += 1
                if rsi < 50: condizioni_verificate += 1
                if macd < macd_signal: condizioni_verificate += 1
                note.append("🔴🗣 Presegnale: EMA7↓EMA25 vicino EMA99 (SELL)")

    # Candele in trend + pattern
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
        if not note and candele_trend <= 1 and not (ema7 > ema25 > ema99):
            note.append("⛔️ Trend esaurito, considera chiusura")
        elif ema7 > ema25 > ema99 and candele_trend <= 2:
            note.append("➖ Trend ancora attivo ma debole")

    if 0 < condizioni_verificate < 5:
        note.append(f"⚙️ Segnale in attesa: {condizioni_verificate}/5 condizioni verificate")

    commento = "\n".join(note).strip()
    return segnale, hist, dist_attuale, commento, tp, sl, supporto
