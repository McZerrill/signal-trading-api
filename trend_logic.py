
import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema

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
    hist = hist.copy()

    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]
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

    candele_trend_buy = conta_candele_trend(hist, rialzista=True)
    candele_trend_sell = conta_candele_trend(hist, rialzista=False)
    pattern = riconosci_pattern_candela(hist)

    # Conferma BUY
    if (
        penultimo['EMA_7'] < penultimo['EMA_25'] < penultimo['EMA_99']
        and ema7 > ema25 > ema99
        and rsi > 50 and macd > macd_signal
        and dist_diff > 0 and candele_trend_buy >= 3
    ):
        segnale = "BUY"
        condizioni_verificate = 5
        resistenza = hist['high'].tail(20).max()
        tp = round(min(close + atr * 1.5, resistenza), 4)
        sl = round(close - atr * 1.2, 4)
        note.append(f"📊 Trend BUY attivo da {candele_trend_buy} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Conferma con pattern: {pattern}")

    # Conferma SELL
    elif (
        penultimo['EMA_7'] > penultimo['EMA_25'] > penultimo['EMA_99']
        and ema7 < ema25 < ema99
        and rsi < 50 and macd < macd_signal
        and dist_diff > 0 and candele_trend_sell >= 3
    ):
        segnale = "SELL"
        condizioni_verificate = 5
        tp = round(max(close - atr * 1.5, supporto), 4)
        sl = round(close + atr * 1.2, 4)
        note.append(f"📊 Trend SELL attivo da {candele_trend_sell} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Conferma con pattern: {pattern}")

    # Presegnali
    elif penultimo['EMA_7'] < penultimo['EMA_25'] and ema7 > ema25 and ema25 < ema99:
        if abs(ema7 - ema99) / ema99 < 0.015:
            note.append("🟢 Presegnale BUY: EMA7 incrocia EMA25 sotto EMA99")
    elif penultimo['EMA_7'] > penultimo['EMA_25'] and ema7 < ema25 and ema25 > ema99:
        if abs(ema7 - ema99) / ema99 < 0.015:
            note.append("🔴 Presegnale SELL: EMA7 incrocia EMA25 sopra EMA99")

    # Diagnostica finale
    if segnale == "HOLD":
        if ema7 > ema25 > ema99 and candele_trend_buy < 3:
            note.append("🟡 Trend in formazione (BUY debole)")
        elif ema7 < ema25 < ema99 and candele_trend_sell < 3:
            note.append("🟡 Trend in formazione (SELL debole)")
        else:
            note.append("⚠️ Trend assente o terminato")

    commento = "
".join(note).strip()
    return segnale, hist, dist_attuale, commento, tp, sl, supporto
