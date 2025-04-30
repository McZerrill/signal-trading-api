
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

    if len(hist) < 10:
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    distanze = {
        "attuale": abs(ema7 - ema25) + abs(ema25 - ema99),
        "precedente": abs(penultimo['EMA_7'] - penultimo['EMA_25']) + abs(penultimo['EMA_25'] - penultimo['EMA_99'])
    }
    dist_diff = distanze["attuale"] - distanze["precedente"]
    distanza_trend = valuta_distanza(distanze["attuale"])

    note = []
    condizioni = 0
    segnale = "HOLD"
    tp = sl = 0.0

    incrocio_macd_buy = penultimo['MACD'] < penultimo['MACD_SIGNAL'] and macd > macd_signal
    incrocio_macd_sell = penultimo['MACD'] > penultimo['MACD_SIGNAL'] and macd < macd_signal

    incrocio_ema_buy = penultimo['EMA_7'] < penultimo['EMA_25'] and ema7 > ema25
    incrocio_ema_sell = penultimo['EMA_7'] > penultimo['EMA_25'] and ema7 < ema25

    vicino_a_ema99_buy = ema25 < ema99 and abs(ema7 - ema99) / ema99 < 0.015
    vicino_a_ema99_sell = ema25 > ema99 and abs(ema7 - ema99) / ema99 < 0.015

    if incrocio_ema_buy and vicino_a_ema99_buy and incrocio_macd_buy and rsi > 50:
        condizioni = 4
        if distanze["attuale"] > 2.5:
            note.append("⚠️ Trend già partito, entrare ora è rischioso")
        else:
            segnale = "BUY"
            condizioni = 5
            resistenza = hist["high"].tail(20).max()
            tp = round(min(close + atr * 1.5, resistenza), 4)
            sl = round(close - atr * 1.2, 4)
            note.append("🟢 Presegnale BUY con breakout in formazione")

    elif incrocio_ema_sell and vicino_a_ema99_sell and incrocio_macd_sell and rsi < 50:
        condizioni = 4
        if distanze["attuale"] > 2.5:
            note.append("⚠️ Trend già partito, entrare ora è rischioso")
        else:
            segnale = "SELL"
            condizioni = 5
            tp = round(max(close - atr * 1.5, supporto), 4)
            sl = round(close + atr * 1.2, 4)
            note.append("🔴 Presegnale SELL con breakout in formazione")

    # Pattern conferma
    pattern = riconosci_pattern_candela(hist)
    if segnale in ["BUY", "SELL"]:
        candele_trend = conta_candele_trend(hist, rialzista=(segnale == "BUY"))
        note.insert(0, f"📊 Trend attivo da {candele_trend} candele | Distanza: {distanza_trend}")
        if candele_trend >= 3 and pattern:
            note.append(f"✅ Conferma con pattern: {pattern}")
        elif candele_trend == 2:
            note.append("🔄 Trend in formazione")

    elif condizioni >= 3:
        note.append("🟡 Presegnale attivo ma mancano conferme")
    else:
        note.append("⚠️ Nessuna condizione forte rilevata")

    commento = "\n".join(note).strip()
    return segnale, hist, distanze["attuale"], commento, tp, sl, supporto


