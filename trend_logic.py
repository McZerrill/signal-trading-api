import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema

MODALITA_TEST = True

def valuta_distanza(d):
    if d < 0.002: return "🔹 Bassa"
    elif d < 0.005: return "🔸 Media"
    return "🔺 Alta"

def conta_candele_trend(hist: pd.DataFrame, rialzista: bool = True, max_candele: int = 20) -> int:
    count = 0
    for i in range(-1, -max_candele - 1, -1):
        e7, e25, e99 = hist['EMA_7'].iloc[i], hist['EMA_25'].iloc[i], hist['EMA_99'].iloc[i]
        if rialzista and (e7 > e25):
            count += 1
        elif not rialzista and (e7 < e25):
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
        return "🪓 Hammer"
    if corpo < 0 and ombra_sup >= 2 * abs(corpo) and ombra_inf <= abs(corpo) * 0.3:
        return "🌠 Shooting Star"
    if corpo > 0 and c['close'] > df['open'].iloc[-2] and c['open'] < df['close'].iloc[-2]:
        return "🔄 Bullish Engulfing"
    if corpo < 0 and c['close'] < df['open'].iloc[-2] and c['open'] > df['close'].iloc[-2]:
        return "🔃 Bearish Engulfing"
    return ""

def analizza_trend(hist: pd.DataFrame, spread: float = 0.0):
    hist = hist.copy()
    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    if len(hist) < 22:
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]
    antepenultimo = hist.iloc[-3]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    note = []

    volume_soglia = 120 if MODALITA_TEST else 300
    atr_minimo = 0.0015 if MODALITA_TEST else 0.001
    distanza_minima = 0.0012 if MODALITA_TEST else 0.0015
    macd_rsi_range = (45, 55)
    macd_signal_threshold = 0.0005 if MODALITA_TEST else 0.001

    if atr / close < atr_minimo:
        note.append("⚠️ ATR basso: bassa volatilità")

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    if volume_attuale < volume_medio * (volume_soglia / 100):
        note.append("⚠️ Volume inferiore alla media")

    distanza_ema = abs(ema7 - ema25)
    dist_level = valuta_distanza(distanza_ema)
    if distanza_ema < distanza_minima:
        note.append("ℹ️ EMA7 e EMA25 molto vicine")
    if abs(macd - macd_signal) < macd_signal_threshold:
        note.append("ℹ️ MACD e signal quasi sovrapposti")
    if macd_rsi_range[0] <= rsi <= macd_rsi_range[1]:
        note.append("ℹ️ RSI neutro")

    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    corpo_candela = abs(ultimo['close'] - ultimo['open'])

    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout rialzista con volume")
        if corpo_candela > atr:
            note.append("🚀 Spike rialzista oltre ATR")
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout ribassista con volume")
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout con volume debole")

    segnale = "HOLD"
    tp = sl = 0.0

    ultime8 = hist.iloc[-8:]
    incrocio_buy = any(ultime8['EMA_7'] > ultime8['EMA_25']) and penultimo['EMA_7'] <= penultimo['EMA_25'] and ema7 > ema25
    incrocio_sell = any(ultime8['EMA_7'] < ultime8['EMA_25']) and penultimo['EMA_7'] >= penultimo['EMA_25'] and ema7 < ema25

    if incrocio_buy:
        segnale = "BUY"
        investimento = 100
        commissione = 0.1
        guadagno_target = 0.5
        commissioni = investimento * 2 * (commissione / 100)
        rendimento_lordo = (guadagno_target + commissioni) / investimento
        tp = round(close * (1 + rendimento_lordo) / (1 - spread / 100), 4)
        sl = round(close * (1 - rendimento_lordo) / (1 - spread / 100), 4)
        note.append("✅ BUY: incrocio EMA7 > EMA25")

    if incrocio_sell:
        segnale = "SELL"
        investimento = 100
        commissione = 0.1
        guadagno_target = 0.5
        commissioni = investimento * 2 * (commissione / 100)
        rendimento_lordo = (guadagno_target + commissioni) / investimento
        tp = round(close / ((1 + rendimento_lordo) * (1 + spread / 100)), 4)
        sl = round(close / ((1 - rendimento_lordo) * (1 - spread / 100)), 4)
        note.append("✅ SELL: incrocio EMA7 < EMA25")

    pattern = riconosci_pattern_candela(hist)
    if segnale in ["BUY", "SELL"]:
        n_candele = conta_candele_trend(hist, rialzista=(segnale == "BUY"))
        note.insert(0, f"📊 Trend {segnale.lower()} da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick: {pattern}")
    else:
        candele_up = conta_candele_trend(hist, rialzista=True)
        candele_down = conta_candele_trend(hist, rialzista=False)
        if ema7 > ema25 and candele_up <= 1:
            note.append("⚠️ Rialzo debole o finito")
        elif ema7 < ema25 and candele_down <= 1:
            note.append("⚠️ Ribasso debole o finito")

        if incrocio_buy:
            note.append("🔍 Presegnale BUY: EMA7 ha appena incrociato EMA25")
        if incrocio_sell:
            note.append("🔍 Presegnale SELL: EMA7 ha appena incrociato EMA25 al ribasso")

    if segnale == "BUY" and pattern and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"]):
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")

    if ema7 > ema25:
        if close < ema7 and close >= ema25:
            note.append("🔁 Pullback nel trend rialzista")
    elif ema7 < ema25:
        if close > ema7 and close <= ema25:
            note.append("🔁 Pullback nel trend ribassista")

    if segnale == "BUY" and (rsi < 50 or macd < macd_signal):
        note.append("❌ RSI o MACD non confermano BUY")
    if segnale == "SELL" and (rsi > 50 or macd > macd_signal):
        note.append("❌ RSI o MACD non confermano SELL")

    note = list(dict.fromkeys(note))
    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
