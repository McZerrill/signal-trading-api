import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema

MODALITA_TEST = False

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
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0, 0.0

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    note = []

    volume_multiplier = 0.6 if MODALITA_TEST else 1.0
    macd_threshold = 0.0005 if MODALITA_TEST else 0.0015
    ema_gap_threshold = 0.0003 if MODALITA_TEST else 0.001
    breakout_volume_factor = 1.0 if MODALITA_TEST else 1.5

    if atr / close < 0.001:
        note.append("⚠️ Nessun segnale: ATR troppo basso rispetto al prezzo")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto, 0.0

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    if volume_attuale < volume_medio * volume_multiplier:
        note.append("⚠️ Volume attuale sotto la soglia, possibile segnale debole")
        if not MODALITA_TEST:
            return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto, 0.0

    dist_attuale = abs(ema7 - ema25) + abs(ema25 - ema99)
    dist_precedente = abs(penultimo['EMA_7'] - penultimo['EMA_25']) + abs(penultimo['EMA_25'] - penultimo['EMA_99'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    segnale = "HOLD"
    tp = sl = 0.0

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    pattern = riconosci_pattern_candela(hist)
    macd_gap = macd - macd_signal

    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    if close > massimo_20 and volume_attuale > volume_medio * breakout_volume_factor:
        note.append("💥 Breakout rialzista confermato")
    elif close < minimo_20 and volume_attuale > volume_medio * breakout_volume_factor:
        note.append("💥 Breakout ribassista confermato")
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume non sufficiente a confermare")

    commissione = 0.001  # 0.1% per lato
    guadagno_target = 0.5  # profitto netto desiderato in USDC
    costi_totali = (commissione * 2) + spread
    percentuale_guadagno = (guadagno_target / 100) + costi_totali  # su 100 USDC

    if ema7 > ema25 and abs(ema7 - ema25) / close > ema_gap_threshold:
        if ema7 > ema99 and ema25 > ema99:
            if rsi > 50 and macd > macd_signal and macd_gap > macd_threshold:
                segnale = "BUY"
                tp = round(close * (1 + percentuale_guadagno), 4)
                sl = round(close * (1 - percentuale_guadagno / 1.5), 4)
                note.append("✅ BUY confermato: trend forte")

    if ema7 < ema25 and abs(ema7 - ema25) / close > ema_gap_threshold:
        if ema7 < ema99 and ema25 < ema99:
            if rsi < 48 and macd < macd_signal and macd_gap < -macd_threshold:
                segnale = "SELL"
                tp = round(close * (1 - percentuale_guadagno), 4)
                sl = round(close * (1 + percentuale_guadagno / 1.5), 4)
                note.append("✅ SELL confermato: trend forte")

    if segnale in ["BUY", "SELL"]:
        n_candele = candele_trend_up if segnale == "BUY" else candele_trend_down
        note.insert(0, f"📊 Trend attivo da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick rilevato: {pattern}")
    else:
        if trend_up and candele_trend_up <= 2:
            note.append("🟡 Trend attivo ma debole")
        elif trend_down and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista ma debole")
        elif candele_trend_up <= 1 and not trend_up:
            note.append("⚠️ Trend terminato")

    if segnale == "BUY" and pattern and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"]):
        note.append("⚠️ Pattern ribassista rilevato: possibile inversione")
        segnale = "HOLD"
    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append("⚠️ Pattern Hammer rilevato: possibile inversione")
        segnale = "HOLD"

    guadagno_netto = 0.0

    return segnale, hist, dist_attuale, "\n".join(note).strip(), tp, sl, supporto, guadagno_netto
