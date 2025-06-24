import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema

MODALITA_TEST = True

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

def rileva_pattern_v(hist: pd.DataFrame) -> bool:
    if len(hist) < 4:
        return False
    sub = hist.iloc[-4:]
    rsi_start = sub['RSI'].iloc[0]
    rsi_end = sub['RSI'].iloc[-1]
    macd = sub['MACD'].iloc[-1]
    pattern = False

    for i in range(-3, 0):
        rossa = sub.iloc[i]['close'] < sub.iloc[i]['open']
        verde = sub.iloc[i+1]['close'] > sub.iloc[i+1]['open']
        corpo_rossa = abs(sub.iloc[i]['close'] - sub.iloc[i]['open'])
        corpo_verde = abs(sub.iloc[i+1]['close'] - sub.iloc[i+1]['open'])
        if rossa and verde and corpo_verde >= corpo_rossa:
            pattern = True
            break

    return pattern and rsi_start < 30 and rsi_end > 50 and abs(macd) < 0.01

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
        note.append("⚠️ ATR troppo basso: mercato poco volatile")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    if volume_attuale < volume_medio * (volume_soglia / 100):
        note.append("⚠️ Volume basso: segnale debole")
        if not MODALITA_TEST:
            return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    distanza_ema = abs(ema7 - ema25)
    curvatura_ema25 = ema25 - penultimo['EMA_25']
    curvatura_precedente = penultimo['EMA_25'] - antepenultimo['EMA_25']
    accelerazione = curvatura_ema25 - curvatura_precedente
    dist_level = valuta_distanza(distanza_ema)

    segnale = "HOLD"
    tp = sl = 0.0

    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    corpo_candela = abs(ultimo['close'] - ultimo['open'])
    breakout_valido = False

    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout rialzista con volume alto")
        if corpo_candela > atr:
            note.append("🚀 Spike rialzista con breakout solido")
            breakout_valido = True
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout ribassista con volume alto")
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume insufficiente")

    incrocio_25_sopra_99 = ema25 > ema99 and penultimo['EMA_25'] <= penultimo['EMA_99']
    ema7_sopra_25 = ema7 > ema25
    ema7_sopra_99 = ema7 > ema99
    allargamento_buy = (ema7 - ema25) > (penultimo['EMA_7'] - penultimo['EMA_25'])

    if ema7_sopra_25 and ema7_sopra_99 and incrocio_25_sopra_99 and allargamento_buy:
        if rsi > macd_rsi_range[0] and (macd > macd_signal or (macd > 0 and macd - macd_signal > -0.005)):
            segnale = "BUY"
            investimento = 100
            commissione = 0.1
            guadagno_target = 0.5
            commissioni = investimento * 2 * (commissione / 100)
            rendimento_lordo = (guadagno_target + commissioni) / investimento
            tp = round(close * (1 + rendimento_lordo) / (1 - spread / 100), 4)
            sl = round(close * (1 - rendimento_lordo) / (1 - spread / 100), 4)
            note.append("✅ BUY confermato: incrocio progressivo + allargamento")

    incrocio_25_sotto_99 = ema25 < ema99 and penultimo['EMA_25'] >= penultimo['EMA_99']
    ema7_sotto_25 = ema7 < ema25
    ema7_sotto_99 = ema7 < ema99
    allargamento_sell = (ema25 - ema7) > (penultimo['EMA_25'] - penultimo['EMA_7'])

    if ema7_sotto_25 and ema7_sotto_99 and incrocio_25_sotto_99 and allargamento_sell:
        if rsi < macd_rsi_range[1] and (macd < macd_signal or (macd < 0 and macd - macd_signal < 0.005)):
            segnale = "SELL"
            investimento = 100
            commissione = 0.1
            guadagno_target = 0.5
            commissioni = investimento * 2 * (commissione / 100)
            rendimento_lordo = (guadagno_target + commissioni) / investimento
            tp = round(close / ((1 + rendimento_lordo) * (1 + spread / 100)), 4)
            sl = round(close / ((1 - rendimento_lordo) * (1 - spread / 100)), 4)
            note.append("✅ SELL confermato: incrocio progressivo + allargamento")

    if segnale == "HOLD" and rileva_pattern_v(hist):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr, 4)
        note.append("📈 Pattern V rilevato: BUY da inversione rapida")

    pattern = riconosci_pattern_candela(hist)

    if segnale in ["BUY", "SELL"]:
        n_candele = conta_candele_trend(hist, rialzista=(segnale == "BUY"))
        note.insert(0, f"📊 Trend attivo da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick rilevato: {pattern}")
    else:
        candele_trend_up = conta_candele_trend(hist, rialzista=True)
        candele_trend_down = conta_candele_trend(hist, rialzista=False)
        if ema7 > ema25 > ema99 and candele_trend_up <= 2:
            note.append("🟡 Trend attivo ma debole")
        elif ema7 < ema25 < ema99 and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista ma debole")
        elif candele_trend_up <= 1 and not (ema7 > ema25 > ema99):
            note.append("⚠️ Trend concluso: attenzione a inversioni")

    if segnale == "BUY" and pattern and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"]):
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
        segnale = "HOLD"
    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
        segnale = "HOLD"

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
