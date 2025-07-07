import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema
from indicators import calcola_percentuale_guadagno


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

    note = []
    tp = 0.0
    sl = 0.0
    distanza_ema = 0.0
    segnale = "HOLD"
    
    supporto = calcola_supporto(hist)
    
    if len(hist) < 22:
        note.append("⚠️ Dati insufficienti per analisi")
        return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]
    antepenultimo = hist.iloc[-3]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']

    variazione = (hist['close'].iloc[-1] - hist['close'].iloc[-4]) / hist['close'].iloc[-4] * 100


    investimento = 100.0
    guadagno_netto_target = 0.5
    commissione = 0.1

    volume_soglia = 200 if MODALITA_TEST else 300
    atr_minimo = 0.0008 if MODALITA_TEST else 0.001
    distanza_minima = 0.0012 if MODALITA_TEST else 0.0015
    macd_signal_threshold = 0.0005 if MODALITA_TEST else 0.001

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()

    bloccato = False
    if atr / close < atr_minimo:
        note.append("⚠️ ATR troppo basso: mercato poco volatile")
        bloccato = True

    if volume_attuale < volume_medio * (volume_soglia / 100):
        note.append("⚠️ Volume basso: segnale debole")
        bloccato = True

    if bloccato:
        return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    distanza_ema = abs(ema7 - ema25)
    curvatura_ema25 = ema25 - penultimo['EMA_25']
    curvatura_precedente = penultimo['EMA_25'] - antepenultimo['EMA_25']
    accelerazione = curvatura_ema25 - curvatura_precedente

    dist_level = valuta_distanza(distanza_ema)

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    recupero_buy = ema7 > ema25 and close > ema25 and ema25 > penultimo['EMA_25']
    recupero_sell = ema7 < ema25 and close < ema25 and ema25 < penultimo['EMA_25']

    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    pattern = riconosci_pattern_candela(hist)
    macd_gap = macd - macd_signal

    breakout_valido = False
    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    corpo_candela = abs(ultimo['close'] - ultimo['open'])
    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout rialzista con volume alto")
        if corpo_candela > atr:
            note.append("🚀 Spike rialzista con breakout solido")
            breakout_valido = True
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout ribassista con volume alto")
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume insufficiente")

    macd_buy_ok = macd > macd_signal and macd_gap > macd_signal_threshold
    macd_buy_debole = macd > 0 and macd_gap > -0.005
    macd_sell_ok = macd < macd_signal and macd_gap < -macd_signal_threshold
    macd_sell_debole = macd < 0.01 and macd_gap < 0.005

    # BUY
    if (trend_up or recupero_buy or (breakout_valido and rsi > 40)) \
        and distanza_ema / close > distanza_minima \
        and (macd_buy_ok or macd_buy_debole) \
        and rsi > 50:

        if rsi > 75 and variazione > 1.0:
            note.append(f"⛔ RSI troppo alto per BUY in trend maturo (+{round(variazione, 2)}%)")
        elif ultimo['close'] < ultimo['open'] and macd_buy_ok is False:
            note.append("⛔ Candela attuale rossa: BUY rischioso")
        elif macd_gap < 0.0003:
            note.append("⛔ MACD troppo debole: no BUY")
        else:
            
            if trend_up and variazione > 0.9 and candele_trend_up > 1:
                note.append(f"⛔ Trend BUY già maturo (+{round(variazione, 2)}% in 3 candele): nessun segnale BUY")
            else:
                forza_trend = min(max(distanza_ema / close, 0.001), 0.01)
                if forza_trend < 0.0018:
                    note.append("⚠️ Trend BUY troppo debole: distanza EMA insufficiente")
                else:
                    segnale = "BUY"
                    coeff_tp = min(1.5 + (accelerazione * 10), 1.6)
                    coeff_sl = 1.0 - (accelerazione * 5)
                    delta_pct = calcola_percentuale_guadagno(guadagno_netto_target, investimento, spread, commissione)
                    atr_ratio = atr / close
                    bonus_tp_pct = 0.0
                    if atr_ratio > 0.003:
                        bonus_tp_pct = min((atr_ratio - 0.003) * 4, 0.01)
                    delta_price = close * delta_pct
                    delta_price_bonus = close * bonus_tp_pct
                    tp = round(close + delta_price * coeff_tp + delta_price_bonus, 4)
                    sl = round(close - delta_price * coeff_sl, 4)
                    tp_max = round(close * 1.008, 4)
                    if tp > tp_max:
                        tp = tp_max
                    if tp <= close:
                        tp = round(close * 1.005, 4)
                    if sl >= close:
                        sl = round(close * 0.995, 4)
                    rapporto_massimo = 1.7
                    delta_tp = abs(tp - close)
                    delta_sl = abs(sl - close)
                    if delta_sl > 0 and delta_tp / delta_sl > rapporto_massimo:
                        delta_tp = delta_sl * rapporto_massimo
                        tp = round(close + delta_tp, 4)
                        note.append(f"⚖️ TP ricalibrato per mantenere R/R ≤ {rapporto_massimo}")
                    note.append("✅ BUY confermato: trend forte" if macd_buy_ok else "⚠️ BUY anticipato: MACD ≈ signal")

    # SELL
    if (trend_down or recupero_sell) \
        and distanza_ema / close > distanza_minima \
        and rsi < 50 \
        and (macd_sell_ok or macd_sell_debole) \
        and abs(macd) > 0.0005:

        if rsi < 25 and variazione < -1.0:
            note.append(f"⛔ RSI troppo basso per SELL in trend maturo ({round(variazione, 2)}%)")
        elif abs(macd) < 0.0005:
            note.append("⛔ MACD troppo debole o piatto")
        elif distanza_ema / close < 0.0025:
            note.append("⛔ Distanza EMA insufficiente: trend SELL debole")
        else:
            
            if trend_down and variazione < -0.9 and candele_trend_down > 1:
                note.append(f"⛔ Trend SELL già maturo (-{round(abs(variazione), 2)}% in 3 candele): nessun segnale SELL")
            else:
                forza_trend = min(max(distanza_ema / close, 0.001), 0.01)
                if forza_trend < 0.0020:
                    note.append("⚠️ Trend SELL troppo debole: distanza EMA insufficiente")
                elif ultimo['close'] > ultimo['open']:
                    note.append("⛔ Candela attuale verde: SELL troppo rischioso")
                else:
                    segnale = "SELL"
                    coeff_tp = min(1.5 + (accelerazione * 10), 1.6)
                    coeff_sl = 1.0 - (accelerazione * 5)
                    delta_pct = calcola_percentuale_guadagno(guadagno_netto_target, investimento, spread, commissione)
                    atr_ratio = atr / close
                    bonus_tp_pct = 0.0
                    if atr_ratio > 0.003:
                        bonus_tp_pct = min((atr_ratio - 0.003) * 4, 0.01)
                    delta_price = close * delta_pct
                    delta_price_bonus = close * bonus_tp_pct
                    tp = round(close - delta_price * coeff_tp - delta_price_bonus, 4)
                    sl = round(close + delta_price * coeff_sl, 4)
                    tp_min = round(close * 0.992, 4)
                    if tp < tp_min:
                        tp = tp_min
                    if tp >= close:
                        tp = round(close * 0.995, 4)
                    if sl <= close:
                        sl = round(close * 1.005, 4)
                    rapporto_massimo = 1.7
                    delta_tp = abs(tp - close)
                    delta_sl = abs(sl - close)
                    if delta_sl > 0 and delta_tp / delta_sl > rapporto_massimo:
                        delta_tp = delta_sl * rapporto_massimo
                        tp = round(close - delta_tp, 4)
                        note.append(f"⚖️ TP ricalibrato per mantenere R/R ≤ {rapporto_massimo}")
                    note.append("✅ SELL confermato: trend forte" if macd_sell_ok else "⚠️ SELL anticipato: MACD ≈ signal")

    if segnale == "HOLD" and rileva_pattern_v(hist):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr, 4)
        note.append("📈 Pattern V rilevato: BUY da inversione rapida")

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
            note.append("⚠️ Trend concluso: attenzione a inversioni")

    if segnale == "BUY" and pattern and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"]):
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")

    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
