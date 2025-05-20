
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
        return "🪓 Hammer"
    if corpo < 0 and ombra_sup >= 2 * abs(corpo) and ombra_inf <= abs(corpo) * 0.3:
        return "🌠 Shooting Star"
    if corpo > 0 and c['close'] > df['open'].iloc[-2] and c['open'] < df['close'].iloc[-2]:
        return "🔄 Bullish Engulfing"
    if corpo < 0 and c['close'] < df['open'].iloc[-2] and c['open'] > df['close'].iloc[-2]:
        return "🔃 Bearish Engulfing"
    return ""

def analizza_trend(hist: pd.DataFrame):
    # --- Preparazione indicatori ---
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

    # --- Valori attuali e precedenti ---
    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]

    ema7, ema25, ema99 = ultimo['EMA_7'], ultimo['EMA_25'], ultimo['EMA_99']
    close, rsi, atr = ultimo['close'], ultimo['RSI'], ultimo['ATR']
    macd, macd_signal = ultimo['MACD'], ultimo['MACD_SIGNAL']
    supporto = calcola_supporto(hist)

    # --- Calcoli distanza e trend ---
    dist_attuale = abs(ema7 - ema25) + abs(ema25 - ema99)
    dist_precedente = abs(penultimo['EMA_7'] - penultimo['EMA_25']) + abs(penultimo['EMA_25'] - penultimo['EMA_99'])
    dist_diff = dist_attuale - dist_precedente
    dist_level = valuta_distanza(dist_attuale)

    note = []
    segnale = "HOLD"
    tp = sl = 0.0
    condizioni_verificate = 0

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    pattern = riconosci_pattern_candela(hist)

    # --- Classificazione forza MACD ---
    macd_gap = macd - macd_signal
    forza_macd = "neutro"
    if abs(macd_gap) < 0.0001 and -0.001 < macd < 0.001:
        forza_macd = "neutro"
    elif macd_gap > 0 and macd < 0.002:
        forza_macd = "buy_anticipato"
    elif macd_gap > 0 and macd >= 0.002:
        forza_macd = "buy_confermato"
    elif macd_gap < 0 and macd > -0.002:
        forza_macd = "sell_anticipato"
    elif macd_gap < 0 and macd <= -0.002:
        forza_macd = "sell_confermato"

    # --- Logica Breakout ---
    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    volume_attuale = hist['volume'].iloc[-1]

    breakout_confirmato = False
    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout rialzista confermato")
        breakout_confirmato = True
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout ribassista confermato")
        breakout_confirmato = True
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume non sufficiente a confermare")

    # --- BUY completo avanzato ---
    if (
        (penultimo['EMA_7'] < penultimo['EMA_25'] < penultimo['EMA_99']
         and trend_up and dist_diff > 0 and rsi > 56 and macd > macd_signal 
         and macd > 0.001 and 2 <= candele_trend_up <= 6) 
        or (trend_up and candele_trend_up in [3, 4] and rsi > 56 and macd > macd_signal and dist_diff > 0)
    ):
        if atr < 0.002 or abs(ema7 - ema25) < 0.0005 or abs(ema25 - ema99) < 0.0005:
            note.append("⚠️ BUY ignorato: volatilità o distanza EMA troppo bassa")
        else:
            segnale = "BUY"
            tp = round(close + atr * 1.5, 4)
            sl = round(close - atr * 1.2, 4)
            note.append("✅ BUY confermato con breakout e allargamento EMA" if breakout_confirmato else "✅ BUY confermato senza breakout ma con allargamento EMA")

    # --- SELL completo avanzato ---
    elif (
        (penultimo['EMA_7'] > penultimo['EMA_25'] > penultimo['EMA_99']
         and trend_down and dist_diff > 0 and rsi < 44 and macd < macd_signal 
         and macd < -0.001 and 2 <= candele_trend_down <= 6) 
        or (trend_down and candele_trend_down in [3, 4] and rsi < 44 and macd < macd_signal and dist_diff > 0)         
    ):
        if atr < 0.002 or abs(ema7 - ema25) < 0.0005 or abs(ema25 - ema99) < 0.0005:
            note.append("⚠️ SELL ignorato: volatilità o distanza EMA troppo bassa")
        else:
            segnale = "SELL"
            tp = round(close - atr * 1.5, 4)
            sl = round(close + atr * 1.2, 4)
            note.append("✅ SELL confermato con breakout e allargamento EMA" if breakout_confirmato else "✅ SELL confermato senza breakout ma con allargamento EMA")

    # --- Presegnali e incroci EMA ---
    else:
        if penultimo['EMA_7'] < penultimo['EMA_25'] and ema7 > ema25:
            if ema25 < ema99 and abs(ema7 - ema99) / ema99 < 0.015:
                if rsi > 50: condizioni_verificate += 1
                if macd > macd_signal: condizioni_verificate += 1
                note.append("🟢 Presegnale BUY: EMA7 incrocia EMA25 sotto EMA99")
        elif penultimo['EMA_7'] > penultimo['EMA_25'] and ema7 < ema25:
            if ema25 > ema99 and abs(ema7 - ema99) / ema99 < 0.015:
                if rsi < 50: condizioni_verificate += 1
                if macd < macd_signal: condizioni_verificate += 1
                note.append("🔴 Presegnale SELL: EMA7 incrocia EMA25 sopra EMA99")

    # --- Annotazioni finali ---
    if segnale in ["BUY", "SELL"]:
        n_candele = candele_trend_up if segnale == "BUY" else candele_trend_down
        note.insert(0, f"📊 Trend attivo da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick rilevato: {pattern}")
    else:
        if condizioni_verificate >= 2:
            note.append("🟡 Trend in formazione (presegnale attivo)")
        elif trend_up and candele_trend_up <= 2:
            note.append("🟡 Trend attivo ma debole")
        elif trend_down and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista ma debole")
        elif candele_trend_up <= 1 and not trend_up:
            note.append("⚠️ Trend terminato")

    # --- Contraddizioni con pattern ---
    if segnale == "BUY" and pattern and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"]):
        note.append("⚠️ Pattern ribassista rilevato: possibile inversione")
        segnale = "HOLD"
    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append("⚠️ Pattern Hammer rilevato: possibile inversione")
        segnale = "HOLD"

    return segnale, hist, dist_attuale, "\n".join(note).strip(), tp, sl, supporto
