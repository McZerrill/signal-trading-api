
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

import pandas as pd
from indicators import calcola_ema, calcola_rsi, calcola_atr, calcola_macd
from pattern_recognition import riconosci_pattern_candela
from supporti import calcola_supporto
from trend_utils import conta_candele_trend, valuta_distanza

import pandas as pd
from indicators import calcola_ema, calcola_rsi, calcola_atr, calcola_macd
from pattern_recognition import riconosci_pattern_candela
from supporti import calcola_supporto
from trend_utils import conta_candele_trend, valuta_distanza

def analizza_trend(hist: pd.DataFrame):
    hist = hist.copy()
    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    # Volume medio e rapporto
    hist['volume_ma20'] = hist['volume'].rolling(window=20).mean()
    volume_ultimo = hist['volume'].iloc[-1]
    volume_medio = hist['volume_ma20'].iloc[-1]
    volume_ratio = volume_ultimo / volume_medio if volume_medio else 0

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

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    pattern = riconosci_pattern_candela(hist)
    pattern_rialzisti = ["Hammer", "Morning Star", "Bullish Engulfing", "Piercing Line", "Inverted Hammer"]
    pattern_ribassisti = ["Shooting Star", "Evening Star", "Bearish Engulfing", "Dark Cloud Cover", "Hanging Man"]
    pattern_info = f"✅ Pattern candlestick rilevato: {pattern}" if pattern else ""

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

    if (
        penultimo['EMA_7'] < penultimo['EMA_25'] < penultimo['EMA_99']
        and trend_up and rsi > 50 and macd > macd_signal and candele_trend_up >= 3 and dist_diff > 0
    ):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr * 1.2, 4)
        note.append("✅ BUY confermato (trend completo)")

    elif trend_up and rsi > 60 and candele_trend_up >= 3 and forza_macd == "buy_anticipato":
        segnale = "BUY"
        tp = round(close + atr * 1.3, 4)
        sl = round(close - atr * 1.1, 4)
        note.append("⚡ BUY anticipato: trend forte, MACD in attivazione")

    elif (
        penultimo['EMA_7'] > penultimo['EMA_25'] > penultimo['EMA_99']
        and trend_down and rsi < 50 and macd < macd_signal and candele_trend_down >= 3 and dist_diff > 0
    ):
        segnale = "SELL"
        tp = round(close - atr * 1.5, 4)
        sl = round(close + atr * 1.2, 4)
        note.append("✅ SELL confermato (trend completo)")

    elif trend_down and rsi < 40 and candele_trend_down >= 3 and forza_macd == "sell_anticipato":
        segnale = "SELL"
        tp = round(close - atr * 1.3, 4)
        sl = round(close + atr * 1.1, 4)
        note.append("⚡ SELL anticipato: trend forte, MACD in attivazione")

    else:
        if penultimo['EMA_7'] < penultimo['EMA_25'] and ema7 > ema25 and ema25 < ema99 and abs(ema7 - ema99) / ema99 < 0.015:
            if rsi > 50: condizioni_verificate += 1
            if macd > macd_signal: condizioni_verificate += 1
            note.append("🟢 Presegnale BUY: EMA7 incrocia EMA25 sotto EMA99")
        elif penultimo['EMA_7'] > penultimo['EMA_25'] and ema7 < ema25 and ema25 > ema99 and abs(ema7 - ema99) / ema99 < 0.015:
            if rsi < 50: condizioni_verificate += 1
            if macd < macd_signal: condizioni_verificate += 1
            note.append("🔴 Presegnale SELL: EMA7 incrocia EMA25 sopra EMA99")

    if segnale in ["BUY", "SELL"]:
        note.insert(0, f"📊 Trend attivo da {candele_trend_up if segnale == 'BUY' else candele_trend_down} candele | Distanza: {dist_level}")
        if pattern_info:
            note.append(pattern_info)
        if volume_ratio > 1.2:
            note.append("📈 Volume superiore alla media: breakout confermato")
        elif volume_ratio < 0.8:
            note.append("⚠️ Volume debole: attenzione a possibili false rotture")
    else:
        if condizioni_verificate >= 2:
            note.append("🟡 Trend in formazione (presegnale attivo)")
        elif trend_up and candele_trend_up <= 2:
            note.append("🟡 Trend attivo ma debole")
        elif trend_down and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista ma debole")
        elif candele_trend_up <= 1 and not trend_up:
            note.append("⚠️ Trend terminato")

    if segnale == "SELL" and any(p in pattern for p in pattern_rialzisti):
        note.append(f"🚫 Pattern {pattern} (BUY) rilevato: possibile falsa rottura ribassista")
        segnale = "HOLD"
    elif segnale == "BUY" and any(p in pattern for p in pattern_ribassisti):
        note.append(f"🚫 Pattern {pattern} (SELL) rilevato: possibile falsa rottura rialzista")
        segnale = "HOLD"

    return segnale, hist, dist_attuale, "\n".join(note).strip(), tp, sl, supporto
