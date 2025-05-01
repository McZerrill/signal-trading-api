
import pandas as pd
from indicators import calcola_ema, calcola_rsi, calcola_atr, calcola_macd
from support_resistance import calcola_supporto

def riconosci_pattern_candela(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return ""

    ultima = df.iloc[-1]
    penultima = df.iloc[-2]
    terzultima = df.iloc[-3]

    body = abs(ultima['close'] - ultima['open'])
    range_candela = ultima['high'] - ultima['low']
    upper_shadow = ultima['high'] - max(ultima['close'], ultima['open'])
    lower_shadow = min(ultima['close'], ultima['open']) - ultima['low']

    # Hammer (BUY)
    if body < range_candela * 0.3 and lower_shadow > body * 2 and upper_shadow < body:
        return "Hammer"

    # Shooting Star (SELL)
    if body < range_candela * 0.3 and upper_shadow > body * 2 and lower_shadow < body:
        return "Shooting Star"

    # Bearish Engulfing
    if penultima['close'] > penultima['open'] and ultima['open'] > ultima['close'] and ultima['open'] > penultima['close'] and ultima['close'] < penultima['open']:
        return "Bearish Engulfing"

    # Bullish Engulfing
    if penultima['close'] < penultima['open'] and ultima['close'] > ultima['open'] and ultima['open'] < penultima['close'] and ultima['close'] > penultima['open']:
        return "Bullish Engulfing"

    # Morning Star
    if (
        terzultima['close'] < terzultima['open'] and
        abs(penultima['close'] - penultima['open']) < (terzultima['open'] - terzultima['close']) * 0.3 and
        ultima['close'] > ultima['open'] and
        ultima['close'] > (terzultima['open'] + terzultima['close']) / 2
    ):
        return "Morning Star"

    # Evening Star
    if (
        terzultima['close'] > terzultima['open'] and
        abs(penultima['close'] - penultima['open']) < (terzultima['close'] - terzultima['open']) * 0.3 and
        ultima['close'] < ultima['open'] and
        ultima['close'] < (terzultima['open'] + terzultima['close']) / 2
    ):
        return "Evening Star"

    return ""

def conta_candele_trend(df: pd.DataFrame, rialzista: bool) -> int:
    conteggio = 0
    for i in range(1, min(10, len(df)) + 1):
        ema7 = df['EMA_7'].iloc[-i]
        ema25 = df['EMA_25'].iloc[-i]
        ema99 = df['EMA_99'].iloc[-i]
        if rialzista:
            if ema7 > ema25 > ema99:
                conteggio += 1
            else:
                break
        else:
            if ema7 < ema25 < ema99:
                conteggio += 1
            else:
                break
    return conteggio

def valuta_distanza(distanza: float) -> str:
    if distanza < 0.001:
        return "bassa"
    elif distanza < 0.01:
        return "media"
    else:
        return "alta"

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

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    pattern = riconosci_pattern_candela(hist)

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

    if segnale in ["BUY", "SELL"]:
        note.insert(0, f"📊 Trend attivo da {candele_trend_up if segnale == 'BUY' else candele_trend_down} candele | Distanza: {dist_level}")
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

    pattern_inverso_sell = pattern and "Hammer" in pattern
    pattern_inverso_buy = pattern and any(p in pattern for p in ["Shooting Star", "Evening Star", "Bearish Engulfing"])

    if segnale == "SELL" and pattern_inverso_sell:
        note.append("⚠️ Pattern Hammer (BUY) rilevato: possibile inversione, prudenza sul segnale SELL")
        segnale = "HOLD"
    if segnale == "BUY" and pattern_inverso_buy:
        note.append("⚠️ Pattern ribassista rilevato: possibile inversione, prudenza sul segnale BUY")
        segnale = "HOLD"

    return segnale, hist, dist_attuale, "\n".join(note).strip(), tp, sl, supporto
