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

def analizza_trend(hist: pd.DataFrame, timeframe: str = "1m"):
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

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)

    breakout_confirmato = False
    massimo_20 = hist['high'].iloc[-21:-1].max()
    minimo_20 = hist['low'].iloc[-21:-1].min()
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    volume_attuale = hist['volume'].iloc[-1]

    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        breakout_confirmato = True
        note.append("💥 Breakout rialzista confermato")
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        breakout_confirmato = True
        note.append("💥 Breakout ribassista confermato")

    condizioni_buy = (
        trend_up and
        candele_trend_up >= 2 and
        macd > macd_signal and
        rsi > 50 and
        breakout_confirmato
    )

    condizioni_sell = (
        trend_down and
        candele_trend_down >= 2 and
        macd < macd_signal and
        rsi < 50 and
        breakout_confirmato
    )

    if condizioni_buy:
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr * 1.2, 4)
        note.append("✅ BUY confermato con breakout e trend attivo")

    if condizioni_sell:
        segnale = "SELL"
        tp = round(close - atr * 1.5, 4)
        sl = round(close + atr * 1.2, 4)
        note.append("✅ SELL confermato con breakout e trend attivo")

    if segnale == "HOLD":
        if trend_up and candele_trend_up <= 2:
            note.append("🟡 Trend rialzista debole")
        elif trend_down and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista debole")
        else:
            note.append("🛑 Nessuna condizione favorevole")

    return segnale, hist, dist_attuale, "\n".join(note).strip(), tp, sl, supporto
