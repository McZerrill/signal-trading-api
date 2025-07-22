import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema
from indicators import calcola_percentuale_guadagno
import logging

logging.basicConfig(level=logging.DEBUG)


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
    logging.debug("🔍 Inizio analisi trend")

    if MODALITA_TEST:
        close = hist['close'].iloc[-1]
        tp = round(close * 1.01, 4)
        sl = round(close * 0.99, 4)
        supporto = hist['low'].min()

        note = ["🧪 Segnale BUY forzato per test (MODALITA_TEST=True)"]
        logging.debug(f"[DEBUG TEST] Forzatura attiva - close={close}, TP={tp}, SL={sl}")

        return "BUY", hist, 0.02, "\n".join(note), tp, sl, supporto

    # ↓↓↓ QUI INIZIA LA LOGICA REALE STANDARD ↓↓↓

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
    investimento = 100.0
    guadagno_netto_target = 0.5
    commissione = 0.1

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    if volume_attuale < volume_medio * 2.5 and not MODALITA_TEST:
        return "HOLD", hist, 0.0, "Volume insufficiente", 0.0, 0.0, supporto

    distanza_ema = abs(ema7 - ema25)
    curvatura_ema25 = ema25 - penultimo['EMA_25']
    curvatura_precedente = penultimo['EMA_25'] - antepenultimo['EMA_25']
    accelerazione = curvatura_ema25 - curvatura_precedente

    trend_up = ema7 > ema25 > ema99
    trend_down = ema7 < ema25 < ema99
    recupero_buy = ema7 > ema25 and close > ema25
    recupero_sell = ema7 < ema25 and close < ema25

    candele_trend_up = conta_candele_trend(hist, True)
    candele_trend_down = conta_candele_trend(hist, False)

    macd_gap = macd - macd_signal
    segnale = "HOLD"
    tp = sl = 0.0

    macd_buy_ok = macd > macd_signal and macd_gap > 0.001
    macd_sell_ok = macd < macd_signal and macd_gap < -0.001

    if (trend_up or recupero_buy) and rsi > 50 and macd_buy_ok:
        segnale = "BUY"
    elif (trend_down or recupero_sell) and rsi < 50 and macd_sell_ok:
        segnale = "SELL"

    if segnale in ["BUY", "SELL"]:
        delta_pct = calcola_percentuale_guadagno(
            guadagno_netto_target, investimento, spread, commissione
        )
        delta_price = close * delta_pct
        coeff_tp = 1.5
        coeff_sl = 1.0

        if segnale == "BUY":
            tp = round(close + delta_price * coeff_tp, 4)
            sl = round(close - delta_price * coeff_sl, 4)
        else:
            tp = round(close - delta_price * coeff_tp, 4)
            sl = round(close + delta_price * coeff_sl, 4)

    return segnale, hist, distanza_ema, "\n".join(note), tp, sl, supporto

