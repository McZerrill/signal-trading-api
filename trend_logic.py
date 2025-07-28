import pandas as pd
import logging
from indicators import calcola_ema, calcola_rsi, calcola_macd, calcola_atr

logging.basicConfig(level=logging.DEBUG)

MODALITA_TEST = True

def rileva_incrocio_progressivo(hist: pd.DataFrame, rialzista: bool = True) -> bool:
    if len(hist) < 5:
        return False

    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]

    e7, e25, e99 = hist['EMA_7'], hist['EMA_25'], hist['EMA_99']

    if rialzista:
        return (
            e7.iloc[-4] < e25.iloc[-4] < e99.iloc[-4] and
            e7.iloc[-3] > e25.iloc[-3] and e25.iloc[-3] < e99.iloc[-3] and
            e7.iloc[-1] > e25.iloc[-1] > e99.iloc[-1]
        )
    else:
        return (
            e7.iloc[-4] > e25.iloc[-4] > e99.iloc[-4] and
            e7.iloc[-3] < e25.iloc[-3] and e25.iloc[-3] > e99.iloc[-3] and
            e7.iloc[-1] < e25.iloc[-1] < e99.iloc[-1]
        )

def conta_candele_trend(hist: pd.DataFrame, rialzista: bool = True, max_candele: int = 20) -> int:
    count = 0
    for i in range(-1, -max_candele - 1, -1):
        e7 = hist['EMA_7'].iloc[i]
        e25 = hist['EMA_25'].iloc[i]
        e99 = hist['EMA_99'].iloc[i]
        if rialzista and e7 > e25 > e99:
            count += 1
        elif not rialzista and e7 < e25 < e99:
            count += 1
        else:
            break
    return count

def analizza_trend(hist: pd.DataFrame, spread: float = 0.0, hist_1m: pd.DataFrame = None):
    logging.debug("🔍 Inizio analisi trend semplificata BUY/SELL")

    note = []
    tp = sl = supporto = 0.0
    segnale = "HOLD"

    if len(hist) < 5:
        return segnale, hist, 0.0, "Dati insufficienti", tp, sl, supporto

    hist = hist.copy()
    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]

    # Aggiungiamo anche altri indicatori per uso futuro
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])
    hist['ATR'] = calcola_atr(hist)

    ultimo = hist.iloc[-1]
    penultimo = hist.iloc[-2]
    antepenultimo = hist.iloc[-3]

    ema7 = ultimo['EMA_7']
    ema25 = ultimo['EMA_25']
    ema99 = ultimo['EMA_99']
    close = ultimo['close']

    distanza_ema = abs(ema7 - ema25)
    curvatura_attuale = ema25 - penultimo['EMA_25']
    curvatura_precedente = penultimo['EMA_25'] - antepenultimo['EMA_25']
    accelerazione = curvatura_attuale - curvatura_precedente

    logging.info(f"[INFO] distanza_ema={distanza_ema:.6f}, accelerazione={accelerazione:.6f}")

    # ==== BLOCCO BUY ====
    if rileva_incrocio_progressivo(hist, rialzista=True):
        if accelerazione > 0 and distanza_ema / close > 0.001:
            durata_trend = conta_candele_trend(hist, rialzista=True)
            segnale = "BUY"
            note.append(f"🕒 Trend BUY attivo da {durata_trend} candele")
            note.append("✅ BUY confermato: trend forte")
            tp = round(close + distanza_ema * 0.8, 4)
            sl = round(ema99 - (ema7 - ema99), 4)
        else:
            note.append("⚠️ Incrocio rialzista rilevato, ma accelerazione insufficiente o distanza bassa")

    # ==== BLOCCO SELL ====
    elif rileva_incrocio_progressivo(hist, rialzista=False):
        if accelerazione < 0 and distanza_ema / close > 0.001:
            durata_trend = conta_candele_trend(hist, rialzista=False)
            segnale = "SELL"
            note.append(f"🕒 Trend SELL attivo da {durata_trend} candele")
            note.append("✅ SELL confermato: trend forte")
            tp = round(close - distanza_ema * 0.8, 4)
            sl = round(ema99 + (ema99 - ema7), 4)
        else:
            note.append("⚠️ Incrocio ribassista rilevato, ma accelerazione insufficiente o distanza bassa")

    else:
        note.append("🔎 Nessun incrocio progressivo rilevato")

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
