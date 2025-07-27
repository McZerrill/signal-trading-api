import pandas as pd
from indicators import calcola_rsi, calcola_macd, calcola_atr, calcola_supporto, calcola_ema
from indicators import calcola_percentuale_guadagno
import logging

logging.basicConfig(level=logging.DEBUG)


MODALITA_TEST = True
MODALITA_TEST_FORZATA = True

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

def ema_in_movimento_coerente(hist_1m: pd.DataFrame, rialzista: bool = True, n_candele: int = 15) -> bool:
    if hist_1m is None or len(hist_1m) < n_candele + 1:
        return False

    ema = calcola_ema(hist_1m, [7, 25, 99])
    hist_1m['EMA_7'] = ema[7]
    hist_1m['EMA_25'] = ema[25]
    hist_1m['EMA_99'] = ema[99]

    e7 = hist_1m['EMA_7'].tail(n_candele)
    e25 = hist_1m['EMA_25'].tail(n_candele)
    e99 = hist_1m['EMA_99'].tail(n_candele)

    delta7 = e7.iloc[-1] - e7.iloc[0]
    delta25 = e25.iloc[-1] - e25.iloc[0]
    delta99 = e99.iloc[-1] - e99.iloc[0]

    if rialzista:
        return delta7 > 0 and delta25 > 0 and delta99 > 0
    else:
        return delta7 < 0 and delta25 < 0 and delta99 < 0



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
    if not {'MACD', 'RSI', 'open', 'close'}.issubset(hist.columns):
        return False

    sub = hist.iloc[-4:]

    try:
        rsi_start = sub['RSI'].iloc[0]
        rsi_end = sub['RSI'].iloc[-1]
        macd = sub['MACD'].iloc[-1]
    except Exception as e:
        logging.warning(f"⚠️ Errore nel rilevamento pattern V: {e}")
        return False

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

def rileva_incrocio_progressivo(hist: pd.DataFrame) -> bool:
    if len(hist) < 5:
        return False

    e7 = hist['EMA_7']
    e25 = hist['EMA_25']
    e99 = hist['EMA_99']

    if not (e7.iloc[-4] < e25.iloc[-4] < e99.iloc[-4]):
        return False
    if not (e7.iloc[-3] > e25.iloc[-3] and e25.iloc[-3] < e99.iloc[-3]):
        return False
    if not (e7.iloc[-1] > e25.iloc[-1] > e99.iloc[-1]):
        return False

    return True



def analizza_trend(hist: pd.DataFrame, spread: float = 0.0, hist_1m: pd.DataFrame = None):
    
    logging.debug("🔍 Inizio analisi trend")
    
    hist = hist.copy()

    if len(hist) < 22:
        logging.warning("⚠️ Dati insufficienti per l'analisi")
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    ema = calcola_ema(hist, [7, 25, 99])
    hist['EMA_7'] = ema[7]
    hist['EMA_25'] = ema[25]
    hist['EMA_99'] = ema[99]
    hist['RSI'] = calcola_rsi(hist['close'])
    hist['ATR'] = calcola_atr(hist)
    hist['MACD'], hist['MACD_SIGNAL'] = calcola_macd(hist['close'])

    

    try:
        ultimo = hist.iloc[-1]
        penultimo = hist.iloc[-2]
        antepenultimo = hist.iloc[-3]
        ema7 = ultimo['EMA_7']
        ema25 = ultimo['EMA_25']
        ema99 = ultimo['EMA_99']
        close = ultimo['close']
        rsi = ultimo['RSI']
        atr = ultimo['ATR']
        macd = ultimo['MACD']
        macd_signal = ultimo['MACD_SIGNAL']
        supporto = calcola_supporto(hist)

 
        logging.debug(f"[DATI] Close={close:.6f}, RSI={rsi:.2f}, MACD={macd:.4f}, Signal={macd_signal:.4f}, ATR={atr:.6f}")
    except Exception as e:
        logging.error(f"❌ Errore nell'accesso ai dati finali: {e}")
        return "HOLD", hist, 0.0, "Errore su iloc finali", 0.0, 0.0, 0.0




    note = []
    investimento = 100.0
    guadagno_netto_target = 0.5
    commissione = 0.1

    volume_soglia = 100 if MODALITA_TEST else 300
    atr_minimo = 0.0012 if MODALITA_TEST else 0.0009
    distanza_minima = 0.0010 if MODALITA_TEST else 0.0012
    macd_rsi_range = (44, 56)
    macd_signal_threshold = 0.0005 if MODALITA_TEST else 0.0006

    if atr / close < atr_minimo:
        logging.info(f"🚫 ATR troppo basso: {atr / close:.6f} < {atr_minimo}")
        note.append("⚠️ ATR troppo basso: mercato poco volatile")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    volume_attuale = hist['volume'].iloc[-1]
    volume_medio = hist['volume'].iloc[-21:-1].mean()
    if volume_attuale < volume_medio * 2.5:
        logging.info(f"🚫 Volume debole: attuale={volume_attuale:.2f}, medio={volume_medio:.2f}")
        note.append("⚠️ Volume basso: segnale debole")
        if not MODALITA_TEST:
            return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    distanza_ema = abs(ema7 - ema25)
    curvatura_ema25 = ema25 - penultimo['EMA_25']
    curvatura_precedente = penultimo['EMA_25'] - antepenultimo['EMA_25']
    accelerazione = curvatura_ema25 - curvatura_precedente

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

    macd_buy_ok = macd > macd_signal and macd_gap > 0.001
    macd_buy_debole = macd > 0 and macd_gap > 0
    macd_sell_ok = macd < macd_signal and macd_gap < -macd_signal_threshold
    macd_sell_debole = macd < 0 and macd_gap < 0.005

    segnale = "HOLD"
    tp = sl = 0.0

    # BUY
    if (trend_up or recupero_buy or breakout_valido) and distanza_ema / close > distanza_minima:
        durata_trend = candele_trend_up
        if rsi >= 52 and macd_buy_ok:
            if durata_trend >= 6:
                note.append(f"⛔ Trend BUY troppo maturo ({durata_trend} candele)")
            elif accelerazione < 0:
                note.append(f"⚠️ BUY evitato: accelerazione negativa ({accelerazione:.6f})")
            else:
                segnale = "BUY"
                note.append(f"🕒 Trend BUY attivo da {durata_trend} candele")
                note.append("✅ BUY confermato: trend forte")
        elif rsi >= 50 and macd_buy_debole:
            note.append("⚠️ BUY debole: RSI > 50 e MACD > signal, ma segnale incerto")

    # SELL
    if (trend_down or recupero_sell) and distanza_ema / close > distanza_minima:
        durata_trend = candele_trend_down
        if rsi <= 48 and macd_sell_ok:
            if durata_trend >= 5:
                note.append(f"⛔ Trend SELL troppo maturo ({durata_trend} candele)")
            elif accelerazione > 0:
                note.append(f"⚠️ SELL evitato: accelerazione in risalita ({accelerazione:.6f})")
            else:
                segnale = "SELL"
                note.append(f"🕒 Trend SELL attivo da {durata_trend} candele")
                note.append("✅ SELL confermato: trend forte")
        elif rsi <= 55 and macd_sell_debole:
            note.append("⚠️ SELL debole: RSI < 55 e MACD < signal, ma segnale incerto")

        if segnale == "HOLD":
            note.append("🔎 Nessun segnale valido rilevato: condizioni insufficienti")
              

    logging.debug(f"[SEGNALE] Tipo: {segnale}, RSI={rsi:.2f}, MACD Gap={macd_gap:.6f}, Distanza EMA={distanza_ema:.6f}")


    if segnale == "HOLD" and rileva_pattern_v(hist):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr, 4)
        note.append("📈 Pattern V rilevato: BUY da inversione rapida")

    if segnale in ["BUY", "SELL"]:
        n_candele = candele_trend_up if segnale == "BUY" else candele_trend_down
        dist_level = valuta_distanza(distanza_ema)
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
        segnale = "HOLD"
    if segnale == "SELL" and pattern and "Hammer" in pattern:
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
        segnale = "HOLD"
    if segnale in ["BUY", "SELL"] and 48 < rsi < 52 and abs(macd - macd_signal) < 0.001:
        note.append("⚠️ RSI e MACD neutri: segnale evitato")
        segnale = "HOLD"

    if segnale not in ["BUY", "SELL"]:
        segnale = "HOLD"
        return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # ✅ Nuova condizione: movimento coerente delle EMA su 1m
    n_check_ema = 5 if MODALITA_TEST else 15
    coerente_1m = ema_in_movimento_coerente(hist_1m, rialzista=(segnale == "BUY"), n_candele=n_check_ema)

    if not coerente_1m:
        note.append("⛔ Segnale annullato: EMA su 1m non in movimento coerente col trend 15m")
        segnale = "HOLD"
        return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto


    # 🟢 BUY forzato se incrocio progressivo EMA rilevato (anche senza altri segnali)
    if segnale == "HOLD" and rileva_incrocio_progressivo(hist):
        segnale = "BUY"
        note.append("📈 Incrocio progressivo EMA(7>25>99) rilevato: BUY confermato")


    
    # 🔽 Calcolo TP/SL solo qui in fondo se il segnale è confermato
    forza_trend = min(max(distanza_ema / close, 0.001), 0.01)
    coeff_tp = min(max(1.5 + (accelerazione * 10), 1.2), 2.0)
    coeff_sl = min(max(1.0 + (accelerazione * 5), 1.0), 1.3)

    durata_trend = candele_trend_up if segnale == "BUY" else candele_trend_down
    if durata_trend >= 6:
        coeff_tp *= 0.9
        coeff_sl *= 1.1

    # Calcolo robusto del delta minimo assoluto
    delta_pct = calcola_percentuale_guadagno(
        guadagno_netto_target,
        investimento,
        spread,
        commissione
    )
    min_delta_abs = 0.002  # almeno 0.2% di variazione
    delta_price = max(close * delta_pct, min_delta_abs)

    # Validazione coefficienti
    if coeff_tp <= 0 or coeff_sl <= 0:
        logging.warning(f"❌ Coefficienti negativi: coeff_tp={coeff_tp:.2f}, coeff_sl={coeff_sl:.2f}")
        coeff_tp = 1.5
        coeff_sl = 1.0
        note.append("⚠️ Coefficienti TP/SL corretti automaticamente")

    # Calcolo coerente di TP/SL
    if segnale == "BUY":
        tp = round(close + delta_price * coeff_tp, 4)
        sl = round(close - delta_price * coeff_sl, 4)
    elif segnale == "SELL":
        tp = round(close - delta_price * coeff_tp, 4)
        sl = round(close + delta_price * coeff_sl, 4)

    # Verifica coerenza logica, ma non blocca il segnale
    if segnale == "BUY" and (sl >= close or tp <= close):
        logging.warning(f"⚠️ TP/SL incoerenti (BUY): ingresso={close}, TP={tp}, SL={sl}")
        note.append("⚠️ TP/SL BUY potenzialmente incoerenti")

    if segnale == "SELL" and (sl <= close or tp >= close):
        logging.warning(f"⚠️ TP/SL incoerenti (SELL): ingresso={close}, TP={tp}, SL={sl}")
        note.append("⚠️ TP/SL SELL potenzialmente incoerenti")

    logging.debug(f"[TP/SL] TP={tp:.4f}, SL={sl:.4f}, coeff_tp={coeff_tp:.2f}, coeff_sl={coeff_sl:.2f}, Δ% richiesta={delta_pct:.4%}")

    logging.debug("✅ Analisi completata\n")
    print(f"[DEBUG ANALYZE] Segnale={segnale}, Note:\n{note}")
    
    

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
