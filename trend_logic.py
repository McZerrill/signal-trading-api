import pandas as pd
from indicators import (
    calcola_rsi,
    calcola_macd,
    calcola_atr,
    calcola_supporto,
    calcola_ema,
)
import logging

# -----------------------------------------------------------------------------
# Costanti di configurazione
# -----------------------------------------------------------------------------
MODALITA_TEST = True
MODALITA_TEST_FORZATA = True
SOGLIA_PUNTEGGIO = 2
DISATTIVA_CHECK_EMA_1M = True

# Parametri separati per test / produzione
_PARAMS_TEST = {
    "volume_soglia": 150,
    "atr_minimo": 0.0004,
    "distanza_minima": 0.0007,
    "macd_rsi_range": (45, 55),
    "macd_signal_threshold": 0.0004,
}

_PARAMS_PROD = {
    "volume_soglia": 300,
    "atr_minimo": 0.0009,
    "distanza_minima": 0.0012,
    "macd_rsi_range": (47, 53),
    "macd_signal_threshold": 0.0006,
}


def _p(key):
    """Restituisce il parametro attivo (test o produzione)."""
    params = _PARAMS_TEST if MODALITA_TEST else _PARAMS_PROD
    return params[key]


# -----------------------------------------------------------------------------
# Helper comuni – evitano ripetizioni
# -----------------------------------------------------------------------------

def enrich_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge EMA, RSI, ATR, MACD (se mancanti) al DataFrame."""
    missing_cols = {"EMA_7", "EMA_25", "EMA_99"} - set(hist.columns)
    if missing_cols:
        ema = calcola_ema(hist, [7, 25, 99])
        hist["EMA_7"], hist["EMA_25"], hist["EMA_99"] = ema[7], ema[25], ema[99]

    if "RSI" not in hist.columns:
        hist["RSI"] = calcola_rsi(hist["close"])

    if {"MACD", "MACD_SIGNAL"}.issubset(hist.columns) is False:
        hist["MACD"], hist["MACD_SIGNAL"] = calcola_macd(hist["close"])

    if "ATR" not in hist.columns:
        hist["ATR"] = calcola_atr(hist)

    return hist


def is_trend_up(e7: float, e25: float, e99: float) -> bool:
    return e7 > e25 > e99


def is_trend_down(e7: float, e25: float, e99: float) -> bool:
    return e7 < e25 < e99


def trend_score_description(score: int) -> str:
    if score >= 4:
        return "🔥 Trend forte"
    if score >= 2:
        return "👍 Trend moderato"
    if score == 1:
        return "🟡 Trend debole positivo"
    if score == 0:
        return "🔍 Trend neutro"
    if score == -1:
        return "🟠 Trend debole negativo"
    if score <= -4:
        return "❌ Trend ribassista forte"
    if score <= -2:
        return "⚠️ Trend ribassista moderato"
    return ""


def pattern_contrario(segnale: str, pattern: str) -> bool:
    return (
        segnale == "BUY"
        and pattern
        and any(p in pattern for p in ["Shooting Star", "Bearish Engulfing"])
    ) or (
        segnale == "SELL" and pattern and "Hammer" in pattern
    )


# -----------------------------------------------------------------------------
# Funzioni originali, ripulite dalle ripetizioni
# -----------------------------------------------------------------------------

def valuta_distanza(distanza: float) -> str:
    if distanza < 1:
        return "bassa"
    elif distanza < 3:
        return "media"
    return "alta"


def conta_candele_trend(hist: pd.DataFrame, rialzista: bool = True, max_candele: int = 20) -> int:
    count = 0
    for i in range(-1, -max_candele - 1, -1):
        e7, e25, e99 = hist["EMA_7"].iloc[i], hist["EMA_25"].iloc[i], hist["EMA_99"].iloc[i]
        if rialzista and is_trend_up(e7, e25, e99):
            count += 1
        elif not rialzista and is_trend_down(e7, e25, e99):
            count += 1
        else:
            break
    return count


def ema_in_movimento_coerente(hist_1m: pd.DataFrame, rialzista: bool = True, n_candele: int = 15) -> bool:
    if hist_1m is None or len(hist_1m) < n_candele + 1:
        return False

    hist_1m = enrich_indicators(hist_1m.copy())

    e7 = hist_1m["EMA_7"].tail(n_candele)
    e25 = hist_1m["EMA_25"].tail(n_candele)
    e99 = hist_1m["EMA_99"].tail(n_candele)

    delta7 = e7.iloc[-1] - e7.iloc[0]
    delta25 = e25.iloc[-1] - e25.iloc[0]
    delta99 = e99.iloc[-1] - e99.iloc[0]

    return (delta7 > 0 and delta25 > 0 and delta99 > 0) if rialzista else (
        delta7 < 0 and delta25 < 0 and delta99 < 0
    )


# --- Le funzioni riconosci_pattern_candela, rileva_pattern_v, rileva_incrocio_progressivo
#     non contenevano grandi duplicazioni logiche; le riprendiamo quasi invariate.


def riconosci_pattern_candela(df: pd.DataFrame) -> str:
    c = df.iloc[-1]
    o, h, l, close = c["open"], c["high"], c["low"], c["close"]
    corpo = abs(close - o)
    ombra_sup = h - max(o, close)
    ombra_inf = min(o, close) - l

    if corpo == 0:
        return ""

    if corpo > 0 and ombra_inf >= 2 * corpo and ombra_sup <= corpo * 0.3:
        return "🪓 Hammer"
    if corpo < 0 and ombra_sup >= 2 * abs(corpo) and ombra_inf <= abs(corpo) * 0.3:
        return "🌠 Shooting Star"
    if corpo > 0 and c["close"] > df["open"].iloc[-2] and c["open"] < df["close"].iloc[-2]:
        return "🔄 Bullish Engulfing"
    if corpo < 0 and c["close"] < df["open"].iloc[-2] and c["open"] > df["close"].iloc[-2]:
        return "🔃 Bearish Engulfing"
    return ""


def rileva_pattern_v(hist: pd.DataFrame) -> bool:
    if len(hist) < 4 or not {"MACD", "RSI", "open", "close"}.issubset(hist.columns):
        return False

    sub = hist.iloc[-4:]
    try:
        rsi_start = sub["RSI"].iloc[0]
        rsi_end = sub["RSI"].iloc[-1]
        macd = sub["MACD"].iloc[-1]
    except Exception as e:
        logging.warning(f"⚠️ Errore nel rilevamento pattern V: {e}")
        return False

    for i in range(-3, 0):
        rossa = sub.iloc[i]["close"] < sub.iloc[i]["open"]
        verde = sub.iloc[i + 1]["close"] > sub.iloc[i + 1]["open"]
        corpo_rossa = abs(sub.iloc[i]["close"] - sub.iloc[i]["open"])
        corpo_verde = abs(sub.iloc[i + 1]["close"] - sub.iloc[i + 1]["open"])
        if rossa and verde and corpo_verde >= corpo_rossa:
            return rsi_start < 30 and rsi_end > 50 and abs(macd) < 0.01
    return False


def rileva_incrocio_progressivo(hist: pd.DataFrame) -> bool:
    if len(hist) < 5:
        return False

    e7, e25, e99 = hist["EMA_7"], hist["EMA_25"], hist["EMA_99"]
    return (
        is_trend_down(e7.iloc[-4], e25.iloc[-4], e99.iloc[-4])
        and e7.iloc[-3] > e25.iloc[-3] < e99.iloc[-3]
        and is_trend_up(e7.iloc[-1], e25.iloc[-1], e99.iloc[-1])
    )


# -----------------------------------------------------------------------------
# Punteggio trend (poche modifiche, ma usa helper is_trend_up/‑down)
# -----------------------------------------------------------------------------

def calcola_punteggio_trend(
    ema7,
    ema25,
    ema99,
    rsi,
    macd,
    macd_signal,
    volume_attuale,
    volume_medio,
    distanza_ema,
    atr,
    close,
):
    punteggio = 0

    if is_trend_up(ema7, ema25, ema99):
        punteggio += 2
    elif is_trend_down(ema7, ema25, ema99):
        punteggio -= 2

    if rsi > 60:
        punteggio += 1
    elif rsi < 40:
        punteggio -= 1

    if macd > macd_signal:
        punteggio += 1
    elif macd < macd_signal:
        punteggio -= 1

    if volume_attuale > volume_medio * 1.5:
        punteggio += 1
    else:
        punteggio -= 1

    if distanza_ema / close > 0.0015:
        punteggio += 1
    elif distanza_ema / close < 0.001:
        punteggio -= 1

    if atr / close > 0.0015:
        punteggio += 1
    elif atr / close < 0.001:
        punteggio -= 1

    return punteggio



def calcola_probabilita_successo(
    ema7,
    ema25,
    ema99,
    rsi,
    macd,
    macd_signal,
    candele_attive,
    breakout,
    volume_attuale,
    volume_medio,
    distanza_ema,
    atr,
    close,
    accelerazione,
    segnale
):
    punteggio = 0

    # 1. Direzione trend chiara (EMA progressive)
    if (ema7 > ema25 > ema99 and segnale == "BUY") or (ema7 < ema25 < ema99 and segnale == "SELL"):
        punteggio += 20

    # 2. RSI coerente
    if segnale == "BUY" and rsi >= 52:
        punteggio += 10
    elif segnale == "SELL" and rsi <= 48:
        punteggio += 10
    elif 48 < rsi < 52:
        punteggio -= 5  # RSI neutro penalizzato

    # 3. MACD coerente con segnale
    macd_gap = macd - macd_signal
    if segnale == "BUY":
        if macd > macd_signal and macd_gap > 0.001:
            punteggio += 10
        elif macd > 0 and macd_gap > 0:
            punteggio += 5  # MACD debole
        elif abs(macd_gap) < 0.001:
            punteggio -= 5  # MACD neutro
    elif segnale == "SELL":
        if macd < macd_signal and macd_gap < -0.001:
            punteggio += 10
        elif macd < 0 and macd_gap < 0:
            punteggio += 5
        elif abs(macd_gap) < 0.001:
            punteggio -= 5

    # 4. Volume coerente
    if volume_attuale > volume_medio * 1.5:
        punteggio += 10
    elif volume_attuale < volume_medio:
        punteggio -= 5  # volume scarso penalizzato

    # 5. Breakout forte
    if breakout:
        punteggio += 10

    # 6. Accelerazione coerente
    if segnale == "BUY" and accelerazione > 0:
        punteggio += 5
    elif segnale == "SELL" and accelerazione < 0:
        punteggio += 5
    else:
        punteggio -= 5  # accelerazione non coerente penalizzata

    # 7. Trend attivo da almeno 3-4 candele, ma non troppo lungo
    if 3 <= candele_attive <= 7:
        punteggio += 10
    elif candele_attive < 3:
        punteggio -= 5  # trend troppo giovane
    elif candele_attive > 8:
        punteggio -= 5  # trend troppo maturo

    # 8. Distanza EMA significativa
    distanza_pct = distanza_ema / close
    if distanza_pct > 0.0015:
        punteggio += 5
    elif distanza_pct < 0.0008:
        punteggio -= 5

    # 9. Volatilità accettabile (ATR)
    if atr / close > 0.0015:
        punteggio += 5
    elif atr / close < 0.0008:
        punteggio -= 5

    # 10. Penalità se ATR assoluto è troppo basso/alto
    if atr < 0.0002 or atr > 0.005:
        punteggio -= 10

    # Normalizzazione finale
    probabilita = min(max(round(punteggio * 1.25), 5), 95)
    return probabilita



# -----------------------------------------------------------------------------
# Funzione principale: analizza_trend (semplificata internamente)
# -----------------------------------------------------------------------------

def analizza_trend(hist: pd.DataFrame, spread: float = 0.0, hist_1m: pd.DataFrame = None):
    logging.debug("🔍 Inizio analisi trend")

    if len(hist) < 22:
        logging.warning("⚠️ Dati insufficienti per l'analisi")
        return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, 0.0

    hist = enrich_indicators(hist.copy())

    try:
        ultimo, penultimo, antepenultimo = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3]
        ema7, ema25, ema99 = ultimo[["EMA_7", "EMA_25", "EMA_99"]]
        close, rsi, atr, macd, macd_signal = ultimo[
            ["close", "RSI", "ATR", "MACD", "MACD_SIGNAL"]
        ]
        supporto = calcola_supporto(hist)
    except Exception as e:
        logging.error(f"❌ Errore nell'accesso ai dati finali: {e}")
        return "HOLD", hist, 0.0, "Errore su iloc finali", 0.0, 0.0, 0.0

    note = []

    volume_attuale = hist["volume"].iloc[-1]
    volume_medio = hist["volume"].iloc[-21:-1].mean()
    distanza_ema = abs(ema7 - ema25)
    curvatura_ema25 = ema25 - penultimo["EMA_25"]
    curvatura_precedente = penultimo["EMA_25"] - antepenultimo["EMA_25"]
    accelerazione = curvatura_ema25 - curvatura_precedente
    trend_up, trend_down = is_trend_up(ema7, ema25, ema99), is_trend_down(ema7, ema25, ema99)
    recupero_buy = ema7 > ema25 < ema99 and close > ema25 and ema25 > penultimo["EMA_25"]
    recupero_sell = ema7 < ema25 > ema99 and close < ema25 and ema25 < penultimo["EMA_25"]
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)
    macd_gap = macd - macd_signal

    # ------------------------------------------------------------------
    # Filtri preliminari (ATR, Volume, distanza)
    # ------------------------------------------------------------------
    if atr / close < _p("atr_minimo"):
        note.append("⚠️ ATR troppo basso: mercato poco volatile")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    if volume_attuale < volume_medio * 2.5 and not MODALITA_TEST:
        note.append("⚠️ Volume basso: segnale debole")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    # ------------------------------------------------------------------
    # Punteggio complessivo + descrizione
    # ------------------------------------------------------------------
    punteggio_trend = calcola_punteggio_trend(
        ema7,
        ema25,
        ema99,
        rsi,
        macd,
        macd_signal,
        volume_attuale,
        volume_medio,
        distanza_ema,
        atr,
        close,
    )
    note.append(f"📊 Punteggio trend complessivo: {punteggio_trend}")
    note.append(trend_score_description(punteggio_trend))

    # ------------------------------------------------------------------
    # Breakout
    # ------------------------------------------------------------------
    breakout_valido = False
    massimo_20 = hist["high"].iloc[-21:-1].max()
    minimo_20 = hist["low"].iloc[-21:-1].min()
    corpo_candela = abs(ultimo["close"] - ultimo["open"])

    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout rialzista con volume alto")
        if corpo_candela > atr:
            note.append("🚀 Spike rialzista con breakout solido")
            breakout_valido = True
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout ribassista con volume alto")
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume insufficiente")

    # ------------------------------------------------------------------
    # Condizioni MACD / RSI
    # ------------------------------------------------------------------
    macd_buy_ok = macd > macd_signal and macd_gap > 0.001
    macd_buy_debole = macd > 0 and macd_gap > 0
    macd_sell_ok = macd < macd_signal and macd_gap < -_p("macd_signal_threshold")
    macd_sell_debole = macd < 0 and macd_gap < 0.005

    segnale, tp, sl = "HOLD", 0.0, 0.0

    # ------------------------------------------------------------------
    # BUY logic
    # ------------------------------------------------------------------
    if (trend_up or recupero_buy or breakout_valido) and (distanza_ema / close) > _p("distanza_minima"):
        durata_trend = candele_trend_up
        if rsi >= 52 and macd_buy_ok and punteggio_trend >= SOGLIA_PUNTEGGIO:
            if durata_trend >= 8:
                note.append(f"⛔ Trend BUY troppo maturo ({durata_trend} candele)")
            elif accelerazione < 0:
                note.append(f"⚠️ BUY evitato: accelerazione negativa ({accelerazione:.6f})")
            else:
                segnale = "BUY"
                #note.append(f"🕒 Trend BUY attivo da {durata_trend} candele")
                note.append("✅ BUY confermato")
        elif rsi >= 50 and macd_buy_debole:
            note.append("⚠️ BUY debole: RSI > 50 e MACD > signal, ma segnale incerto")

    # ------------------------------------------------------------------
    # SELL logic
    # ------------------------------------------------------------------
    if (trend_down or recupero_sell) and (distanza_ema / close) > _p("distanza_minima"):
        durata_trend = candele_trend_down
        if rsi <= 48 and macd_sell_ok and punteggio_trend <= -SOGLIA_PUNTEGGIO:
            if durata_trend >= 8:
                note.append(f"⛔ Trend SELL troppo maturo ({durata_trend} candele)")
            elif accelerazione > 0:
                note.append(f"⚠️ SELL evitato: accelerazione in risalita ({accelerazione:.6f})")
            else:
                segnale = "SELL"
                #note.append(f"🕒 Trend SELL attivo da {durata_trend} candele")
                note.append("✅ SELL confermato")
        elif rsi <= 55 and macd_sell_debole:
            note.append("⚠️ SELL debole: RSI < 55 e MACD < signal, ma segnale incerto")

    if segnale == "HOLD" and not any([trend_up, trend_down]):
        note.append("🔎 Nessun segnale valido rilevato: condizioni insufficienti")

    # ------------------------------------------------------------------
    # Pattern V rapido
    # ------------------------------------------------------------------
    if segnale == "HOLD" and rileva_pattern_v(hist):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr, 4)
        note.append("📈 Pattern V rilevato: BUY da inversione rapida")

    # ------------------------------------------------------------------
    # Se segnale BUY/SELL aggiungi meta info
    # ------------------------------------------------------------------
    pattern = riconosci_pattern_candela(hist)
    if segnale in ["BUY", "SELL"]:
        n_candele = candele_trend_up if segnale == "BUY" else candele_trend_down
        dist_level = valuta_distanza(distanza_ema)
        note.insert(0, f"📊 Trend attivo da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick rilevato: {pattern}")
    else:
        if trend_up and candele_trend_up <= 2:
            note.append("🟡 Trend attivo")
        elif trend_down and candele_trend_down <= 2:
            note.append("🟡 Trend ribassista")
        elif candele_trend_up <= 1 and not trend_up:
            note.append("⚠️ Trend concluso: attenzione a inversioni")

    # ------------------------------------------------------------------
    # Invalidation per pattern contrario o neutralità MACD/RSI
    # ------------------------------------------------------------------
    if pattern_contrario(segnale, pattern):
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    if segnale in ["BUY", "SELL"] and 48 < rsi < 52 and abs(macd_gap) < 0.001:
        note.append("⚠️ RSI e MACD neutri: segnale evitato")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # --------------------------------------------------------------
    # Controllo facoltativo: EMA su 1 m coerenti col trend 15 m
    # --------------------------------------------------------------

    if not DISATTIVA_CHECK_EMA_1M:
        n_check_ema = 5 if MODALITA_TEST else 15
        if not ema_in_movimento_coerente(hist_1m, rialzista=(segnale == "BUY"), n_candele=n_check_ema):
            note.append("⛔ Segnale annullato: EMA su 1m non in movimento coerente col trend 15m")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto


    # ------------------------------------------------------------------
    # BUY forzato su incrocio progressivo
    # ------------------------------------------------------------------
    if segnale == "HOLD" and rileva_incrocio_progressivo(hist):
        segnale = "BUY"
        note.append("📈 Incrocio progressivo EMA(7>25>99) rilevato: BUY confermato")

    # ------------------------------------------------------------------
    # Calcolo TP/SL finale realistico ma coerente con l’ATR
    # ------------------------------------------------------------------

    # Massimo TP in percentuale del prezzo (es: 5%)
    PERCENTUALE_TP_MAX = 0.05  # 5%
    massimo_tp = close * PERCENTUALE_TP_MAX

    # Calcolo moltiplicatore massimo per ATR in base alla distanza massima
    fattore_tp = min(6.0, massimo_tp / atr)
    fattore_sl = fattore_tp / 2  # SL più stretto

    # Calcolo TP/SL in base al segnale
    if segnale == "BUY":
        tp = round(close + atr * fattore_tp, 4)
        sl = round(close - atr * fattore_sl, 4)
        if sl >= close or tp <= close:
            logging.warning(f"⚠️ TP/SL incoerenti (BUY): ingresso={close}, TP={tp}, SL={sl}")
            note.append("⚠️ TP/SL BUY potenzialmente incoerenti")

    elif segnale == "SELL":
        tp = round(close - atr * fattore_tp, 4)
        sl = round(close + atr * fattore_sl, 4)
        if sl <= close or tp >= close:
            logging.warning(f"⚠️ TP/SL incoerenti (SELL): ingresso={close}, TP={tp}, SL={sl}")
            note.append("⚠️ TP/SL SELL potenzialmente incoerenti")

    # ------------------------------------------------------------------
    # Calcolo tempo stimato per raggiungere TP (con forchetta)
    # ------------------------------------------------------------------
    try:
        if segnale in ["BUY", "SELL"] and atr > 0 and tp > 0:
            distanza = abs(tp - close)

            # Coefficiente di efficienza: quanto ATR "va nella direzione giusta"
            EFFICIENZA_MIN = 0.3  # più lento, più rimbalzi
            EFFICIENZA_MAX = 0.6  # più diretto

            ore_min = round((distanza / (atr * EFFICIENZA_MAX)) * 0.25, 1)
            ore_max = round((distanza / (atr * EFFICIENZA_MIN)) * 0.25, 1)

            if ore_min == ore_max:
                note.append(f"🎯 Target stimato in ~{ore_min}h")
            else:
                note.append(f"🎯 Target stimato tra ~{ore_min}h e {ore_max}h")
    except Exception as e:
        logging.warning(f"⚠️ Errore calcolo tempo stimato: {e}")


    # Calcolo probabilità di successo
    try:
        if segnale in ["BUY", "SELL"]:
            n_candele = candele_trend_up if segnale == "BUY" else candele_trend_down
            probabilita = calcola_probabilita_successo(
                ema7=ema7,
                ema25=ema25,
                ema99=ema99,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                candele_attive=n_candele,
                breakout=breakout_valido,
                volume_attuale=volume_attuale,
                volume_medio=volume_medio,
                distanza_ema=distanza_ema,
                atr=atr,
                close=close,
                accelerazione=accelerazione,
                segnale=segnale
            )
            note.append(f"📊 Prob. successo stimata: {probabilita}%")
    except Exception as e:
        logging.warning(f"⚠️ Errore calcolo probabilità successo: {e}")

    logging.debug("✅ Analisi completata")

    if segnale not in ["BUY", "SELL"]:
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    note = list(dict.fromkeys(note))

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
