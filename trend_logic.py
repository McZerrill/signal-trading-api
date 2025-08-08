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
SOGLIA_PUNTEGGIO = 2
DISATTIVA_CHECK_EMA_1M = True

# Parametri separati per test / produzione
_PARAMS_TEST = {
    "volume_soglia": 150,
    "volume_alto": 2.0,
    "volume_medio": 1.2,
    "volume_basso": 0.8,
    "volume_molto_basso": 0.5,
    "atr_minimo": 0.0004,
    "atr_buono": 0.0015,
    "atr_basso": 0.0008,
    "atr_troppo_basso": 0.0002,
    "atr_troppo_alto": 0.005,
    "distanza_minima": 0.0005,
    "distanza_bassa": 0.0005,
    "distanza_media": 0.001,
    "distanza_alta": 0.002,
    "macd_rsi_range": (43, 57),
    "macd_signal_threshold": 0.0002,
    "macd_gap_forte": 0.0015,
    "macd_gap_debole": 0.001,
    "rsi_buy_forte": 53,
    "rsi_buy_debole": 52,
    "rsi_sell_forte": 49,
    "rsi_sell_debole": 45,
    "accelerazione_minima": 0.00005,

}

_PARAMS_PROD = {
    "volume_soglia": 300,
    "volume_alto": 2.0,
    "volume_medio": 1.2,
    "volume_basso": 0.8,
    "volume_molto_basso": 0.5,
    "atr_minimo": 0.0009,
    "atr_buono": 0.0015,
    "atr_basso": 0.0008,
    "atr_troppo_basso": 0.0002,
    "atr_troppo_alto": 0.005,
    "distanza_minima": 0.0012,
    "distanza_bassa": 0.0005,
    "distanza_media": 0.001,
    "distanza_alta": 0.002,
    "macd_rsi_range": (47, 53),
    "macd_signal_threshold": 0.0006,
    "macd_gap_forte": 0.002,
    "macd_gap_debole": 0.001,
    "rsi_buy_forte": 54,
    "rsi_buy_debole": 50,
    "rsi_sell_forte": 46,
    "rsi_sell_debole": 50,
    "accelerazione_minima": 0.00001,

}


def _p(key):
    """Restituisce il parametro attivo (test o produzione)."""
    params = _PARAMS_TEST if MODALITA_TEST else _PARAMS_PROD
    return params[key]

# --- Bridge punteggio_trend -> probabilità fusa (0..1) ---
TREND_MIN, TREND_MAX = -6, 8   # adatta ai tuoi range osservati
W_TREND, W_CONTEXT = 0.6, 0.4  # quanto pesa il trend vs la stima contestuale

def _clamp(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)

def _norm_trend(score: float) -> float:
    return _clamp((score - TREND_MIN) / (TREND_MAX - TREND_MIN))

def _fuse_prob(punteggio_trend: float, probabilita_percent: float) -> float:
    p_trend = _norm_trend(punteggio_trend)
    p_ctx   = _clamp(probabilita_percent / 100.0, 0, 1)
    return _clamp(W_TREND * p_trend + W_CONTEXT * p_ctx)


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

def valuta_distanza(distanza: float, close: float) -> str:
    
    distanza_pct = distanza / close
    if distanza_pct < _p("distanza_bassa"):
        return "bassa"
    elif distanza_pct < _p("distanza_media"):
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
    body = close - o
    body_abs = abs(body)
    upper = h - max(o, close)
    lower = min(o, close) - l

    if body_abs == 0:
        return ""

    if body > 0 and lower >= 2 * body_abs and upper <= body_abs * 0.3:
        return "🪓 Hammer"
    if body < 0 and upper >= 2 * body_abs and lower <= body_abs * 0.3:
        return "🌠 Shooting Star"
    if body > 0 and close > df["open"].iloc[-2] and o < df["close"].iloc[-2]:
        return "🔄 Bullish Engulfing"
    if body < 0 and close < df["open"].iloc[-2] and o > df["close"].iloc[-2]:
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

    for i in range(3): 
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

    # 1. Direzione del trend (più forte)
    if is_trend_up(ema7, ema25, ema99):
        punteggio += 3
    elif is_trend_down(ema7, ema25, ema99):
        punteggio -= 3

    # 2. RSI
    if rsi >= 65:
        punteggio += 2
    elif rsi >= 55:
        punteggio += 1
    elif rsi <= 35:
        punteggio -= 2
    elif rsi <= 45:
        punteggio -= 1

    # 3. MACD
    macd_gap = macd - macd_signal
    if macd_gap > 0.002:
        punteggio += 2
    elif macd_gap > 0:
        punteggio += 1
    elif macd_gap < -0.002:
        punteggio -= 2
    elif macd_gap < 0:
        punteggio -= 1

    # 4. Volume
    if volume_attuale > volume_medio * _p("volume_alto"):
        punteggio += 2
    elif volume_attuale > volume_medio * _p("volume_medio"):
        punteggio += 1
    elif volume_attuale < volume_medio * _p("volume_molto_basso"):
        punteggio -= 2
    elif volume_attuale < volume_medio * _p("volume_basso"):
        punteggio -= 1


    # 5. Distanza EMA
    distanza_pct = distanza_ema / close
    if distanza_pct > _p("distanza_alta"):
        punteggio += 2
    elif distanza_pct > _p("distanza_media"):
        punteggio += 1
    elif distanza_pct < _p("distanza_bassa"):
        punteggio -= 1

    # 6. Volatilità (ATR)
    atr_pct = atr / close
    if atr_pct > _p("atr_buono"):
        punteggio += 2
    elif atr_pct > _p("atr_minimo"):
        punteggio += 1
    elif atr_pct < _p("atr_basso"):
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
    segnale,
    hist=None,
    tp=None,
    sl=None,
    escursione_media=None,
    supporto=None
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
    low, high = _p("macd_rsi_range")
    if low < rsi < high:
        punteggio -= 5


    # 3. MACD coerente con segnale
    macd_gap = macd - macd_signal
    if segnale == "BUY":
        if macd > macd_signal and macd_gap > 0.001:
            punteggio += 10
        elif macd > 0 and macd_gap > 0:
            punteggio += 5
        elif abs(macd_gap) < _p("macd_signal_threshold"):  
            punteggio -= 5
    elif segnale == "SELL":
        if macd < macd_signal and macd_gap < -0.001:
            punteggio += 10
        elif macd < 0 and macd_gap < 0:
            punteggio += 5
        elif abs(macd_gap) < _p("macd_signal_threshold"):   
            punteggio -= 5

    # 4. Volume coerente
    if volume_attuale > volume_medio * _p("volume_alto"):
        punteggio += 10
    elif volume_attuale < volume_medio * _p("volume_basso"):
        punteggio -= 5

    # 5. Breakout forte
    if breakout:
        punteggio += 10

    # 6. Accelerazione coerente
    soglia_acc = _p("accelerazione_minima")

    if segnale == "BUY" and accelerazione > soglia_acc:
        punteggio += 5
    elif segnale == "SELL" and accelerazione < -soglia_acc:
        punteggio += 5
    else:
        punteggio -= 5


    # 7. Trend attivo da almeno 3-4 candele
    if 3 <= candele_attive <= 7:
        punteggio += 10
    elif candele_attive < 3:
        punteggio -= 5
    elif candele_attive > 8:
        punteggio -= 5

    # 8. Distanza EMA significativa
    distanza_pct = distanza_ema / close
    if distanza_pct > _p("distanza_media"):
        punteggio += 5
    elif distanza_pct < _p("distanza_bassa"):
        punteggio -= 5

    # 9. Volatilità accettabile (ATR)
    atr_pct = atr / close
    if atr_pct > _p("atr_buono"):
        punteggio += 5
    elif atr_pct < _p("atr_basso"):
        punteggio -= 5

    # 10. Penalità se ATR assoluto troppo basso/alto
    if atr < _p("atr_troppo_basso") or atr > _p("atr_troppo_alto"):
        punteggio -= 10

    # ------------------------------------------
    # 🔍 Estensioni avanzate (filtri ex-bloccanti)
    # ------------------------------------------

    if hist is not None:
        try:
            # a. Curvatura EMA25
            curvatura_ema25 = ema25 - hist["EMA_25"].iloc[-2]
            if segnale == "BUY" and curvatura_ema25 <= 0:
                punteggio -= 4
            elif segnale == "SELL" and curvatura_ema25 >= 0:
                punteggio -= 4

            # b. Escursione media troppo bassa
            if escursione_media / close < 0.002:
                punteggio -= 3

            # c. Candela attuale contraria
            open_attuale = hist["open"].iloc[-1]
            if segnale == "BUY" and close < open_attuale:
                punteggio -= 3
            elif segnale == "SELL" and close > open_attuale:
                punteggio -= 3

            # d. Supporto vicino (solo per SELL)
            if segnale == "SELL":
                distanza_supporto = abs(close - supporto) / close
                if distanza_supporto < 0.005:
                    punteggio -= 5

            # e. Rischio/Rendimento sfavorevole
            if tp and sl:
                rischio = abs(close - sl)
                rendimento = abs(tp - close)
                if rischio == 0 or rendimento / rischio < 1.5:
                    punteggio -= 5

        except Exception as e:
            import logging
            logging.warning(f"⚠️ Errore calcolo penalità avanzate: {e}")

    # ------------------------------------------
    # 🧮 Normalizzazione finale
    # ------------------------------------------
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
    escursione_media = (hist["high"] - hist["low"]).iloc[-10:].mean()
    distanza_ema = abs(ema7 - ema25)
    distanza_ok = (distanza_ema / close) > _p("distanza_minima")
    curvatura_ema25 = ema25 - penultimo["EMA_25"]
    curvatura_precedente = penultimo["EMA_25"] - antepenultimo["EMA_25"]
    accelerazione = curvatura_ema25 - curvatura_precedente
    trend_up, trend_down = is_trend_up(ema7, ema25, ema99), is_trend_down(ema7, ema25, ema99)
    recupero_buy = ema7 > ema25 < ema99 and close > ema25 and ema25 > penultimo["EMA_25"]
    recupero_sell = ema7 < ema25 > ema99 and close < ema25 and ema25 < penultimo["EMA_25"]
    candele_trend_up = conta_candele_trend(hist, rialzista=True)
    candele_trend_down = conta_candele_trend(hist, rialzista=False)
    macd_gap = macd - macd_signal
    gap_forte = _p("macd_gap_forte")
    gap_debole = _p("macd_gap_debole")

    # ------------------------------------------------------------------
    # Filtri preliminari (ATR, Volume, distanza)
    # ------------------------------------------------------------------
    if atr / close < _p("atr_minimo"):
        note.append("⚠️ ATR troppo basso: mercato poco volatile")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    if volume_attuale < _p("volume_soglia") and not MODALITA_TEST:
        note.append(f"⚠️ Volume troppo basso: {volume_attuale:.0f} < soglia minima {_p('volume_soglia')}")
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
    desc = trend_score_description(punteggio_trend)
    if desc:
        note.append(desc)

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
        if corpo_candela > atr:
            note.append("🚨 Spike ribassista con breakout solido")
            breakout_valido = True

    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout sospetto: volume insufficiente")

    # ------------------------------------------------------------------
    # Condizioni MACD / RSI
    # ------------------------------------------------------------------
    macd_buy_ok = macd > macd_signal and macd_gap > gap_forte
    macd_buy_debole = macd > 0 and macd_gap > gap_debole
    
    macd_sell_ok = macd < macd_signal and macd_gap < -gap_forte
    macd_sell_debole = macd < 0 and macd_gap < -gap_debole
    
    segnale, tp, sl = "HOLD", 0.0, 0.0
    probabilita = 50

    # Se il trend c'è ma la distanza è insufficiente, spiega perché resti HOLD
    if not distanza_ok and (trend_up or recupero_buy or trend_down or recupero_sell):
        note.append(f"📏 Distanza EMA troppo bassa ({(distanza_ema/close):.4f} < {_p('distanza_minima'):.4f})")
        
    # ------------------------------------------------------------------
    # BUY logic
    # ------------------------------------------------------------------
    if (trend_up or recupero_buy or breakout_valido) and distanza_ok:
        durata_trend = candele_trend_up
        if rsi >= _p("rsi_buy_forte") and macd_buy_ok and punteggio_trend >= SOGLIA_PUNTEGGIO:
            if durata_trend >= 10:
                note.append(f"⛔ Trend rialzista troppo maturo ({durata_trend} candele)")
            #elif accelerazione < -_p("accelerazione_minima"):
                #note.append(f"⚠️ BUY evitato: accelerazione negativa ({accelerazione:.6f})")
            else:
                segnale = "BUY"
                #note.append(f"🕒 Trend BUY attivo da {durata_trend} candele")
                note.append("✅ BUY confermato")
        elif rsi >= _p("rsi_buy_debole") and macd_buy_debole:
            if punteggio_trend >= SOGLIA_PUNTEGGIO + 2 and candele_trend_up <= 10:
                segnale = "BUY"
                note.append("✅ BUY confermato (setup debole + punteggio alto)")
            else:
                note.append("🤔 Segnale rialzista debole: RSI > 50 e MACD > signal, ma segnale incerto")


    # ------------------------------------------------------------------
    # SELL logic
    # ------------------------------------------------------------------
    if (trend_down or recupero_sell) and distanza_ok:
        durata_trend = candele_trend_down
        if rsi <= _p("rsi_sell_forte") and macd_sell_ok and punteggio_trend <= -SOGLIA_PUNTEGGIO:
            if durata_trend >= 10:
                note.append(f"⛔ Trend ribassista troppo maturo ({durata_trend} candele)")
            #elif accelerazione > _p("accelerazione_minima"):
                #note.append(f"⚠️ SELL evitato: accelerazione in risalita ({accelerazione:.6f})")
            else:
                segnale = "SELL"
                #note.append(f"🕒 Trend SELL attivo da {durata_trend} candele")
                note.append("✅ SELL confermato")
        elif rsi <= _p("rsi_sell_debole") and macd_sell_debole:
            if punteggio_trend <= -SOGLIA_PUNTEGGIO - 2 and candele_trend_down <= 10:
                segnale = "SELL"
                note.append("✅ SELL confermato (setup debole + punteggio alto)")
            else:
                note.append("🤔 Segnale ribassista debole: RSI < soglia e MACD < signal, ma segnale incerto")


    if segnale == "HOLD" and not any([trend_up, trend_down]):
        note.append("🔎 Nessun segnale valido rilevato: condizioni insufficienti")

    if segnale == "HOLD" and False:
        note.append(
            f"🧪 DEBUG – rsi={rsi:.1f} macd_gap={macd_gap:.5f} "
            f"gap_rel={abs(macd_gap)/close:.6f} punteggio={punteggio_trend} "
            f"distEMA%={(distanza_ema/close)*100:.3f} candele_up={candele_trend_up} candele_down={candele_trend_down}"
        )


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
        dist_level = valuta_distanza(distanza_ema, close)
        note.insert(0, f"📊 Trend attivo da {n_candele} candele | Distanza: {dist_level}")
        if pattern:
            note.append(f"✅ Pattern candlestick rilevato: {pattern}")
    else:
        if trend_up and candele_trend_up <= 2:
            note.append("🔼 Trend attivo")
        elif trend_down and candele_trend_down <= 2:
            note.append("🔽 Trend ribassista")
        elif candele_trend_up <= 1 and not trend_up:
            note.append("🔚 Trend concluso: attenzione a inversioni")

    # ------------------------------------------------------------------
    # Invalidation per pattern contrario o neutralità MACD/RSI
    # ------------------------------------------------------------------
    if pattern_contrario(segnale, pattern):
        note.append(f"⚠️ Pattern contrario: possibile inversione ({pattern})")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    low, high = _p("macd_rsi_range")
    soglia_macd = _p("macd_signal_threshold")

    if segnale in ["BUY", "SELL"] and low < rsi < high and abs(macd_gap) < soglia_macd:
        note.append(f"⚠️ RSI ({rsi:.1f}) e MACD neutri (gap={macd_gap:.5f}): probabilità ridotta")
        probabilita = max(probabilita - 10, 5)

    #if segnale in ["BUY", "SELL"] and low < rsi < high and abs(macd_gap) < soglia_macd:
        #note.append(f"⚠️ RSI ({rsi:.1f}) e MACD neutri (gap={macd_gap:.5f}): segnale evitato")
        #return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto


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

    # Calcolo probabilità di successo (con penalità avanzate)
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
                segnale=segnale,
                hist=hist,      # ⬅️ Aggiunto per candela contraria, curvatura, ecc.
                tp=tp,          # ⬅️ Aggiunto per rischio/rendimento
                sl=sl,           # ⬅️ Aggiunto per rischio/rendimento
                supporto=supporto,
                escursione_media=escursione_media

            )
            note.append(f"📊 Prob. successo stimata: {probabilita}%")
    except Exception as e:
        logging.warning(f"⚠️ Errore calcolo probabilità successo: {e}")
        
    # ------------------------------------------------------------------
    # Calcolo TP/SL realistico in base a probabilità, ATR e qualità trend
    # ------------------------------------------------------------------

    if segnale in ["BUY", "SELL"]:
        # ------------------- Fusione probabilità con punteggio_trend + penalità maturità -------------------
        durata = candele_trend_up if segnale == "BUY" else candele_trend_down
        pen_maturita = max(0, durata - 8) * 0.5        # -0.5 al trend per ogni candela oltre 8
        punteggio_trend_adj = punteggio_trend - pen_maturita

        prob_fusa = _fuse_prob(punteggio_trend_adj, probabilita)  # 0..1
        note.append(f"🧪 Probabilità fusa (trend+contesto): {round(prob_fusa*100)}%")

        # ------------------- Gate unico di entrata coerente con prob_fusa -------------------
        P_ENTER = 0.55  # 55% equivalente (regolabile)
        if prob_fusa < P_ENTER:
            note.append(f"⏸️ Gate non superato: prob_fusa {prob_fusa:.2f} < {P_ENTER:.2f}")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

        # ------------------- TP/SL proporzionali alla probabilità fusa -------------------
        ATR_MIN_FRAC = 0.003
        TP_BASE, TP_SPAN = 1.1, 1.2     # TP = 1.1x..2.3x ATR (crescente con prob_fusa)
        SL_BASE, SL_SPAN = 1.0, 0.4     # SL = 1.0x..0.6x ATR (decrescente con prob_fusa)
        RR_MIN = 1.5
        DELTA_MINIMO = 0.1
        TICK = 0.0001  # TODO: sostituisci con il tick_size reale del symbol

        atr_eff = atr if atr and atr > 0 else close * ATR_MIN_FRAC

        tp_mult = TP_BASE + TP_SPAN * prob_fusa
        sl_mult = SL_BASE - SL_SPAN * prob_fusa
        if tp_mult <= sl_mult:
            tp_mult = sl_mult + DELTA_MINIMO

        if segnale == "BUY":
            tp_raw = close + atr_eff * tp_mult
            sl_raw = close - atr_eff * sl_mult
        else:  # SELL
            tp_raw = close - atr_eff * tp_mult
            sl_raw = close + atr_eff * sl_mult

        # Correzione spread realistica (ask/bid)
        half_spread = (spread or 0.0) / 2.0
        if segnale == "BUY":
            tp_raw += half_spread
            sl_raw -= half_spread
        else:
            tp_raw -= half_spread
            sl_raw += half_spread

        # Enforce RR minimo (prima allargo TP)
        risk = abs(close - sl_raw)
        reward = abs(tp_raw - close)
        if risk == 0 or reward / risk < RR_MIN:
            needed_tp = close + (RR_MIN * risk) if segnale == "BUY" else close - (RR_MIN * risk)
            tp_raw = needed_tp
            reward = abs(tp_raw - close)

        # ------------------- Adattamento TP/SL alla finestra temporale stimata -------------------
        #Tempo target (timeframe 15m => 0.25h per candela)
        T_TARGET_MIN_H = 0.7          # evita target troppo vicino
        T_TARGET_MAX_H = 6.0          # oltre è poco efficiente
        T_HARD_CAP_H   = 12.0         # veto se troppo lungo
        MAX_TP_SPAN_ATR = 3.0         # non allargare TP oltre 3x ATR

        # Stima tempo con l'attuale TP
        distanza_tp = abs(tp_raw - close)
        range_medio = (hist["high"] - hist["low"]).iloc[-10:].mean()
        if range_medio == 0:
            range_medio = atr_eff if atr_eff > 0 else close * 0.003

        # efficienza dinamica in base a prob_fusa (0..1) → 0.2..0.7
        eff_min = 0.2 + 0.3 * prob_fusa
        eff_max = eff_min + 0.2
        ore_min = round((distanza_tp / (range_medio * eff_max)) * 0.25, 2)
        ore_max = round((distanza_tp / (range_medio * eff_min)) * 0.25, 2)

        # Se troppo lento, prova a stringere TP mantenendo RR minimo
        if ore_max > T_TARGET_MAX_H:
            scala = T_TARGET_MAX_H / ore_max
            nuova_dist = distanza_tp * max(0.4, min(1.0, scala))  # non oltre -60% in un colpo
            tp_candidato = (close + nuova_dist) if segnale == "BUY" else (close - nuova_dist)
            reward_cand = abs(tp_candidato - close)
            if risk > 0 and reward_cand / risk >= RR_MIN:
                tp_raw = tp_candidato
                distanza_tp = reward_cand
                ore_min = round((distanza_tp / (range_medio * eff_max)) * 0.25, 2)
                ore_max = round((distanza_tp / (range_medio * eff_min)) * 0.25, 2)
            else:
                note.append("⚠️ TP non ridotto: RR sarebbe < minimo")

        # Se ancora troppo lento, veto hard
        if ore_max > T_HARD_CAP_H:
            note.append(f"⛔ Tempo stimato troppo lungo: {ore_min}–{ore_max}h (cap {T_HARD_CAP_H}h)")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

        # Se troppo veloce, allarga TP (entro un limite) preservando RR
        if ore_min < T_TARGET_MIN_H:
            dist_attuale_atr = distanza_tp / (atr_eff if atr_eff > 0 else 1)
            if dist_attuale_atr < MAX_TP_SPAN_ATR:
                scala_up = T_TARGET_MIN_H / max(ore_min, 0.05)
                nuova_dist = min(distanza_tp * scala_up, MAX_TP_SPAN_ATR * atr_eff)
                tp_candidato = (close + nuova_dist) if segnale == "BUY" else (close - nuova_dist)
                reward_cand = abs(tp_candidato - close)
                if risk > 0 and reward_cand / risk >= RR_MIN:
                    tp_raw = tp_candidato
                    distanza_tp = reward_cand
                    ore_min = round((distanza_tp / (range_medio * eff_max)) * 0.25, 2)
                    ore_max = round((distanza_tp / (range_medio * eff_min)) * 0.25, 2)

        # Nota tempi finali
        note.append(f"⏱️ Tempo stimato TP: ~{ore_min}–{ore_max}h")

        # ------------------- Rounding a tick e safety finale -------------------
        def _round_tick(x, tick=TICK):
            return round(round(x / tick) * tick, 10)

        tp = _round_tick(tp_raw)
        sl = _round_tick(sl_raw)

        if segnale == "BUY" and not (sl < close < tp):
            note.append("⚠️ TP/SL BUY incoerenti, ricalcolo consigliato")
        if segnale == "SELL" and not (tp < close < sl):
            note.append("⚠️ TP/SL SELL incoerenti, ricalcolo consigliato")


    # ------------------------------------------------------------------
    # Calcolo tempo stimato per raggiungere TP (forchetta realistica)
    # ------------------------------------------------------------------
    try:
        if segnale in ["BUY", "SELL"] and tp > 0:
            distanza = abs(tp - close)

            # Escursione media delle ultime 10 candele
            range_medio = (hist["high"] - hist["low"]).iloc[-10:].mean()

            if range_medio == 0:
                raise ValueError("Range medio nullo")

            # Efficienza adattiva in base al punteggio del trend
            if punteggio_trend >= 4:
                eff_min, eff_max = 0.4, 0.7   # Trend forte
            elif punteggio_trend >= 2:
                eff_min, eff_max = 0.3, 0.55  # Trend moderato
            else:
                eff_min, eff_max = 0.2, 0.4   # Trend debole

            ore_min = round((distanza / (range_medio * eff_max)) * 0.25, 1)
            ore_max = round((distanza / (range_medio * eff_min)) * 0.25, 1)

            if ore_min == ore_max:
                note.append(f"🎯 Target stimato in ~{ore_min}h")
            #else:
                #note.append(f"🎯 Target stimato tra ~{ore_min}h e {ore_max}h")
    except Exception as e:
        logging.warning(f"⚠️ Errore calcolo tempo stimato: {e}")


    
    logging.debug("✅ Analisi completata")


    if segnale not in ["BUY", "SELL"]:
        if not note:
            note.append("ℹ️ Nessun criterio abbastanza forte per un segnale operativo")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # Ordina note: conferme → punteggi → warning → altro
    priorita = lambda x: (
        0 if "✅" in x else
        1 if "📊" in x or "⏱️" in x else
        2 if "⚠️" in x or "⛔" in x else
        3
    )
    note = list(dict.fromkeys(note))  # Elimina duplicati
    note.sort(key=priorita)


    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
