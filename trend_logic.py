import pandas as pd
from patterns import (
    detect_double_bottom,
    detect_ascending_triangle,
    detect_price_channel,
    detect_candlestick_patterns,
)
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
SOLO_BUY = False  # ⬅️ (2) SELL abilitati per i test

# -------------------------------------------------------------------------
# STRATEGIA GLOBALE DEL BACKEND (EMA / DB / TRI)
# -------------------------------------------------------------------------
STRATEGIA = "EMA"   # ⬅️ (1) Cambia qui: "EMA", "DB" oppure "TRI"


# --- Override BUY da pattern strutturali ---
CONF_MIN_PATTERN = 0.45        # confidenza minima del pattern per l'override
VOL_MULT_TRIANGLE = 1.30       # volume > 1.3x media per override triangolo
PATTERN_SOFT_BOOST_DB  = 0.03  # +3% * conf su prob_fusa se doppio minimo
PATTERN_SOFT_BOOST_TRI = 0.02  # +2% * conf su prob_fusa se triangolo


# Parametri separati per test / produzione
_PARAMS_TEST = {
    "volume_soglia": 120,
    "volume_alto": 1.8,
    "volume_medio": 1.2,
    "volume_basso": 0.7,
    "volume_molto_basso": 0.4,

    "atr_minimo": 0.0005,       # ⬅️ (9) ATR minimo più realistico in test
    "atr_buono": 0.001,
    "atr_basso": 0.0005,
    "atr_troppo_basso": 0.0001,
    "atr_troppo_alto": 0.01,

    "distanza_minima": 0.0008,
    "distanza_bassa": 0.0003,
    "distanza_media": 0.0008,
    "distanza_alta": 0.0015,

    "macd_rsi_range": (45, 55),
    "macd_signal_threshold": 0.00015,  # assoluta
    "macd_gap_forte": 0.0005,
    "macd_gap_debole": 0.0002,
    "macd_gap_rel_forte": 0.0006,  
    "macd_gap_rel_debole": 0.00025,

    "rsi_buy_forte": 52,
    "rsi_buy_debole": 50,
    "rsi_sell_forte": 42,
    "rsi_sell_debole": 46,

    "accelerazione_minima": 0.00003,
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
    "macd_gap_rel_forte": 8e-4,   # 0.08% del prezzo
    "macd_gap_rel_debole": 3e-4,

    "rsi_buy_forte": 54,
    "rsi_buy_debole": 50,
    "rsi_sell_forte": 46,
    "rsi_sell_debole": 50,

    "accelerazione_minima": 0.00001,
}


# --- Helper per resample OHLCV (15m -> 1h) ---
def _resample_ohlcv(df: pd.DataFrame, rule: str = "1H") -> pd.DataFrame:
    if not {"open","high","low","close","volume"}.issubset(df.columns):
        return pd.DataFrame()
    _df = df[["open","high","low","close","volume"]].copy()

    # Garantisce DatetimeIndex (se serve prova a convertire l'indice)
    if not isinstance(_df.index, pd.DatetimeIndex):
        try:
            _df.index = pd.to_datetime(_df.index, utc=True, errors="coerce")
        except Exception:
            return pd.DataFrame()
    _df = _df.dropna(subset=["open","high","low","close"])

    out = (
        _df.resample(rule, label="left", closed="left")  # ⬅️ (12) Allineamento stile Binance/TV
           .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
           .dropna()
    )
    return out



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
# Helper comuni – sicurezza divisioni & arricchimento indicatori
# -----------------------------------------------------------------------------
def _safe_close(close: float, eps_rel: float = 1e-9) -> float:
    """Ritorna un close sicuro (>0) per evitare divisioni problematiche."""
    try:
        if close is None or not pd.notna(close):
            return eps_rel
        return close if close > eps_rel else eps_rel
    except Exception:
        return eps_rel

def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    """Divisione sicura con denominatore clampato."""
    d = den if den and abs(den) > eps else eps
    return num / d

def _frac_of_close(value: float, close: float) -> float:
    """value / close in modo sicuro."""
    return _safe_div(value, _safe_close(close))


def enrich_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge EMA, RSI, ATR, MACD (se mancanti) al DataFrame."""
    missing_cols = {"EMA_7", "EMA_25", "EMA_99"} - set(hist.columns)
    if missing_cols:
        ema = calcola_ema(hist, [7, 25, 99])
        hist["EMA_7"], hist["EMA_25"], hist["EMA_99"] = ema[7], ema[25], ema[99]

    if "RSI" not in hist.columns:
        hist["RSI"] = calcola_rsi(hist["close"])

    if not {"MACD", "MACD_SIGNAL"}.issubset(hist.columns):
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
        return "🔥 Trend↑ Forte"
    if score >= 2:
        return "👍 Trend↑ Moderato"
    if score == 1:
        return "🟡 Trend↑ Debole"
    if score == 0:
        return "🔍 Trend Neutro"
    if score == -1:
        return "🟠 Trend↓ Debole"
    if score <= -4:
        return "❌ Trend↓ Forte"
    if score <= -2:
        return "⚠️ Trend↓ Moderato"
    return ""


def pattern_contrario(segnale: str, pattern: str) -> bool:
    """(10+11) Pattern candela contrario alla direzione del segnale."""
    if not pattern:
        return False

    # Normalizza (togli emoji, lascia le parole chiave)
    # In pratica: lavoriamo per "substring" sui nomi noti.
    bearish_keys = [
        "Shooting Star",
        "Bearish Engulfing",
        "Evening Star",
        "Dark Cloud",
        "Hanging Man",
        "Bearish Harami",
    ]
    bullish_keys = [
        "Hammer",
        "Bullish Engulfing",
        "Bullish Harami",
        "Three White Soldiers",
    ]

    if segnale == "BUY":
        return any(k in pattern for k in bearish_keys)
    if segnale == "SELL":
        return any(k in pattern for k in bullish_keys)
    return False


def sintetizza_canale(chan: dict, solo_buy: bool = True) -> str:
    """
    Restituisce una riga corta per la notifica Android, es.: 
    '🛤️ Canale ↑ 74% • Compra'
    Operatività: 1 parola (Compra / Vendi / Attendi / Evita).
    """
    if not chan or not chan.get("found"):
        return ""

    tipo = chan.get("type", "")
    freccia = {"ascending": "↑", "descending": "↓", "sideways": "↔︎"}.get(tipo, "•")
    conf = int(round(float(chan.get("confidence", 0) * 100)))

    # Operatività (1 parola)
    op = "Attendi"
    bo = bool(chan.get("breakout_confirmed"))
    side = chan.get("breakout_side", "")

    if tipo == "ascending":
        op = "Compra" if (bo and side != "down") or conf >= 70 else "Attendi"
    elif tipo == "sideways":
        if bo:
            op = "Compra" if side == "up" else ("Evita" if solo_buy else "Vendi")
        else:
            op = "Attendi"
    elif tipo == "descending":
        op = "Evita" if solo_buy else ("Vendi" if (bo and side == "down") else "Attendi")

    return f"🛤️ Canale {freccia} {conf}% • {op}"


# -----------------------------------------------------------------------------
# Funzioni originali, ripulite dalle ripetizioni
# -----------------------------------------------------------------------------
def valuta_distanza(distanza: float, close: float) -> str:
    distanza_pct = _frac_of_close(distanza, close)
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

def conta_candele_reali(hist: pd.DataFrame, side: str = "BUY", max_candele: int = 50) -> int:
    """
    Conteggio 'reale': quante candele consecutive rispettano la condizione operativa
    che usi per entrare.
      BUY : trend_up (7>25>99)  OPPURE recupero (7>25, 25<99, close>25, EMA25 in salita)
      SELL: trend_down (7<25<99) OPPURE recupero (7<25, 25>99, close<25, EMA25 in discesa)
    """
    if len(hist) < 3:
        return 0

    count = 0
    for i in range(len(hist) - 1, max(-1, len(hist) - 1 - max_candele), -1):
        if i - 1 < 0:
            break
        e7   = hist["EMA_7"].iloc[i]
        e25  = hist["EMA_25"].iloc[i]
        e99  = hist["EMA_99"].iloc[i]
        c    = hist["close"].iloc[i]
        e25p = hist["EMA_25"].iloc[i - 1]

        if side == "BUY":
            cond_trend    = (e7 > e25 > e99)
            cond_recupero = (e7 > e25) and (e25 < e99) and (c > e25) and (e25 > e25p)
            cond = cond_trend or cond_recupero
        else:  # SELL
            cond_trend    = (e7 < e25 < e99)
            cond_recupero = (e7 < e25) and (e25 > e99) and (c < e25) and (e25 < e25p)
            cond = cond_trend or cond_recupero

        if cond:
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


def riconosci_pattern_candela(df: pd.DataFrame) -> str:
    """Ritorna una stringa col pattern rilevato.

    Ordine di priorità:
      1) pattern 'classici' interni (Hammer, Engulfing, Soldiers, Harami)
      2) se nulla, fallback sui pattern BUY da patterns.detect_candlestick_patterns
         (Morning Star, Three White Soldiers, Piercing Line, Tweezer Bottom, Doji).
    """
    if len(df) < 3:
        return ""

    c1 = df.iloc[-1]
    o1, h1, l1, c1c = c1["open"], c1["high"], c1["low"], c1["close"]
    body1     = c1c - o1
    body1_abs = abs(body1)
    upper1 = h1 - max(o1, c1c)
    lower1 = min(o1, c1c) - l1

    # ---- pattern classici (tuo codice esistente) ----
    if body1_abs != 0:
        # Hammer
        if body1 > 0 and lower1 >= 2 * body1_abs and upper1 <= body1_abs * 0.3:
            return "🪓 Hammer"
        # Shooting Star
        if body1 < 0 and upper1 >= 2 * body1_abs and lower1 <= body1_abs * 0.3:
            return "🌠 Shooting Star"

        if len(df) >= 2:
            o0, c0 = df["open"].iloc[-2], df["close"].iloc[-2]
            # Bullish Engulfing
            if body1 > 0 and c1c > o0 and o1 < c0:
                return "🔄 Bullish Engulfing"
            # Bearish Engulfing
            if body1 < 0 and c1c < o0 and o1 > c0:
                return "🔃 Bearish Engulfing"

    # Three White Soldiers (versione interna, come prima)
    if len(df) >= 4:
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        def bull(x): return x["close"] > x["open"]
        def body(x): return abs(x["close"] - x["open"])
        def upper_wick(x): return x["high"] - max(x["open"], x["close"])
        def lower_wick(x): return min(x["open"], x["close"]) - x["low"]

        if bull(a) and bull(b) and bull(c):
            if (c["close"] > b["close"] > a["close"] and
                body(b) >= 0.7 * body(a) and body(c) >= 0.7 * body(b)):
                cond_open = (b["open"]  >= min(a["open"], a["close"]) and b["open"]  <= max(a["open"], a["close"]) and
                             c["open"]  >= min(b["open"], b["close"]) and c["open"]  <= max(b["open"], b["close"]))
                cond_wicks = (upper_wick(a) <= body(a) and lower_wick(a) <= body(a) and
                              upper_wick(b) <= body(b) and lower_wick(b) <= body(b) and
                              upper_wick(c) <= body(c) and lower_wick(c) <= body(c))
                if cond_open and cond_wicks:
                    return "🟩 Three White Soldiers"

    # Bullish Harami (come prima)
    if len(df) >= 2 and body1_abs != 0:
        prev = df.iloc[-2]
        recent = (df["close"] - df["open"]).abs().iloc[-10:-2]
        avg_body = recent.mean() if len(recent) else (abs(prev["close"] - prev["open"]) or 1e-9)

        prev_red  = prev["close"] < prev["open"]
        big_prev  = abs(prev["close"] - prev["open"]) > 1.1 * avg_body
        small_now = body1_abs < 0.6 * abs(prev["close"] - prev["open"])

        dentro = (min(o1, c1c) >= min(prev["open"], prev["close"]) and
                  max(o1, c1c) <= max(prev["open"], prev["close"]))

        if prev_red and (c1c > o1) and big_prev and small_now and dentro:
            return "🤰 Bullish Harami"

    # ------------------------------------------------
    # Fallback: pattern avanzati pro-BUY da patterns.py
    # ------------------------------------------------
    try:
        atr_series = df["ATR"] if "ATR" in df.columns else None
        vol_series = df["volume"] if "volume" in df.columns else None
        patt_list = detect_candlestick_patterns(df, atr=atr_series, vol=vol_series)
    except Exception as e:
        logging.warning(f"⚠️ Errore detect_candlestick_patterns: {e}")
        return ""

    if not patt_list:
        return ""

    # preferisci pattern bull, poi il più confidente
    bulls = [p for p in patt_list if p.get("direction") == "bull"]
    candidates = bulls or patt_list
    best = max(candidates, key=lambda p: p.get("confidence", 0.0))

    name = best.get("name", "")
    if not name:
        return ""

    # mapping semplice per icone (puoi ampliarlo se vuoi)
    emoji_map = {
        "Morning Star": "🌅",
        "Three White Soldiers": "🟩",
        "Piercing Line": "📈",
        "Tweezer Bottom": "🧲",
        "Dragonfly Doji": "🪰",
        "Long-legged Doji": "🦵",
    }
    prefix = emoji_map.get(name, "✨")
    return f"{prefix} {name}"



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
# Punteggio trend
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

    # 3. MACD (usiamo gap *relativo* sul prezzo, come nel resto della logica)
    macd_gap = macd - macd_signal
    gap_rel = _frac_of_close(macd_gap, close)

    if gap_rel > _p("macd_gap_rel_forte"):
        punteggio += 2
    elif gap_rel > _p("macd_gap_rel_debole"):
        punteggio += 1
    elif gap_rel < -_p("macd_gap_rel_forte"):
        punteggio -= 2
    elif gap_rel < -_p("macd_gap_rel_debole"):
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
    distanza_pct = _frac_of_close(distanza_ema, close)
    if distanza_pct > _p("distanza_alta"):
        punteggio += 2
    elif distanza_pct > _p("distanza_media"):
        punteggio += 1
    elif distanza_pct < _p("distanza_bassa"):
        punteggio -= 1

    # 6. Volatilità (ATR)
    atr_pct = _frac_of_close(atr, close)
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
    supporto=None,
    pattern_candela=None,
    pattern_db_ok=False,
    pattern_tri_ok=False,
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

    # 5b. Pattern strutturali – uso SOLO degli OK passati da analizza_trend
    if pattern_db_ok:
        punteggio += 4
    if pattern_tri_ok:
        punteggio += 3

    
    # 6. Accelerazione coerente
    soglia_acc = _p("accelerazione_minima")
    if segnale == "BUY" and accelerazione > soglia_acc:
        punteggio += 5
    elif segnale == "SELL" and accelerazione < -soglia_acc:
        punteggio += 5
    elif abs(accelerazione) < soglia_acc * 0.5:
        # accelerazione quasi piatta → neutra (nessun bonus/penalità)
        pass
    else:
        # accelerazione moderatamente controtrend → piccola penalità
        punteggio -= 3

    # 7. Trend attivo da almeno 3-4 candele
    if 3 <= candele_attive <= 7:
        punteggio += 10
    elif candele_attive < 3:
        punteggio -= 3
    elif candele_attive > 8:
        punteggio -= 3


    # 8. Distanza EMA significativa
    distanza_pct = _frac_of_close(distanza_ema, close)
    if distanza_pct > _p("distanza_media"):
        punteggio += 5
    elif distanza_pct < _p("distanza_bassa"):
        punteggio -= 5

    # 9. Volatilità accettabile (ATR)
    atr_pct = _frac_of_close(atr, close)
    if atr_pct > _p("atr_buono"):
        punteggio += 5
    elif atr_pct < _p("atr_basso"):
        punteggio -= 5

    # 10. Penalità se ATR assoluto troppo basso/alto
    if atr < _p("atr_troppo_basso") or atr > _p("atr_troppo_alto"):
        punteggio -= 10

    # ------------------------------------------
    # ------------------------------------------
    # 11. Boost/malus per pattern candlestick
    # (i pattern strutturali sono già conteggiati al punto 5b)
    # ------------------------------------------
    try:
        if pattern_candela:
            patt = pattern_candela

            bullish_keys = [
                "Hammer",
                "Bullish Engulfing",
                "Three White Soldiers",
                "Bullish Harami",
                "Morning Star",
                "Piercing Line",
                "Tweezer Bottom",
                "Dragonfly Doji",
                "Long-legged Doji",
            ]
            bearish_keys = [
                "Shooting Star",
                "Bearish Engulfing",
                "Evening Star",       # se in futuro la aggiungi
                "Dark Cloud Cover",   # idem
            ]

            is_bullish = any(k in patt for k in bullish_keys)
            is_bearish = any(k in patt for k in bearish_keys)

            # pattern a favore del segnale
            if segnale == "BUY" and is_bullish:
                punteggio += 3
            if segnale == "SELL" and is_bearish:
                punteggio += 3

            # pattern contro il segnale (oltre a pattern_contrario che può già annullare)
            if segnale == "BUY" and is_bearish:
                punteggio -= 4
            if segnale == "SELL" and is_bullish:
                punteggio -= 4

    except Exception as e:
        logging.warning(f"⚠️ Errore contributo pattern in probabilità: {e}")

    # ------------------------------------------
    # 🔍 Penalità avanzate (NON bloccanti)
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
            if _frac_of_close(escursione_media, close) < 0.002:
                punteggio -= 3

            # c. Candela attuale contraria
            open_attuale = hist["open"].iloc[-1]
            if segnale == "BUY" and close < open_attuale:
                punteggio -= 3
            elif segnale == "SELL" and close > open_attuale:
                punteggio -= 3

            # d. Supporto vicino (solo SELL)
            if segnale == "SELL" and supporto is not None:
                distanza_supporto = _frac_of_close(abs(close - supporto), close)
                if distanza_supporto < 0.005:
                    punteggio -= 5

            # e. Rischio/Rendimento sfavorevole
            if tp and sl:
                rischio = abs(close - sl)
                rendimento = abs(tp - close)
                if rischio == 0 or _safe_div(rendimento, rischio) < 1.5:
                    punteggio -= 5

        except Exception as e:
            logging.warning(f"⚠️ Errore calcolo penalità avanzate: {e}")

    # ------------------------------------------
    # 🧮 Normalizzazione finale (5..95)
    # ------------------------------------------
    probabilita = min(max(round(punteggio * 1.25), 5), 95)
    return probabilita


# -----------------------------------------------------------------------------
# Funzione principale: analizza_trend
# -----------------------------------------------------------------------------
def analizza_trend(hist: pd.DataFrame, spread: float = 0.0, hist_1m: pd.DataFrame = None, sistema: str = "EMA"):

    logging.debug("🔍 Inizio analisi trend")
    hist = hist.copy()  
    if "volume" not in hist.columns:
        hist["volume"] = 0.0

    pump_flag = False
    pump_msg  = None
    if len(hist) < 22:
        try:
            ultimo = hist.iloc[-1]
            close_s = _safe_close(ultimo["close"])
            volume_attuale = ultimo.get("volume", 0)

            corpo = abs(ultimo["close"] - ultimo["open"])
            range_c = ultimo["high"] - ultimo["low"]
            body_frac = _safe_div(corpo, max(range_c, 1e-9))
            range_rel = _frac_of_close(range_c, close_s)

            # (7) Quick pump meno aggressivo in test: serve volume decente e range più ampio
            vol_ok = (
                volume_attuale >= _p("volume_soglia")
                if not MODALITA_TEST
                else volume_attuale >= _p("volume_soglia") * 0.5
            )

            cond_quick_pump = (
                (ultimo["close"] > ultimo["open"])
                and (body_frac >= 0.85)
                and (range_rel >= 0.04)  # prima 0.02 → più selettivo
                and vol_ok
            )

            if cond_quick_pump:
                pump_flag = True
                pump_msg  = "🚀 Possibile Pump (listing)"
               
            else:
                logging.warning("⚠️ Dati insufficienti per l'analisi")
                return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, None
        except Exception as e:
            logging.warning(f"Quick-pump listing check error: {e}")
            logging.warning("⚠️ Dati insufficienti per l'analisi")
            return "HOLD", hist, 0.0, "Dati insufficienti", 0.0, 0.0, None



    hist = enrich_indicators(hist)

    # (1) Strategia globale: ignora eventuale parametro esterno
    sistema = (STRATEGIA or "EMA").upper()
    if sistema not in {"EMA", "DB", "TRI"}:
        sistema = "EMA"

    note = []
    #note.append(f"🔧 Sistema: {sistema}")


    try:
        ultimo, penultimo, antepenultimo = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3]
        ema7, ema25, ema99 = ultimo[["EMA_7", "EMA_25", "EMA_99"]]
        close, rsi, atr, macd, macd_signal = ultimo[["close", "RSI", "ATR", "MACD", "MACD_SIGNAL"]]
        supporto = calcola_supporto(hist)
        close_s = _safe_close(close)
    except Exception as e:
        logging.error(f"❌ Errore nell'accesso ai dati finali: {e}")
        return "HOLD", hist, 0.0, "Errore su iloc finali", 0.0, 0.0, 0.0


    volume_attuale = hist["volume"].iloc[-1]
    volume_medio = hist["volume"].iloc[-21:-1].mean()
    if pd.isna(volume_medio) or volume_medio <= 0:
        k = min(5, max(1, len(hist)-1))
        finestra = hist["volume"].iloc[-k:-1]
        volume_medio = finestra.mean() if len(finestra) else volume_attuale


    n = min(21, max(1, len(hist)-1))
    prev_highs = hist["high"].iloc[-n:-1]
    prev_lows  = hist["low"].iloc[-n:-1]
    massimo_20 = prev_highs.max() if len(prev_highs) else ultimo["high"]
    minimo_20  = prev_lows.min()  if len(prev_lows)  else ultimo["low"]
    atr = float(atr) if pd.notna(atr) else 0.0

    
    escursione_media = (hist["high"] - hist["low"]).iloc[-10:].mean()
    if pd.isna(escursione_media) or escursione_media <= 0:
        escursione_media = max(
            (hist["high"] - hist["low"]).tail(3).mean(),
            atr,
            1e-9,
        )

    distanza_ema = abs(ema7 - ema25)
    distanza_ok = _frac_of_close(distanza_ema, close_s) > _p("distanza_minima")
    curvatura_ema25 = ema25 - penultimo["EMA_25"]
    curvatura_precedente = penultimo["EMA_25"] - antepenultimo["EMA_25"]
    accelerazione = curvatura_ema25 - curvatura_precedente
    trend_up, trend_down = is_trend_up(ema7, ema25, ema99), is_trend_down(ema7, ema25, ema99)
    recupero_buy = ema7 > ema25 < ema99 and close > ema25 and ema25 > penultimo["EMA_25"]
    recupero_sell = ema7 < ema25 > ema99 and close < ema25 and ema25 < penultimo["EMA_25"]
    
    # Conteggio "strict" (solo 7>25>99 o 7<25<99)
    candele_trend_up_strict = conta_candele_trend(hist, rialzista=True)
    candele_trend_down_strict = conta_candele_trend(hist, rialzista=False)
    # Conteggio "reale" (trend pieno O recupero progressivo)
    candele_reali_up = conta_candele_reali(hist, side="BUY")
    candele_reali_down = conta_candele_reali(hist, side="SELL")


    # MACD: gap normalizzato sul prezzo → robusto a simboli diversi
    macd_gap = macd - macd_signal
    gap_rel = _frac_of_close(abs(macd_gap), close_s)

    gap_rel_forte  = _p("macd_gap_rel_forte")
    gap_rel_debole = _p("macd_gap_rel_debole")

    macd_buy_ok      = (macd > macd_signal) and (gap_rel > gap_rel_forte)
    macd_buy_debole  = (macd > 0)            and (gap_rel > gap_rel_debole)

    macd_sell_ok     = (macd < macd_signal)  and (gap_rel > gap_rel_forte)
    macd_sell_debole = (macd < 0)            and (gap_rel > gap_rel_debole)

    pattern_buy_override = False

   


    # ------------------------------------------------------------------
    # Filtri preliminari (ATR, Volume)
    # ------------------------------------------------------------------
    if _frac_of_close(atr, close_s) < _p("atr_minimo") and not pump_flag:
        note.append("⚠️ ATR Basso: poco volatile")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    if volume_attuale < _p("volume_soglia") and not MODALITA_TEST and not pump_flag:
        note.append(f"⚠️ Volume Basso: {volume_attuale:.0f} < soglia minima {_p('volume_soglia')}")
        return "HOLD", hist, 0.0, "\n".join(note).strip(), 0.0, 0.0, supporto

    # ------------------------------------------------------------------
    # Punteggio complessivo + descrizione
    # ------------------------------------------------------------------
    punteggio_trend = calcola_punteggio_trend(
        ema7, ema25, ema99, rsi, macd, macd_signal,
        volume_attuale, volume_medio, distanza_ema, atr, close,
    )
    note.append(f"📊 Trend score: {punteggio_trend}")
    desc = trend_score_description(punteggio_trend)
    if desc:
        note.append(desc)

    # ------------------------------------------------------------------
    # Breakout
    # ------------------------------------------------------------------
    breakout_valido = False
    corpo_candela = abs(ultimo["close"] - ultimo["open"])

    if close > massimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout↑ con Volume Alto")
        if corpo_candela > atr:
            note.append("🚀 Spike↑ con Breakout")
            breakout_valido = True
    elif close < minimo_20 and volume_attuale > volume_medio * 1.5:
        note.append("💥 Breakout↓ con Volume Alto")
        if corpo_candela > atr:
            note.append("🚨 Spike↓ con Breakout")
            breakout_valido = True
    elif (close > massimo_20 or close < minimo_20) and volume_attuale < volume_medio:
        note.append("⚠️ Breakout? Vol↓")

    # ✅ Promuovi il quick pump a breakout
    if pump_flag and not breakout_valido:
        breakout_valido = True
        punteggio_trend += 1

    # ------------------------------------------------------------------
    # Rilevamento Pump Verticale
    # ------------------------------------------------------------------

    try:
        corpo_candela = abs(ultimo["close"] - ultimo["open"])
        range_candela = ultimo["high"] - ultimo["low"]

        # Serie su 20 candele precedenti (esclude l'ultima)
        body_series = (hist["close"] - hist["open"]).iloc[-21:-1].abs()
        vol_series  = hist["volume"].iloc[-21:-1]

        # Riferimenti robusti (mediana)
        corpo_ref  = body_series.median() if len(body_series) else 0.0
        volume_ref = vol_series.median()  if len(vol_series)  else 0.0

        # Parametri (tienili qui per tuning rapido)
        BODY_MULT = 2.5      # corpo > 2.5× mediana corpi precedenti
        RANGE_ATR = 1.8      # range > 1.8× ATR
        VOL_MULT  = 2.0      # volume > 2× mediana
        WICK_MAX  = 0.35     # wicks totali < 35% del range
        BODY_FRAC = 0.65     # corpo ≥ 65% del range

        upper_wick = ultimo["high"] - max(ultimo["open"], ultimo["close"])
        lower_wick = min(ultimo["open"], ultimo["close"]) - ultimo["low"]
        wick_ratio = _safe_div(upper_wick + lower_wick, max(range_candela, 1e-9))
        body_frac  = _safe_div(corpo_candela,           max(range_candela, 1e-9))

        atr_eff = atr if atr > 0 else max(escursione_media, close_s * 0.003)
        cond_range = range_candela > RANGE_ATR * atr_eff
      
        cond_corpo    = corpo_candela > BODY_MULT * max(corpo_ref, 1e-9)
        cond_volume   = volume_attuale > VOL_MULT * max(volume_ref, 1e-9)
        cond_wick     = wick_ratio < WICK_MAX
        cond_bodyfrac = body_frac  >= BODY_FRAC

        # Richiedi sempre: volume alto + candela "piena"
        # Poi: o corpo enorme, oppure range enorme (vs ATR) con wicks contenute
        cond_pump = cond_volume and cond_bodyfrac and (cond_corpo or (cond_range and cond_wick))

        if cond_pump:
            pump_flag = True
            pump_msg = "🚀 Possibile Pump Verticale" if ultimo["close"] >= ultimo["open"] else "📉 Possibile Dump Verticale"
            breakout_valido = True
            punteggio_trend += 1

    except Exception as e:
        logging.warning(f"⚠️ Errore rilevamento pump: {e}")

    # ------------------------------------------------------------------
    # --- Canale di prezzo (15m + validazione 1h) ---
    # ------------------------------------------------------------------
    channel_prob_adj = 0.0   # verrà applicato più avanti, dopo il calcolo di prob_fusa
    gate_buy_canale  = False  # opzionale: blocca BUY contro canale 1h discendente forte

    try:
        chan_15 = detect_price_channel(
            hist, lookback=200, min_touches_side=2,
            parallel_tolerance=0.20, touch_tol_mult_atr=0.55,
            min_confidence=0.55, require_volume_for_breakout=True,
            breakout_vol_mult=1.30
        )
    except Exception as e:
        logging.warning(f"⚠️ Errore channel detection 15m: {e}")
        chan_15 = {}

    # Prova a costruire un 1h dal 15m (se non disponi già di hist_1h)
    try:
        df_1h = _resample_ohlcv(hist, "1H")
        if len(df_1h) >= 30:
            chan_1h = detect_price_channel(
                df_1h, lookback=200, min_touches_side=3,
                parallel_tolerance=0.15, touch_tol_mult_atr=0.50,
                min_confidence=0.65, require_volume_for_breakout=True,
                breakout_vol_mult=1.30
            )
        else:
            chan_1h = {}
    except Exception as e:
        logging.warning(f"⚠️ Errore channel detection 1h: {e}")
        chan_1h = {}

    if chan_15.get("found"):
        # allinea la tua logica esistente per il breakout 15m
        if chan_15.get("breakout_confirmed"):
            breakout_valido = True
            punteggio_trend += 1

        # boost/malus e gate in base al canale 1h
        if chan_1h.get("found"):
            t1h   = chan_1h.get("type", "")
            c1h   = float(chan_1h.get("confidence", 0.0))
            strong_1h = c1h >= 0.65

            if t1h == "ascending" and strong_1h:
                channel_prob_adj = +0.01 * c1h  # ⬅️ (6) effetto più morbido
            elif t1h == "descending" and strong_1h:
                channel_prob_adj = -0.01 * c1h  # ⬅️ (6) effetto più morbido
                if SOLO_BUY and not (chan_15.get("breakout_confirmed") and chan_15.get("breakout_side") == "up") and not pump_flag:
                    gate_buy_canale = True      # opzionale: blocca BUY contro-trend 1h

        # Notifica compatta su un rigo (icona, freccia, %, 1 parola operativa)
        # di default usiamo la sintesi 15m
        riga_canale = sintetizza_canale(chan_15, solo_buy=SOLO_BUY)

        # se 1h è discendente forte, forziamo l’operatività a una parola più prudente
        if chan_1h.get("found") and chan_1h.get("type") == "descending" and float(chan_1h.get("confidence",0)) >= 0.65:
            freccia = {"ascending":"↑","descending":"↓","sideways":"↔︎"}.get(chan_15.get("type",""), "•")
            conf15  = int(round(float(chan_15.get("confidence",0)*100)))
            op_word = "Evita" if SOLO_BUY else "Vendi"
            riga_canale = f"🛤️ Canale {freccia} {conf15}% • {op_word}"

        if riga_canale:
            note.append(riga_canale)


    # ------------------------------------------------------------------
    # Pattern strutturali (solo BUY)
    # ------------------------------------------------------------------
    p_db  = detect_double_bottom(hist, lookback=180, neckline_confirm=True) or {}
    p_tri = detect_ascending_triangle(hist, lookback=200, breakout_confirm=True) or {}

    def _pattern_recent(p: dict, max_age: int = 40) -> bool:
        pts = p.get("points", {}) or {}
        candidates = [
            p.get("breakout_index"),
            p.get("neckline_break_idx"),
            p.get("right_bottom_idx"),
            pts.get("bottom2_idx"),
            pts.get("bottom1_idx"),
        ]
        for idx in candidates:
            if idx is None:
                continue
            try:
                pos = hist.index.get_loc(idx)
            except Exception:
                try:
                    pos = int(idx)
                except Exception:
                    continue
            if (len(hist) - 1 - pos) <= max_age:
                return True
        # se non databile, mantieni permissivo come ora
        return True


    

    # (7+B) Gating robusto pattern + contesto tecnico
    pattern_db_ok = (
        p_db.get("found")
        and p_db.get("confidence", 0) >= CONF_MIN_PATTERN
        and _pattern_recent(p_db, max_age=40)
        and p_db.get("neckline_confirmed", p_db.get("neckline_breakout", False))
        and (macd_gap > 0) and (rsi > 50)
        and (trend_up or recupero_buy)
    )


    pattern_tri_ok = (
        p_tri.get("found")
        and p_tri.get("confidence", 0) >= CONF_MIN_PATTERN
        and _pattern_recent(p_tri, max_age=40)
        and p_tri.get("breakout_confirmed", False)
        and (volume_attuale > volume_medio * VOL_MULT_TRIANGLE)
        and (trend_up or recupero_buy)
    )



    # Mostra le note SOLO se il pattern supera i criteri
    if pattern_db_ok:
        note.append(f"🧩 {p_db.get('pattern', 'Double Bottom')} ({int(p_db.get('confidence',0)*100)}%)")
    if pattern_tri_ok:
        note.append(f"🧩 {p_tri.get('pattern', 'Ascending Triangle')} ({int(p_tri.get('confidence',0)*100)}%)")

    # Condizione base per strategia + override coerente
    if sistema == "EMA":
        cond_base = (trend_up or recupero_buy or breakout_valido or pump_flag)
        pattern_buy_override = False
    elif sistema == "DB":
        cond_base = pattern_db_ok
        pattern_buy_override = pattern_db_ok
    elif sistema == "TRI":
        cond_base = pattern_tri_ok
        pattern_buy_override = pattern_tri_ok
    else:
        cond_base = False
        pattern_buy_override = False

    # Se distanza EMA insufficiente, disattiva tutto
    if not distanza_ok and not pump_flag:
        cond_base = False
        pattern_buy_override = False


    # ------------------------------------------------------------------
    # Condizioni MACD / RSI → logica primaria BUY/SELL
    # ------------------------------------------------------------------
    segnale, tp, sl = "HOLD", 0.0, 0.0
    probabilita = 50

    # Se il trend c'è ma la distanza è insufficiente, spiega perché resti HOLD
    if not distanza_ok and (trend_up or recupero_buy or trend_down or recupero_sell):
        note.append(f"📏 Dist EMA Bassa ({_frac_of_close(distanza_ema, close_s):.4f} < {_p('distanza_minima'):.4f})")

    # ------------------------------------------------------------------
    # BUY logic (con RSI in crescita + override pattern)
    # ------------------------------------------------------------------
    
    if cond_base:
        if sistema in ("DB","TRI") and pattern_buy_override:
            segnale = "BUY"
            note.append("✅ BUY per Pattern override" + (" (Double Bottom)" if sistema=="DB" else " (Ascending Triangle)"))
        elif sistema == "EMA":
            durata_trend = candele_reali_up
            rsi_in_crescita = (rsi > penultimo["RSI"] > antepenultimo["RSI"])
            
            # Regole standard (tuo codice esistente)
            if (
                rsi >= _p("rsi_buy_forte")
                and macd_buy_ok
                and punteggio_trend >= SOGLIA_PUNTEGGIO
                and rsi_in_crescita
            ):
                # consenti anche trend lunghi se breakout/pump
                LIM_MATURITA = 10
                if durata_trend >= LIM_MATURITA and not (breakout_valido or pump_flag):
                    note.append(f"⛔ Trend↑ Maturo ({durata_trend} candele)")
                else:
                    segnale = "BUY"
                    note.append("✅ BUY confermato")

            elif (
                rsi >= _p("rsi_buy_debole")
                and macd_buy_debole
                and rsi_in_crescita
            ):
                if punteggio_trend >= SOGLIA_PUNTEGGIO + 1 and durata_trend <= 12:
                    segnale = "BUY"
                    note.append("✅ BUY confermato Moderato")
                else:
                    note.append("🤔 Segnale↑ Debole")


    # ------------------------------------------------------------------
    # SELL logic (con RSI in calo integrato)
    # ------------------------------------------------------------------
    if not SOLO_BUY and (trend_down or recupero_sell) and distanza_ok:
        durata_trend = candele_reali_down
        rsi_in_calo = (rsi < penultimo["RSI"] < antepenultimo["RSI"])

        if (
            rsi <= _p("rsi_sell_forte")
            and macd_sell_ok
            and punteggio_trend <= -SOGLIA_PUNTEGGIO
            and rsi_in_calo
        ):
            if durata_trend >= 15:
                note.append(f"⛔ Trend↓ Maturo ({durata_trend} candele)")
            else:
                segnale = "SELL"
                note.append("✅ SELL confermato")

        elif (
            rsi <= _p("rsi_sell_debole")
            and macd_sell_debole
            and rsi_in_calo
        ):
            if punteggio_trend <= -SOGLIA_PUNTEGGIO - 2 and durata_trend <= 10:
                segnale = "SELL"
                note.append("✅ SELL confermato Moderato")
            else:
                note.append("🤔 Segnale↓ Debole")


    # Pattern V rapido → solo se il trend è coerente (5)
    if (
        sistema == "EMA"
        and segnale == "HOLD"
        and (trend_up or recupero_buy)
        and rsi > 45
        and rileva_pattern_v(hist)
    ):
        segnale = "BUY"
        tp = round(close + atr * 1.5, 4)
        sl = round(close - atr, 4)
        note.append("📈 Pattern V: BUY da inversione rapida")

    # Se segnale BUY/SELL aggiungi meta info
    pattern = riconosci_pattern_candela(hist)
    if segnale in ["BUY", "SELL"]:
        n_candele = candele_reali_up if segnale == "BUY" else candele_reali_down
        dist_level = valuta_distanza(distanza_ema, close)
        note.insert(0, f"📊 {n_candele} Candele | Distanza {dist_level}")

        if segnale == "BUY" and not is_trend_up(ema7, ema25, ema99):
            note.append("↗️ Recupero 7>25 (99 sopra)")
        elif segnale == "SELL" and not is_trend_down(ema7, ema25, ema99):
            note.append("↘️ Recupero 7<25 (99 sotto)")

        if pattern:
            note.append(f"✅ Pattern: {pattern}")
    else:
        if not (trend_up or trend_down):
            note.append("🔍 Nessun segnale: trend indeciso")
        else:
            note.append("⏸️ Criteri incompleti su 15m")


    if segnale == "HOLD":
        if trend_up and candele_trend_up_strict <= 2:
            note.append("🔼 Trend↑ in avvio (strict)")
        if trend_down and candele_trend_down_strict <= 2:
            note.append("🔽 Trend↓ in avvio (strict)")

    


    # Invalidation per pattern contrario
    if pattern_contrario(segnale, pattern):
        note.append(f"⚠️ Pattern contrario: inversione ({pattern})")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # RSI/MACD neutri → solo warning (la penalità è già gestita in calcola_probabilita_successo)
    low, high = _p("macd_rsi_range")
    soglia_macd = _p("macd_signal_threshold")
    if segnale in ["BUY", "SELL"] and low < rsi < high and abs(macd_gap) < soglia_macd:
        note.append(
            f"⚠️ RSI/MACD neutri → prob↓"
        )

        

    # Controllo facoltativo EMA 1m
    if not DISATTIVA_CHECK_EMA_1M:
        n_check_ema = 5 if MODALITA_TEST else 15
        if not ema_in_movimento_coerente(hist_1m, rialzista=(segnale == "BUY"), n_candele=n_check_ema):
            note.append("⛔ Segnale annullato: EMA su 1m non in movimento coerente col trend 15m")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # BUY forzato su incrocio progressivo
    if sistema == "EMA" and segnale == "HOLD" and rileva_incrocio_progressivo(hist):
        segnale = "BUY"
        note.append("📈 Incrocio Progressivo EMA(7>25>99): BUY")

    # Calcolo probabilità di successo (con penalità avanzate)
    try:
        if segnale in ["BUY", "SELL"]:
            n_candele = candele_reali_up if segnale == "BUY" else candele_reali_down
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
                hist=hist,
                tp=tp,
                sl=sl,
                supporto=supporto,
                escursione_media=escursione_media,
                pattern_candela=pattern,
                pattern_db_ok=pattern_db_ok,
                pattern_tri_ok=pattern_tri_ok,
            )
            note.append(f"📊 Prob. successo stimata: {probabilita}%")
    except Exception as e:
        logging.warning(f"⚠️ Errore calcolo probabilità successo: {e}")

    # ------------------------------------------------------------------
    # TP/SL realistico + Probabilità fusa + stima tempi
    # ------------------------------------------------------------------
    if segnale in ["BUY", "SELL"]:
        # Fusione probabilità con punteggio_trend + penalità maturità
        durata = candele_reali_up if segnale == "BUY" else candele_reali_down
        MAT_START = 10
        MAT_SLOPE = 0.25  # dimezza la penalità
        pen_maturita = max(0, durata - MAT_START) * MAT_SLOPE
        punteggio_trend_adj = punteggio_trend - pen_maturita

        prob_fusa = _fuse_prob(punteggio_trend_adj, probabilita)  # 0..1
        if pump_flag:
            prob_fusa = min(1.0, prob_fusa + 0.06)

        # Boost soft solo se si sta realmente usando DB o TRI
        if sistema == "DB" and pattern_db_ok:
            prob_fusa = min(1.0, prob_fusa + PATTERN_SOFT_BOOST_DB * p_db["confidence"])
        if sistema == "TRI" and pattern_tri_ok:
            prob_fusa = min(1.0, prob_fusa + PATTERN_SOFT_BOOST_TRI * p_tri["confidence"])



        # --- BOOST/MALUS da pattern candlestick (opzionale) ---
        if pattern:
            # aggiungi la nota solo se non già presente
            if f"✅ Pattern: {pattern}" not in note:
                note.append(f"✅ Pattern: {pattern}")

        # esempi di aggiustamenti leggeri (solo BUY perché i due pattern sono bullish)
        if segnale == "BUY":
            if pattern in ("Three White Soldiers", "🟩 Three White Soldiers"):
                prob_fusa = min(1.0, prob_fusa + 0.03)   # +3% affidabilità
            elif pattern in ("Bullish Harami", "🤰 Bullish Harami"):
                prob_fusa = min(1.0, prob_fusa + 0.02)   # +2% affidabilità

        # Soft boost su prob_fusa per canale trend

        CHANNEL_SOFT_BOOST = 0.01      # base
        CHANNEL_SOFT_BOOST_BO = 0.015  # extra se breakout del canale

        if chan_15.get('found'):
            if segnale == "BUY" and chan_15.get("type") in ("ascending", "sideways"):
                prob_fusa = min(1.0, prob_fusa + CHANNEL_SOFT_BOOST * chan_15["confidence"])
                if chan_15.get("breakout_confirmed") and chan_15.get("breakout_side") == "up":
                    prob_fusa = min(1.0, prob_fusa + CHANNEL_SOFT_BOOST_BO * chan_15["confidence"])

        

        # applica piccolo boost/malus dal multi-TF del canale
        prob_fusa = max(0.0, min(1.0, prob_fusa + channel_prob_adj))
        if gate_buy_canale and segnale == "BUY" and not pump_flag:
            note.append("⛔ Gate TF1h: canale↓ forte")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

        note.append(f"🧪 Affidabilità: {round(prob_fusa*100)}%")


        # (3) Gate di entrata coerente con prob_fusa
        if breakout_valido or pump_flag:
            P_ENTER = 0.45     # segnali forti
        else:
            P_ENTER = 0.52     # segnali normali più selettivi

        if prob_fusa < P_ENTER:
            note.append(f"⏸️ Gate KO ({prob_fusa:.2f})")
            return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

        # TP/SL proporzionale alla probabilità fusa
        ATR_MIN_FRAC = 0.003
        TP_BASE, TP_SPAN = 0.6, 0.6     # TP più vicino → meno fallimenti immediati
        SL_BASE, SL_SPAN = 1.6, 0.35    # SL meno stretto → meno stop-loss immediati
        RR_MIN = 1.2
        DELTA_MINIMO = 0.1
        TICK = 0.0001  # TODO: sostituire con tick_size reale del symbol (passalo da fuori se puoi)

        atr_eff = atr if atr and atr > 0 else close_s * ATR_MIN_FRAC

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
        half_spread = close_s * (max(spread, 0.0) / 100.0) / 2.0
        if segnale == "BUY":
            tp_raw += half_spread
            sl_raw -= half_spread
        else:
            tp_raw -= half_spread
            sl_raw += half_spread

        # Enforce RR minimo (prima allargo TP)
        rr_note_added = False
        EPS = 1e-9
        risk = abs(close - sl_raw)
        if risk < EPS:
            step = max(TICK, 0.2 * (atr_eff if atr_eff > 0 else close_s * 0.003))
            sl_raw = sl_raw - step if segnale == "BUY" else sl_raw + step
            risk = abs(close - sl_raw)

        reward = abs(tp_raw - close)
        if risk < EPS or _safe_div(reward, risk) < RR_MIN:
            needed_tp = (close + (RR_MIN * risk)) if segnale == "BUY" else (close - (RR_MIN * risk))
            tp_raw = needed_tp
            if not rr_note_added:
                note.append("ℹ️ TP adjusted (RR ≥ {:.1f})".format(RR_MIN))
                rr_note_added = True

        # Stima tempi TP
        # (timeframe 15m => 0.25h per candela)
        T_TARGET_MIN_H = 0.7          # evita target troppo vicino
        T_TARGET_MAX_H = 6.0          # oltre è poco efficiente
        T_HARD_CAP_H   = 12.0         # oltre è poco efficiente ma NON blocchiamo più il segnale
        MAX_TP_SPAN_ATR = 3.0         # non allargare TP oltre 3x ATR

        distanza_tp = abs(tp_raw - close)
        range_medio = (hist["high"] - hist["low"]).iloc[-10:].mean()
        if range_medio <= 0:
            range_medio = atr_eff if atr_eff > 0 else close_s * 0.003

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
            if risk > 0 and _safe_div(reward_cand, risk) >= RR_MIN:
                tp_raw = tp_candidato
                distanza_tp = reward_cand
                ore_min = round((distanza_tp / (range_medio * eff_max)) * 0.25, 2)
                ore_max = round((distanza_tp / (range_medio * eff_min)) * 0.25, 2)
            else:
                note.append("⚠️ TP fermo: RR basso")

        # Se ancora molto lungo, segnala solo warning, ma NON annulla il segnale
        if ore_max > T_HARD_CAP_H:
            note.append(f"⚠️ Tempo TP lungo: {ore_min}–{ore_max}h (cap {T_HARD_CAP_H}h)")

        # Se troppo veloce, allarga TP (entro un limite) preservando RR
        if ore_min < T_TARGET_MIN_H:
            dist_attuale_atr = _safe_div(distanza_tp, atr_eff if atr_eff > 0 else 1)
            if dist_attuale_atr < MAX_TP_SPAN_ATR:
                scala_up = T_TARGET_MIN_H / max(ore_min, 0.05)
                nuova_dist = min(distanza_tp * scala_up, MAX_TP_SPAN_ATR * atr_eff)
                tp_candidato = (close + nuova_dist) if segnale == "BUY" else (close - nuova_dist)
                reward_cand = abs(tp_candidato - close)
                if risk > 0 and _safe_div(reward_cand, risk) >= RR_MIN:
                    tp_raw = tp_candidato
                    distanza_tp = reward_cand
                    ore_min = round((distanza_tp / (range_medio * eff_max)) * 0.25, 2)
                    ore_max = round((distanza_tp / (range_medio * eff_min)) * 0.25, 2)

        # Nota tempi finali
        note.append(f"⏱️ Target TP: ~{ore_min}–{ore_max}h")

        # Se c’è pump, mostra la nota immediatamente dopo la stima tempi
        if pump_flag and pump_msg:
            note.append(pump_msg)

        # Rounding a tick e safety finale
        def _round_tick(x, tick=TICK):
            if tick and tick > 0:
                return round(round(x / tick) * tick, 10)
            return round(x, 10)

        tp = _round_tick(tp_raw)
        sl = _round_tick(sl_raw)

        if segnale == "BUY" and not (sl < close < tp):
            note.append("⚠️ TP/SL BUY incoerenti")
        if segnale == "SELL" and not (tp < close < sl):
            note.append("⚠️ TP/SL SELL incoerenti")

    logging.debug("✅ Analisi completata")

    if segnale not in ["BUY", "SELL"]:
        if not note:
            note.append("ℹ️ Nessun criterio abbastanza forte per un segnale operativo")
        return "HOLD", hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto

    # Ordina note: conferme → punteggi/tempi/eventi → warning → altro
    priorita = lambda x: (
        0 if "✅" in x else
        1 if ("📊" in x or "⏱️" in x or "⚡" in x or "🚀" in x or "🚨" in x or "💥" in x) else
        2 if ("⚠️" in x or "⛔" in x or "ℹ️" in x) else
        3
    )

    note = list(dict.fromkeys(note))  # Elimina duplicati
    note.sort(key=priorita)

    return segnale, hist, distanza_ema, "\n".join(note).strip(), tp, sl, supporto
