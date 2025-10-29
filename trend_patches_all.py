# trend_patches_all.py
# ------------------------------------------------------------
# Patch modulari per il backend "Segnali di Borsa"
# - structural_pattern_adjust
# - apply_1h_micro_adjust
# - flat_market_penalty
# - hysteresis_adjust
# - channel_defense_adjust
# - util: compute_ema_slopes, post_fusion_pipeline
# ------------------------------------------------------------
from collections import deque
from typing import Optional, Tuple, Dict, Any, Iterable
import logging

# --------------------------------
# Utils
# --------------------------------
def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def compute_ema_slopes(hist, lookback: int = 6) -> Dict[str, float]:
    """
    Restituisce slope semplici delle EMA in unità 'prezzo per candela':
    { 'ema7': m7, 'ema25': m25, 'ema99': m99 }
    """
    if hist is None or len(hist) < lookback + 1:
        return {"ema7": 0.0, "ema25": 0.0, "ema99": 0.0}
    e7 = hist["EMA_7"].iloc[-lookback-1:]
    e25 = hist["EMA_25"].iloc[-lookback-1:]
    e99 = hist["EMA_99"].iloc[-lookback-1:]
    m7 = (e7.iloc[-1] - e7.iloc[0]) / max(lookback, 1)
    m25 = (e25.iloc[-1] - e25.iloc[0]) / max(lookback, 1)
    m99 = (e99.iloc[-1] - e99.iloc[0]) / max(lookback, 1)
    return {"ema7": float(m7), "ema25": float(m25), "ema99": float(m99)}

# --------------------------------
# 1) Structural pattern adjust (15m + 1h, ATR/Volume aware)
# --------------------------------
def structural_pattern_adjust(prob_fusa: float,
                              segnale: str,
                              patt_15m: Optional[Dict[str, Any]],
                              patt_1h: Optional[Dict[str, Any]],
                              atr_pct: float,
                              vol_ratio: float,
                              near_breakout: bool,
                              log: Optional[list] = None) -> float:
    """
    - patt_X: dict con chiavi tipiche: {"found": bool, "name": str, "confidence": 0..1, "direction": "BUY"/"SELL"}
    - atr_pct: ATR/close (es. 0.01 = 1%)
    - vol_ratio: volume_attuale / volume_medio (1.0 = in media)
    - near_breakout: se prezzo è vicino al livello chiave del pattern
    """
    base = _clamp01(prob_fusa)
    boost = 0.0

    # Pesi adattivi
    atr_weight = 1.0
    if atr_pct >= 0.015:
        atr_weight = 0.70
    elif atr_pct >= 0.012:
        atr_weight = 0.85

    vol_weight = 1.0
    if vol_ratio >= 1.5:
        vol_weight = 1.15
    elif vol_ratio >= 1.2:
        vol_weight = 1.07

    def _dir_from_name(nm: str) -> Optional[str]:
        nm = (nm or "").lower()
        if nm in ("double_bottom", "ascending_triangle"): return "BUY"
        if nm in ("double_top", "descending_triangle"):   return "SELL"
        return None

    def _pattern_boost(p: Optional[Dict[str, Any]], src: str) -> float:
        if not p or not p.get("found"):
            return 0.0
        implied = p.get("direction") or _dir_from_name(p.get("name"))
        if not implied:
            return 0.0
        conf = _clamp01(p.get("confidence", 0.0))
        w = 0.018 if src == "15m" else 0.012  # 15m più forte di 1h nel micro-boost (1h rifinisce col canale)
        if near_breakout and implied == segnale:
            w *= 1.4
        sgn = +1 if implied == segnale else -1
        return sgn * w * conf

    boost += _pattern_boost(patt_15m, "15m")
    boost += _pattern_boost(patt_1h,  "1h")

    boost *= atr_weight * vol_weight
    newp = _clamp01(base + boost)

    if log is not None:
        log.append(f"[STRUCT] boost={boost:+.3f} → p={newp:.3f} (atr_w={atr_weight:.2f}, vol_w={vol_weight:.2f}, near_br={near_breakout})")

    return newp

# --------------------------------
# 2) Micro-adjust 1h (canale + pattern 1h)
# --------------------------------
def apply_1h_micro_adjust(prob_fusa: float,
                          segnale: str,
                          chan_1h: Optional[Dict[str, Any]],
                          patt_1h: Optional[Dict[str, Any]] = None,
                          log: Optional[list] = None) -> float:
    """
    Applica un micro-adjust su 1h in base a:
    - canale 1h (ascending/descending + confidence)
    - pattern 1h direzionale (BUY/SELL)
    """
    base = _clamp01(prob_fusa)
    adj_total = 0.0

    if chan_1h and chan_1h.get("found"):
        t1h = (chan_1h.get("type") or "").lower()
        c1h = _clamp01(chan_1h.get("confidence", 0.0))
        adj = 0.0
        if segnale == "BUY":
            if t1h == "ascending":   adj = +0.02 * c1h
            elif t1h == "descending":adj = -0.02 * c1h
        elif segnale == "SELL":
            if t1h == "descending":  adj = +0.02 * c1h
            elif t1h == "ascending": adj = -0.02 * c1h
        adj_total += adj

    if patt_1h and patt_1h.get("found"):
        dir1h = (patt_1h.get("direction") or "").upper()
        cp1h  = _clamp01(patt_1h.get("confidence", 0.0))
        if dir1h == segnale:
            adj_total += +0.015 * cp1h
        elif dir1h in ("BUY","SELL"):
            adj_total += -0.015 * cp1h

    newp = _clamp01(base + adj_total)
    if log is not None:
        t1desc = f"{chan_1h.get('type')}/{_safe_float(chan_1h.get('confidence'),0):.2f}" if (chan_1h and chan_1h.get("found")) else "none"
        p1desc = f"{(patt_1h or {}).get('direction')}/{_safe_float((patt_1h or {}).get('confidence'),0):.2f}" if (patt_1h and patt_1h.get("found")) else "none"
        log.append(f"[1H] adj_total={adj_total:+.3f} chan={t1desc} patt={p1desc} → p={newp:.3f}")
    return newp

# --------------------------------
# 3) Flat market penalty (EMA piatte + bassa volatilità)
# --------------------------------
def flat_market_penalty(prob_fusa: float,
                        ema_slopes: Dict[str, float],
                        atr_pct: float,
                        log: Optional[list] = None) -> float:
    """
    Penalizza lievemente mercati piatti/stretti:
    - EMA7/25/99 con slope molto bassa
    - ATR% basso
    """
    base = _clamp01(prob_fusa)
    m7  = abs(_safe_float(ema_slopes.get("ema7"), 0.0))
    m25 = abs(_safe_float(ema_slopes.get("ema25"), 0.0))
    m99 = abs(_safe_float(ema_slopes.get("ema99"), 0.0))

    # soglie conservative (in unità prezzo per candela), e ATR% basso
    flat = (m7 < 0.0005 and m25 < 0.0003 and m99 < 0.0002) and (atr_pct < 0.008)
    newp = base
    if flat:
        newp = max(0.0, base - 0.03)
        if log is not None:
            log.append(f"[FLAT] penalty 3% (m7={m7:.5f}, m25={m25:.5f}, m99={m99:.5f}, atr%={atr_pct:.4f}) → p={newp:.3f}")
    else:
        if log is not None:
            log.append(f"[FLAT] no penalty (m7={m7:.5f}, m25={m25:.5f}, m99={m99:.5f}, atr%={atr_pct:.4f})")
    return newp

# --------------------------------
# 4) Hysteresis anti-flip (micro-memoria recentissima)
# --------------------------------
def hysteresis_adjust(raw_signal: str,
                      prob: float,
                      buy_thr: float,
                      sell_thr: float,
                      recent_signals: Optional[deque] = None,
                      margin: float = 0.03,
                      log: Optional[list] = None) -> Tuple[str, float, deque]:
    """
    Evita flip rapidi BUY↔SELL quando la probabilità è vicino alle soglie.
    - buy_thr/sell_thr in [0..1]
    - margin: extra margine richiesto per invertire
    """
    dq = recent_signals or deque(maxlen=3)
    if not dq:
        dq.append(raw_signal)
        return raw_signal, prob, dq

    last = dq[-1]
    final = raw_signal
    # se si tenta un'inversione diretta, chiedi margine extra
    if last in ("BUY","SELL") and raw_signal in ("BUY","SELL") and last != raw_signal:
        if raw_signal == "BUY" and prob < (buy_thr + margin):
            final = "HOLD"
        elif raw_signal == "SELL" and prob > (sell_thr - margin):
            final = "HOLD"

    dq.append(final)
    if log is not None:
        log.append(f"[HYST] last={last} raw={raw_signal} final={final} prob={prob:.3f} (thrB={buy_thr:.2f}, thrS={sell_thr:.2f}, m={margin:.2f})")
    return final, prob, dq

# --------------------------------
# 5) Channel defense adjust (gestione trailing/uscita prudente)
# --------------------------------
def channel_defense_adjust(state: Dict[str, Any],
                           indicators: Dict[str, Any],
                           ch15: Optional[Dict[str, Any]],
                           log: Optional[list] = None) -> Dict[str, Any]:
    """
    Se il prezzo rientra nel canale 15m e tocca il bordo opposto con momentum debole,
    stringe il trailing/TP.
    - state: dict della posizione/simulazione (mutato e restituito)
    - indicators: include 'last_price', 'rsi_15m', 'macd_hist_15m'
    - ch15: {"found": bool, "type": "...", "upper": float, "lower": float}
    """
    if not ch15 or not ch15.get("found"):
        return state

    price = indicators.get("last_price")
    rsi   = indicators.get("rsi_15m")
    macdh = indicators.get("macd_hist_15m")
    if price is None:
        return state

    upper = ch15.get("upper")
    lower = ch15.get("lower")

    touching_upper = (upper is not None) and (price >= float(upper) * 0.998)
    touching_lower = (lower is not None) and (price <= float(lower) * 1.002)

    weak_momentum = ((macdh is not None and _safe_float(macdh) < 0) or
                     (rsi is not None and 45 <= _safe_float(rsi) <= 55))

    if weak_momentum and (touching_upper or touching_lower):
        prev = float(state.get("trailing_tp_step", 0.0) or 0.0)
        newv = max(prev, 0.003)  # +0.3% minimo
        state["trailing_tp_step"] = newv
        note = state.get("note", "")
        if "Channel defense" not in note:
            state["note"] = (note + " | Channel defense: momentum weak at border, tightened trailing.").strip(" |")
        if log is not None:
            log.append(f"[CHDEF] weak@border → trailing_tp_step {prev:.4f}→{newv:.4f}")
    return state

# --------------------------------
# 6) Pipeline consigliata post-fusione
# --------------------------------
def post_fusion_pipeline(prob_fusa: float,
                         segnale: str,
                         chan_15: Optional[Dict[str, Any]],
                         chan_1h: Optional[Dict[str, Any]],
                         patt_15m: Optional[Dict[str, Any]],
                         patt_1h: Optional[Dict[str, Any]],
                         atr_pct: float,
                         vol_ratio: float,
                         ema_slopes: Dict[str, float],
                         log: Optional[list] = None) -> float:
    """
    Applica in ordine:
    1) structural_pattern_adjust (prima rifinitura)
    2) apply_1h_micro_adjust (coerenza canale/pattern su 1h)
    3) flat_market_penalty (mercato piatto)
    near_breakout è inferito da canale 15m o flag di breakout su pattern.
    """
    # near_breakout: true se breakout confermato sul canale 15m o pattern 15m
    near_br = False
    if chan_15 and chan_15.get("found"):
        if chan_15.get("breakout_confirmed"):
            near_br = True
    if patt_15m and patt_15m.get("found"):
        # qualsiasi hint di breakout/neckline fa scattare near_breakout
        if patt_15m.get("breakout_confirmed") or patt_15m.get("neckline_confirmed") or patt_15m.get("neckline_breakout"):
            near_br = True

    p = _clamp01(prob_fusa)

    # 1) struttura
    p = structural_pattern_adjust(p, segnale, patt_15m, patt_1h, atr_pct, vol_ratio, near_br, log=log)
    # 2) 1h micro adjust
    p = apply_1h_micro_adjust(p, segnale, chan_1h, patt_1h, log=log)
    # 3) flat penalty
    p = flat_market_penalty(p, ema_slopes, atr_pct, log=log)

    if log is not None:
        log.append(f"[PIPE] post-fusion → p={p:.3f}")
    return p
