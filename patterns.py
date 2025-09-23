# patterns.py
import pandas as pd
import numpy as np


# ---------- Utils ----------

def _pct(a, b):
    if b is None or b == 0:
        return 0.0
    return (a - b) / b

def _rolling_argrelextrema(series: pd.Series, order: int = 3, mode: str = "min"):
    """Trova min/max locali con finestra 2*order+1 (solo estremi univoci)."""
    if len(series) < (2 * order + 1):
        return []
    idx = []
    vals = series.values
    for i in range(order, len(series) - order):
        win = vals[i - order:i + order + 1]
        c = vals[i]
        if mode == "min" and c == win.min() and (win < c + 1e-12).sum() == 1:
            idx.append(series.index[i])
        if mode == "max" and c == win.max() and (win > c - 1e-12).sum() == 1:
            idx.append(series.index[i])
    return idx

def _level_touch_ratio(series: pd.Series, level: float, tol_frac: float):
    tol = abs(level) * tol_frac
    hits = ((series - level).abs() <= tol).sum()
    return hits / max(1, len(series))


# ------------------------------
# 1) Double Bottom
# ------------------------------
def detect_double_bottom(df: pd.DataFrame,
                         lookback: int = 180,
                         min_separation: int = 5,
                         bottoms_tol: float = 0.02,
                         neckline_confirm: bool = True):
    """
    Ritorna un dict con almeno:
      - found: bool
      - pattern: "Double Bottom"
      - confidence: float (0..1)
      - points: {bottom1_idx, bottom2_idx}
      - levels: {neckline, bottom_avg}
      - neckline_confirmed (alias: neckline_breakout): bool
      - breakout_index, last_index: indice dell'ultima barra considerata
    """
    d = df.tail(lookback).copy()
    if len(d) < 10 or not {"low", "high", "close"}.issubset(d.columns):
        return {"found": False, "pattern": "Double Bottom", "confidence": 0.0}

    lows = d["low"]; closes = d["close"]
    mins = _rolling_argrelextrema(lows, order=3, mode="min")
    if len(mins) < 2:
        return {"found": False, "pattern": "Double Bottom", "confidence": 0.0, "last_index": d.index[-1]}

    for i in range(len(mins) - 1):
        i1, i2 = mins[i], mins[i + 1]
        p1, p2 = lows.loc[i1], lows.loc[i2]

        # separazione temporale minima
        if (d.index.get_loc(i2) - d.index.get_loc(i1)) < min_separation:
            continue

        # simmetria/altezza compatibili
        if abs(_pct(p1, p2)) > bottoms_tol:
            continue

        seg = d.loc[i1:i2]
        neckline = float(seg["high"].max())
        breakout_now = bool(closes.iloc[-1] > neckline)

        if neckline_confirm and not breakout_now:
            # niente conferma → continua a cercare altri doppi minimi
            continue

        # confidenza: media pesata di simmetria e breakout %
        sym = 1 - min(1.0, abs(_pct(p1, p2)) / bottoms_tol)
        brk = max(0.0, _pct(closes.iloc[-1], neckline))
        conf = float(np.clip(0.5 * sym + 0.5 * min(brk / 0.02, 1.0), 0, 1))

        return {
            "found": True,
            "pattern": "Double Bottom",
            "points": {
                "bottom1_idx": i1,
                "bottom2_idx": i2,
            },
            "levels": {
                "neckline": neckline,
                "bottom_avg": float((p1 + p2) / 2.0),
            },
            "confidence": conf,
            "neckline_confirmed": breakout_now,
            "neckline_breakout": breakout_now,      # alias per compatibilità
            "breakout_index": d.index[-1] if breakout_now else None,
            "last_index": d.index[-1],
            "note": f"Neckline {neckline:.6f} — simmetria {sym:.2f}, breakout {brk:.2%}",
        }

    # nessun DB valido trovato
    return {
        "found": False,
        "pattern": "Double Bottom",
        "confidence": 0.0,
        "last_index": d.index[-1],
    }


# -----------------------------------------
# 2) Ascending Triangle (bullish)
# -----------------------------------------
def detect_ascending_triangle(df: pd.DataFrame,
                              lookback: int = 200,
                              touch_tol: float = 0.015,
                              min_touches: int = 2,
                              breakout_confirm: bool = True):
    """
    Ritorna un dict con almeno:
      - found: bool
      - pattern: "Ascending Triangle"
      - confidence: float (0..1)
      - levels: {resistance}
      - breakout_confirmed: bool
      - breakout_index, last_index
    """
    d = df.tail(lookback).copy()
    if len(d) < 10 or not {"high", "low", "close"}.issubset(d.columns):
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0}

    highs = d["high"]; lows = d["low"]; closes = d["close"]

    # resistenza piatta (tipicamente top recente)
    res = float(highs.quantile(0.90))
    hits = _level_touch_ratio(highs, res, touch_tol)
    if hits < (min_touches / max(5, len(d))):
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    # base ascendente (minimi crescenti via trendline)
    x = np.arange(len(lows))
    slope = np.polyfit(x, lows.values, 1)[0]
    if slope <= 0:
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    breakout_now = bool(closes.iloc[-1] > res)
    if breakout_confirm and not breakout_now:
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    brk = max(0.0, _pct(closes.iloc[-1], res))
    # confidenza: breakout% (60%) + pendenza base normalizzata (40%)
    conf = float(np.clip(min(brk / 0.02, 1.0) * 0.6 + min(slope / (res * 0.002), 1.0) * 0.4, 0, 1))

    return {
        "found": True,
        "pattern": "Ascending Triangle",
        "points": {},
        "levels": {"resistance": res},
        "confidence": conf,
        "breakout_confirmed": breakout_now,
        "breakout_index": d.index[-1] if breakout_now else None,
        "last_index": d.index[-1],
        "note": f"Resistenza ~{res:.6f}, breakout {brk:.2%}, slope lows {slope:.6g}",
    }

def _fit_line(x: np.ndarray, y: np.ndarray):
    """Ritorna slope, intercept, r2"""
    if len(x) < 2 or len(y) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0, 0.0
    m, q = np.polyfit(x, y, 1)
    yhat = m * x + q
    # R^2 robusto
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(m), float(q), float(max(min(r2, 1.0), 0.0))

def detect_price_channel(
    hist: pd.DataFrame,
    lookback: int = 200,
    min_touches_side: int = 2,
    parallel_tolerance: float = 0.20,    # max 20% di differenza tra pendenze
    touch_tol_mult_atr: float = 0.55,    # tolleranza = 0.55 * ATR (fallback su ampiezza/200)
    min_confidence: float = 0.55,
    require_volume_for_breakout: bool = True,
    breakout_vol_mult: float = 1.30,
):
    """
    Rileva canale di prezzo (ascending/descending/sideways).
    Ritorna dict con chiavi:
      - found (bool), type ('ascending'|'descending'|'sideways'|''), confidence (0..1)
      - slope_upper, slope_lower, intercept_upper, intercept_lower
      - width_now, touches_upper, touches_lower, points (indici di tocchi)
      - breakout_confirmed (bool), breakout_side ('up'|'down'|'')
    """
    out = {
        "found": False, "type": "", "confidence": 0.0,
        "slope_upper": 0.0, "slope_lower": 0.0,
        "intercept_upper": 0.0, "intercept_lower": 0.0,
        "width_now": 0.0, "touches_upper": 0, "touches_lower": 0,
        "points": {"upper_idx": [], "lower_idx": []},
        "breakout_confirmed": False, "breakout_side": ""
    }

    if hist is None or len(hist) < max(lookback, 30):
        return out

    df = hist.tail(lookback).copy()
    n = len(df)
    x = np.arange(n).astype(float)

    # Serie per fit
    highs = df["high"].values.astype(float)
    lows  = df["low"].values.astype(float)

    m_u, q_u, r2_u = _fit_line(x, highs)
    m_l, q_l, r2_l = _fit_line(x, lows)

    # Parallelismo (relativo)
    denom = max(abs(m_u), abs(m_l), 1e-12)
    par_err = abs(m_u - m_l) / denom
    parallel_ok = par_err <= parallel_tolerance

    # Ampiezza canale "istantanea"
    upper_now = m_u * (n - 1) + q_u
    lower_now = m_l * (n - 1) + q_l
    width_now = max(upper_now - lower_now, 1e-12)

    # Tolleranza tocchi: ATR o fallback su ampiezza canale
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else 0.0
    touch_tol = touch_tol_mult_atr * atr_now if atr_now > 0 else width_now / 200.0

    # Conta tocchi (distanza verticale entro tolleranza)
    upper_line = m_u * x + q_u
    lower_line = m_l * x + q_l
    dist_up = np.abs(highs - upper_line)
    dist_lo = np.abs(lows  - lower_line)

    up_idx = np.where(dist_up <= touch_tol)[0].tolist()
    lo_idx = np.where(dist_lo <= touch_tol)[0].tolist()

    touches_up = len(up_idx)
    touches_lo = len(lo_idx)

    # Direzione canale
    ch_type = ""
    if m_u > 0 and m_l > 0:
        ch_type = "ascending"
    elif m_u < 0 and m_l < 0:
        ch_type = "descending"
    else:
        ch_type = "sideways"

    # Qualità del fit
    r2_score = max(0.0, (r2_u + r2_l) / 2.0)

    # Copertura tocchi su entrambi i lati
    touches_ok = (touches_up >= min_touches_side) and (touches_lo >= min_touches_side)

    # Confidence (pesi semplici: parallellismo, r2, tocchi, copertura)
    # clamp in [0..1]
    par_score = max(0.0, 1.0 - min(par_err / parallel_tolerance, 1.0))
    touches_score = min((touches_up + touches_lo) / (2.0 * min_touches_side), 1.0)
    coverage_score = 1.0 if touches_ok else 0.0
    confidence = 0.35 * par_score + 0.35 * r2_score + 0.20 * touches_score + 0.10 * coverage_score
    confidence = max(0.0, min(confidence, 1.0))

    # Se non è parallelo o non abbastanza toccato, abort
    if not parallel_ok or confidence < min_confidence:
        return out

    # Breakout su ultima candela
    close_last = float(df["close"].iloc[-1])
    high_last  = float(df["high"].iloc[-1])
    low_last   = float(df["low"].iloc[-1])
    vol_last   = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0
    vol_mean   = float(df["volume"].iloc[:-1].mean()) if "volume" in df.columns and len(df) > 1 else 0.0

    upper_last = upper_now
    lower_last = lower_now

    # tolleranza breakout: 0.3*ATR o 8% ampiezza canale (fallback)
    bo_tol = 0.3 * atr_now if atr_now > 0 else width_now * 0.08

    breakout_up   = (close_last > upper_last + bo_tol) or (high_last > upper_last + bo_tol)
    breakout_down = (close_last < lower_last - bo_tol) or (low_last  < lower_last - bo_tol)

    if require_volume_for_breakout and vol_mean > 0:
        breakout_up   = breakout_up   and (vol_last > vol_mean * breakout_vol_mult)
        breakout_down = breakout_down and (vol_last > vol_mean * breakout_vol_mult)

    out.update({
        "found": True,
        "type": ch_type,
        "confidence": confidence,
        "slope_upper": m_u, "slope_lower": m_l,
        "intercept_upper": q_u, "intercept_lower": q_l,
        "width_now": width_now,
        "touches_upper": touches_up, "touches_lower": touches_lo,
        "points": {"upper_idx": up_idx, "lower_idx": lo_idx},
        "breakout_confirmed": bool(breakout_up or breakout_down),
        "breakout_side": "up" if breakout_up else ("down" if breakout_down else "")
    })
    return out
