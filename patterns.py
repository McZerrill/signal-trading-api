# patterns.py
from typing import Any, Dict, List
import numpy as np
import pandas as pd


# ---------- Utils ----------

def _pct(a: float, b: float) -> float:
    if b is None or b == 0:
        return 0.0
    return (a - b) / b


def _rolling_argrelextrema(
    series: pd.Series, order: int = 3, mode: str = "min"
) -> List[Any]:
    """Trova min/max locali con finestra 2*order+1 (solo estremi univoci)."""
    if len(series) < (2 * order + 1):
        return []
    idx: List[Any] = []
    vals = series.values
    for i in range(order, len(series) - order):
        win = vals[i - order : i + order + 1]
        c = vals[i]
        if mode == "min" and c == win.min() and (win < c + 1e-12).sum() == 1:
            idx.append(series.index[i])
        if mode == "max" and c == win.max() and (win > c - 1e-12).sum() == 1:
            idx.append(series.index[i])
    return idx


def _level_touch_ratio(series: pd.Series, level: float, tol_frac: float) -> float:
    tol = abs(level) * tol_frac
    hits = ((series - level).abs() <= tol).sum()
    return hits / max(1, len(series))


# ------------------------------
# 1) Double Bottom
# ------------------------------

def detect_double_bottom(
    df: pd.DataFrame,
    lookback: int = 180,
    min_separation: int = 5,
    bottoms_tol: float = 0.02,
    neckline_confirm: bool = True,
) -> Dict[str, Any]:
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

    lows = d["low"]
    closes = d["close"]
    mins = _rolling_argrelextrema(lows, order=3, mode="min")
    if len(mins) < 2:
        return {
            "found": False,
            "pattern": "Double Bottom",
            "confidence": 0.0,
            "last_index": d.index[-1],
        }

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
            "points": {"bottom1_idx": i1, "bottom2_idx": i2},
            "levels": {
                "neckline": neckline,
                "bottom_avg": float((p1 + p2) / 2.0),
            },
            "confidence": conf,
            "neckline_confirmed": breakout_now,
            "neckline_breakout": breakout_now,  # alias per compatibilità
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

def detect_ascending_triangle(
    df: pd.DataFrame,
    lookback: int = 200,
    touch_tol: float = 0.015,
    min_touches: int = 2,
    breakout_confirm: bool = True,
) -> Dict[str, Any]:
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

    highs = d["high"]
    lows = d["low"]
    closes = d["close"]

    # resistenza piatta (tipicamente top recente)
    res = float(highs.quantile(0.90))
    hits = _level_touch_ratio(highs, res, touch_tol)
    if hits < (min_touches / max(5, len(d))):
        return {
            "found": False,
            "pattern": "Ascending Triangle",
            "confidence": 0.0,
            "last_index": d.index[-1],
        }

    # base ascendente (minimi crescenti via trendline)
    x = np.arange(len(lows))
    slope = np.polyfit(x, lows.values, 1)[0]
    if slope <= 0:
        return {
            "found": False,
            "pattern": "Ascending Triangle",
            "confidence": 0.0,
            "last_index": d.index[-1],
        }

    breakout_now = bool(closes.iloc[-1] > res)
    if breakout_confirm and not breakout_now:
        return {
            "found": False,
            "pattern": "Ascending Triangle",
            "confidence": 0.0,
            "last_index": d.index[-1],
        }

    brk = max(0.0, _pct(closes.iloc[-1], res))
    # confidenza: breakout% (60%) + pendenza base normalizzata (40%)
    denom = max(res * 0.002, 1e-12)  # protezione da divisione per zero
    conf = float(np.clip(min(brk / 0.02, 1.0) * 0.6 + min(slope / denom, 1.0) * 0.4, 0, 1))

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


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ritorna slope, intercept, r^2."""
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
    parallel_tolerance: float = 0.20,   # max 20% di differenza relativa tra pendenze
    touch_tol_mult_atr: float = 0.55,   # tolleranza tocchi = 0.55 * ATR (fallback su ampiezza/200)
    min_confidence: float = 0.55,
    require_volume_for_breakout: bool = True,
    breakout_vol_mult: float = 1.30,
) -> Dict[str, Any]:
    """
    Rileva un canale di prezzo (ascending/descending/sideways) su 'lookback' barre.

    Ritorna un dict:
      - found: bool
      - type: 'ascending' | 'descending' | 'sideways' | ''
      - confidence: float 0..1
      - slope_upper, slope_lower, intercept_upper, intercept_lower: float
      - width_now: ampiezza canale sull'ultima barra
      - touches_upper, touches_lower: num tocchi entro tolleranza
      - points: {'upper_idx': [Index|int], 'lower_idx': [Index|int]}
      - breakout_confirmed: bool
      - breakout_side: 'up' | 'down' | ''
    """
    out: Dict[str, Any] = {
        "found": False,
        "type": "",
        "confidence": 0.0,
        "slope_upper": 0.0,
        "slope_lower": 0.0,
        "intercept_upper": 0.0,
        "intercept_lower": 0.0,
        "width_now": 0.0,
        "touches_upper": 0,
        "touches_lower": 0,
        "points": {"upper_idx": [], "lower_idx": []},
        "breakout_confirmed": False,
        "breakout_side": "",
    }

    # Dati minimi
    if hist is None or len(hist) < 30:
        return out
    req_cols = {"high", "low", "close"}
    if not req_cols.issubset(hist.columns):
        return out

    # Usa anche se len(df) < lookback
    df = hist.tail(lookback).copy()
    n = len(df)
    x = np.arange(n).astype(float)

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)

    # Fit lineare upper/lower + R^2
    m_u, q_u, r2_u = _fit_line(x, highs)
    m_l, q_l, r2_l = _fit_line(x, lows)

    # Linee e valori correnti
    upper_line = m_u * x + q_u
    lower_line = m_l * x + q_l
    upper_now = float(upper_line[-1])
    lower_now = float(lower_line[-1])

    # Se per numerica capita lower > upper, normalizza i lati
    if lower_now > upper_now:
        m_u, m_l = m_l, m_u
        q_u, q_l = q_l, q_u
        upper_line, lower_line = lower_line, upper_line
        upper_now, lower_now = lower_now, upper_now
        r2_u, r2_l = r2_l, r2_u

    width_now = max(upper_now - lower_now, 1e-12)

    # Parallelismo (relativo)
    denom = max(abs(m_u), abs(m_l), 1e-12)
    par_err = abs(m_u - m_l) / denom
    parallel_ok = par_err <= parallel_tolerance

    # Tolleranza tocchi: ATR o fallback su ampiezza canale
    if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]):
        atr_now = float(df["ATR"].iloc[-1])
    else:
        atr_now = 0.0
    touch_tol = (touch_tol_mult_atr * atr_now) if atr_now > 0 else (width_now / 200.0)

    # Conta tocchi su entrambi i lati (distanza verticale entro tolleranza)
    dist_up = np.abs(highs - upper_line)
    dist_lo = np.abs(lows - lower_line)
    up_mask = dist_up <= touch_tol
    lo_mask = dist_lo <= touch_tol

    up_idx = df.index.to_numpy()[up_mask].tolist()
    lo_idx = df.index.to_numpy()[lo_mask].tolist()
    touches_up = len(up_idx)
    touches_lo = len(lo_idx)

    # Tipo canale
    if m_u > 0 and m_l > 0:
        ch_type = "ascending"
    elif m_u < 0 and m_l < 0:
        ch_type = "descending"
    else:
        ch_type = "sideways"

    # Qualità del fit / copertura tocchi
    r2_score = max(0.0, (r2_u + r2_l) / 2.0)
    touches_score = min((touches_up + touches_lo) / (2.0 * max(1, min_touches_side)), 1.0)
    coverage_score = 1.0 if (touches_up >= min_touches_side and touches_lo >= min_touches_side) else 0.0
    par_score = max(0.0, 1.0 - min(par_err / max(parallel_tolerance, 1e-12), 1.0))

    confidence = 0.35 * par_score + 0.35 * r2_score + 0.20 * touches_score + 0.10 * coverage_score
    confidence = max(0.0, min(confidence, 1.0))

    # Filtro qualità minima
    if not parallel_ok or confidence < min_confidence:
        return out

    # Breakout sull'ultima candela (con filtro volume opzionale)
    close_last = float(df["close"].iloc[-1])
    high_last = float(df["high"].iloc[-1])
    low_last = float(df["low"].iloc[-1])
    if "volume" in df.columns and len(df) > 1:
        vol_last = float(df["volume"].iloc[-1])
        vol_mean = float(df["volume"].iloc[:-1].mean())
    else:
        vol_last, vol_mean = 0.0, 0.0

    bo_tol = (0.3 * atr_now) if atr_now > 0 else (width_now * 0.08)
    upper_edge = upper_now + bo_tol
    lower_edge = lower_now - bo_tol

    breakout_up = (close_last > upper_edge) or (high_last > upper_edge)
    breakout_down = (close_last < lower_edge) or (low_last < lower_edge)

    if require_volume_for_breakout and vol_mean > 0:
        breakout_up = breakout_up and (vol_last > vol_mean * breakout_vol_mult)
        breakout_down = breakout_down and (vol_last > vol_mean * breakout_vol_mult)

    return {
        "found": True,
        "type": ch_type,
        "confidence": confidence,
        "slope_upper": float(m_u),
        "slope_lower": float(m_l),
        "intercept_upper": float(q_u),
        "intercept_lower": float(q_l),
        "width_now": float(width_now),
        "touches_upper": int(touches_up),
        "touches_lower": int(touches_lo),
        "points": {"upper_idx": up_idx, "lower_idx": lo_idx},
        "breakout_confirmed": bool(breakout_up or breakout_down),
        "breakout_side": "up" if breakout_up else ("down" if breakout_down else ""),
    }
