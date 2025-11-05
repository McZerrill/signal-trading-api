# patterns.py
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# =========================================================
# Utils
# =========================================================

def _pct(a: float, b: float) -> float:
    if b is None or b == 0:
        return 0.0
    return (a - b) / b


def _rolling_argrelextrema(
    series: pd.Series, order: int = 3, mode: str = "min"
) -> List[Any]:
    """Trova min/max locali con finestra 2*order+1 (estremi univoci)."""
    if series is None or len(series) < (2 * order + 1):
        return []
    s = series.dropna()
    if len(s) < (2 * order + 1):
        return []
    idx: List[Any] = []
    vals = s.values
    for i in range(order, len(s) - order):
        win = vals[i - order : i + order + 1]
        c = vals[i]
        if mode == "min" and c == win.min() and (win < c + 1e-12).sum() == 1:
            idx.append(s.index[i])
        if mode == "max" and c == win.max() and (win > c - 1e-12).sum() == 1:
            idx.append(s.index[i])
    return idx


def _level_touch_ratio(series: pd.Series, level: float, tol_frac: float) -> float:
    if series is None or len(series) == 0:
        return 0.0
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    tol = abs(level) * tol_frac
    hits = ((s - level).abs() <= tol).sum()
    return hits / max(1, len(s))


# =========================================================
# 1) Double Bottom (bullish)
# =========================================================

def detect_double_bottom(
    df: pd.DataFrame,
    lookback: int = 180,
    min_separation: int = 5,
    bottoms_tol: float = 0.02,
    neckline_confirm: bool = True,
) -> Dict[str, Any]:
    """
    Ritorna un dict con:
      found, pattern, confidence, points, levels, neckline_confirmed/neckline_breakout,
      breakout_index, last_index, note
    """
    d = df.tail(lookback).copy()
    req = {"low", "high", "close"}
    if len(d) < 10 or not req.issubset(d.columns):
        return {"found": False, "pattern": "Double Bottom", "confidence": 0.0}

    d = d.dropna(subset=list(req))
    if len(d) < 10:
        return {"found": False, "pattern": "Double Bottom", "confidence": 0.0}

    lows = d["low"]
    closes = d["close"]
    mins = _rolling_argrelextrema(lows, order=3, mode="min")
    if len(mins) < 2:
        return {"found": False, "pattern": "Double Bottom", "confidence": 0.0, "last_index": d.index[-1]}

    for i in range(len(mins) - 1):
        i1, i2 = mins[i], mins[i + 1]
        p1, p2 = lows.loc[i1], lows.loc[i2]

        # separazione minima
        if (d.index.get_loc(i2) - d.index.get_loc(i1)) < min_separation:
            continue

        # simmetria
        if abs(_pct(p1, p2)) > bottoms_tol:
            continue

        seg = d.loc[i1:i2]
        neckline = float(seg["high"].max())
        breakout_now = bool(closes.iloc[-1] > neckline)
        if neckline_confirm and not breakout_now:
            continue

        sym = 1 - min(1.0, abs(_pct(p1, p2)) / bottoms_tol)
        brk = max(0.0, _pct(closes.iloc[-1], neckline))
        conf = float(np.clip(0.5 * sym + 0.5 * min(brk / 0.02, 1.0), 0, 1))

        return {
            "found": True,
            "pattern": "Double Bottom",
            "points": {"bottom1_idx": i1, "bottom2_idx": i2},
            "levels": {"neckline": neckline, "bottom_avg": float((p1 + p2) / 2.0)},
            "confidence": conf,
            "neckline_confirmed": breakout_now,
            "neckline_breakout": breakout_now,  # alias
            "breakout_index": d.index[-1] if breakout_now else None,
            "last_index": d.index[-1],
            "note": f"Neckline {neckline:.6f} — simmetria {sym:.2f}, breakout {brk:.2%}",
        }

    return {"found": False, "pattern": "Double Bottom", "confidence": 0.0, "last_index": d.index[-1]}


# =========================================================
# 1b) Double Top (bearish, mirror)
# =========================================================

def detect_double_top(
    df: pd.DataFrame,
    lookback: int = 180,
    min_separation: int = 5,
    tops_tol: float = 0.02,
    neckline_confirm: bool = True,
) -> Dict[str, Any]:
    d = df.tail(lookback).copy()
    req = {"low", "high", "close"}
    if len(d) < 10 or not req.issubset(d.columns):
        return {"found": False, "pattern": "Double Top", "confidence": 0.0}

    d = d.dropna(subset=list(req))
    if len(d) < 10:
        return {"found": False, "pattern": "Double Top", "confidence": 0.0}

    highs = d["high"]
    closes = d["close"]
    maxs = _rolling_argrelextrema(highs, order=3, mode="max")
    if len(maxs) < 2:
        return {"found": False, "pattern": "Double Top", "confidence": 0.0, "last_index": d.index[-1]}

    for i in range(len(maxs) - 1):
        i1, i2 = maxs[i], maxs[i + 1]
        p1, p2 = highs.loc[i1], highs.loc[i2]

        if (d.index.get_loc(i2) - d.index.get_loc(i1)) < min_separation:
            continue
        if abs(_pct(p1, p2)) > tops_tol:
            continue

        seg = d.loc[i1:i2]
        neckline = float(seg["low"].min())
        breakout_now = bool(closes.iloc[-1] < neckline)
        if neckline_confirm and not breakout_now:
            continue

        sym = 1 - min(1.0, abs(_pct(p1, p2)) / tops_tol)
        brk = max(0.0, _pct(neckline, closes.iloc[-1]))
        conf = float(np.clip(0.5 * sym + 0.5 * min(brk / 0.02, 1.0), 0, 1))

        return {
            "found": True,
            "pattern": "Double Top",
            "points": {"top1_idx": i1, "top2_idx": i2},
            "levels": {"neckline": neckline, "top_avg": float((p1 + p2) / 2.0)},
            "confidence": conf,
            "neckline_confirmed": breakout_now,
            "neckline_breakout": breakout_now,  # alias
            "breakout_index": d.index[-1] if breakout_now else None,
            "last_index": d.index[-1],
            "note": f"Neckline {neckline:.6f} — simmetria {sym:.2f}, breakout {brk:.2%}",
        }

    return {"found": False, "pattern": "Double Top", "confidence": 0.0, "last_index": d.index[-1]}


# =========================================================
# 2) Ascending Triangle (bullish)
# =========================================================

def detect_ascending_triangle(
    df: pd.DataFrame,
    lookback: int = 200,
    touch_tol: float = 0.015,
    min_touches: int = 2,
    breakout_confirm: bool = True,
    use_tail_window: bool = True,
    require_volume_for_breakout: bool = False,
    breakout_vol_mult: float = 1.30,
) -> Dict[str, Any]:
    """
    Ritorna: found, pattern, confidence, levels, breakout_confirmed, breakout_index, last_index, note
    Compatibile con chiamate esistenti (parametri extra hanno default).
    """
    d = df.tail(lookback).copy()
    req = {"high", "low", "close"}
    if len(d) < 10 or not req.issubset(d.columns):
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0}

    d = d.dropna(subset=list(req))
    if len(d) < 10:
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0}

    highs = d["high"]
    lows = d["low"]
    closes = d["close"]

    # Resistenza piatta: coda recente o quantile 0.90
    if use_tail_window:
        k = max(5, len(d) // 3)
        res = float(highs.tail(k).max())
    else:
        res = float(highs.quantile(0.90))

    hits = _level_touch_ratio(highs, res, touch_tol)
    if hits < (min_touches / max(5, len(d))):
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    # Base ascendente (trendline minimi)
    x = np.arange(len(lows))
    slope = np.polyfit(x, lows.values, 1)[0]
    if slope <= 0:
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    breakout_now = bool(closes.iloc[-1] > res)

    if require_volume_for_breakout and len(d) > 1 and "volume" in d.columns:
        vol_last = float(d["volume"].iloc[-1])
        vol_mean = float(d["volume"].iloc[:-1].mean())
        breakout_now = breakout_now and (vol_mean > 0 and vol_last > vol_mean * breakout_vol_mult)

    if breakout_confirm and not breakout_now:
        return {"found": False, "pattern": "Ascending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    brk = max(0.0, _pct(closes.iloc[-1], res))
    denom = max(res * 0.002, 1e-12)  # normalizza slope
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


# =========================================================
# 2b) Descending Triangle (bearish, mirror)
# =========================================================

def detect_descending_triangle(
    df: pd.DataFrame,
    lookback: int = 200,
    touch_tol: float = 0.015,
    min_touches: int = 2,
    breakout_confirm: bool = True,
    use_tail_window: bool = True,
    require_volume_for_breakout: bool = False,
    breakout_vol_mult: float = 1.30,
) -> Dict[str, Any]:
    d = df.tail(lookback).copy()
    req = {"high", "low", "close"}
    if len(d) < 10 or not req.issubset(d.columns):
        return {"found": False, "pattern": "Descending Triangle", "confidence": 0.0}

    d = d.dropna(subset=list(req))
    if len(d) < 10:
        return {"found": False, "pattern": "Descending Triangle", "confidence": 0.0}

    highs = d["high"]
    lows = d["low"]
    closes = d["close"]

    # Supporto piatto: coda recente o quantile 0.10
    if use_tail_window:
        k = max(5, len(d) // 3)
        sup = float(lows.tail(k).min())
    else:
        sup = float(lows.quantile(0.10))

    hits = _level_touch_ratio(lows, sup, touch_tol)
    if hits < (min_touches / max(5, len(d))):
        return {"found": False, "pattern": "Descending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    # Top discendente (trendline massimi)
    x = np.arange(len(highs))
    slope = np.polyfit(x, highs.values, 1)[0]
    if slope >= 0:
        return {"found": False, "pattern": "Descending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    breakout_now = bool(closes.iloc[-1] < sup)

    if require_volume_for_breakout and len(d) > 1 and "volume" in d.columns:
        vol_last = float(d["volume"].iloc[-1])
        vol_mean = float(d["volume"].iloc[:-1].mean())
        breakout_now = breakout_now and (vol_mean > 0 and vol_last > vol_mean * breakout_vol_mult)

    if breakout_confirm and not breakout_now:
        return {"found": False, "pattern": "Descending Triangle", "confidence": 0.0, "last_index": d.index[-1]}

    brk = max(0.0, _pct(sup, closes.iloc[-1]))
    denom = max(sup * 0.002, 1e-12)
    conf = float(np.clip(min(brk / 0.02, 1.0) * 0.6 + min(abs(slope) / denom, 1.0) * 0.4, 0, 1))

    return {
        "found": True,
        "pattern": "Descending Triangle",
        "points": {},
        "levels": {"support": sup},
        "confidence": conf,
        "breakout_confirmed": breakout_now,
        "breakout_index": d.index[-1] if breakout_now else None,
        "last_index": d.index[-1],
        "note": f"Supporto ~{sup:.6f}, breakout {brk:.2%}, slope highs {slope:.6g}",
    }


# =========================================================
# 3) Price Channel (ascending/descending/sideways)
# =========================================================

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Ritorna slope, intercept, r^2."""
    if len(x) < 2 or len(y) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0, 0.0
    m, q = np.polyfit(x, y, 1)
    yhat = m * x + q
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(m), float(q), float(max(min(r2, 1.0), 0.0))


def detect_price_channel(
    hist: pd.DataFrame,
    lookback: int = 200,
    min_touches_side: int = 2,
    parallel_tolerance: float = 0.20,   # differenza relativa pendenze
    touch_tol_mult_atr: float = 0.55,   # tolleranza tocchi = 0.55 * ATR (fallback width/200)
    min_confidence: float = 0.55,
    require_volume_for_breakout: bool = True,
    breakout_vol_mult: float = 1.30,
) -> Dict[str, Any]:
    """
    Rileva un canale di prezzo.
    Ritorna: found, type, confidence, slopes/intercepts, width_now, touches, points,
             breakout_confirmed, breakout_side
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

    if hist is None or len(hist) < 30:
        return out
    req_cols = {"high", "low", "close"}
    if not req_cols.issubset(hist.columns):
        return out

    df = hist.tail(lookback).copy()
    df = df.dropna(subset=list(req_cols))
    if len(df) < 30:
        return out

    n = len(df)
    x = np.arange(n).astype(float)
    highs = df["high"].values.astype(float)
    lows  = df["low"].values.astype(float)

    # Fit upper/lower
    m_u, q_u, r2_u = _fit_line(x, highs)
    m_l, q_l, r2_l = _fit_line(x, lows)

    upper_line = m_u * x + q_u
    lower_line = m_l * x + q_l
    upper_now = float(upper_line[-1])
    lower_now = float(lower_line[-1])

    # Normalizza se invertiti
    if lower_now > upper_now:
        m_u, m_l = m_l, m_u
        q_u, q_l = q_l, q_u
        upper_line, lower_line = lower_line, upper_line
        upper_now, lower_now = lower_now, upper_now
        r2_u, r2_l = r2_l, r2_u

    width_now = max(upper_now - lower_now, 1e-12)

    # Parallelismo relativo
    denom = max(abs(m_u), abs(m_l), 1e-12)
    par_err = abs(m_u - m_l) / denom
    parallel_ok = par_err <= parallel_tolerance

    # Tolleranza tocchi
    if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]):
        atr_now = float(df["ATR"].iloc[-1])
    else:
        atr_now = 0.0
    touch_tol = (touch_tol_mult_atr * atr_now) if atr_now > 0 else (width_now / 200.0)

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

    # Confidence composita
    r2_score = max(0.0, (r2_u + r2_l) / 2.0)
    touches_score = min((touches_up + touches_lo) / (2.0 * max(1, min_touches_side)), 1.0)
    coverage_score = 1.0 if (touches_up >= min_touches_side and touches_lo >= min_touches_side) else 0.0
    par_score = max(0.0, 1.0 - min(par_err / max(parallel_tolerance, 1e-12), 1.0))

    confidence = 0.35 * par_score + 0.35 * r2_score + 0.20 * touches_score + 0.10 * coverage_score
    confidence = max(0.0, min(confidence, 1.0))

    if not parallel_ok or confidence < min_confidence:
        return out

    # Breakout ultima candela (con filtro volume opzionale)
    close_last = float(df["close"].iloc[-1])
    high_last  = float(df["high"].iloc[-1])
    low_last   = float(df["low"].iloc[-1])
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
        breakout_up   = breakout_up   and (vol_last > vol_mean * breakout_vol_mult)
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


# =========================================================
# 4) Candlestick avanzati (Morning/Evening Star, Soldiers/Crows,
#    Piercing/Dark Cloud, Tweezer, Doji vari)
# =========================================================

def _real_body(o, c): 
    return abs(c - o)

def _body_pct(o, h, l, c):
    rng = max(1e-9, h - l)
    return _real_body(o, c) / rng

def _color(o, c):
    return "bull" if c > o else ("bear" if c < o else "doji")

def _is_small_body(o, h, l, c, th=0.25):
    return _body_pct(o, h, l, c) <= th

def _is_long_body(o, h, l, c, th=0.55):
    return _body_pct(o, h, l, c) >= th

def _upper_shadow(o, h, l, c):
    return h - max(o, c)

def _lower_shadow(o, h, l, c):
    return min(o, c) - l

def _gap_down(prev_h, curr_l):
    # Per crypto: gap "logico"
    return curr_l < prev_h and (prev_h - curr_l) / max(1e-9, prev_h) > 0.001

def _gap_up(prev_l, curr_h):
    return curr_h > prev_l and (curr_h - prev_l) / max(1e-9, prev_l) > 0.001


def detect_candlestick_patterns(
    df: pd.DataFrame,
    atr: Optional[pd.Series] = None,
    vol: Optional[pd.Series] = None
) -> List[Dict[str, Any]]:
    """
    Rileva pattern avanzati sulle ultime barre:
      Morning/Evening Star, Three Soldiers/Crows (robust),
      Piercing/Dark Cloud, Tweezer Top/Bottom, Doji (Dragonfly/Gravestone/Long-legged)
    Ritorna: list[{name, direction, confidence, notes, window}]
    """
    out: List[Dict[str, Any]] = []
    if df is None or len(df) < 3:
        return out

    # Pulizia minima
    cols = ["open", "high", "low", "close"]
    if not set(cols).issubset(df.columns):
        return out
    d = df.dropna(subset=cols)
    if len(d) < 3:
        return out

    o2,h2,l2,c2 = d.iloc[-3][['open','high','low','close']]
    o1,h1,l1,c1 = d.iloc[-2][['open','high','low','close']]
    o0,h0,l0,c0 = d.iloc[-1][['open','high','low','close']]

    # --- Morning Star ---
    if _color(o2,c2)=="bear" and _is_long_body(o2,h2,l2,c2) and _is_small_body(o1,h1,l1,c1,0.35):
        cond = (_color(o0,c0)=="bull" and _is_long_body(o0,h0,l0,c0) and c0 >= (o2 + (c2 - o2)*0.5))
        if cond:
            conf = 0.65 + (0.05 if _gap_down(h2, l1) else 0.0)
            out.append({"name":"Morning Star","direction":"bull","confidence":min(conf,0.9),"notes":"3-candle reversal","window":3})

    # --- Evening Star ---
    if _color(o2,c2)=="bull" and _is_long_body(o2,h2,l2,c2) and _is_small_body(o1,h1,l1,c1,0.35):
        cond = (_color(o0,c0)=="bear" and _is_long_body(o0,h0,l0,c0) and c0 <= (o2 + (c2 - o2)*0.5))
        if cond:
            conf = 0.65 + (0.05 if _gap_up(l2, h1) else 0.0)
            out.append({"name":"Evening Star","direction":"bear","confidence":min(conf,0.9),"notes":"3-candle reversal","window":3})

    # --- Three White Soldiers (robust) ---
    if len(d) >= 4:
        a,b,c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
        def bull(x): return x["close"] > x["open"]
        def body(x): return abs(x["close"] - x["open"])
        def up_w(x): return x["high"] - max(x["open"], x["close"])
        def lo_w(x): return min(x["open"], x["close"]) - x["low"]
        cond = (
            bull(a) and bull(b) and bull(c) and
            (c["close"] > b["close"] > a["close"]) and
            (body(b) >= 0.7 * body(a)) and (body(c) >= 0.7 * body(b)) and
            (b["open"] >= min(a["open"], a["close"]) and b["open"] <= max(a["open"], a["close"])) and
            (c["open"] >= min(b["open"], b["close"]) and c["open"] <= max(b["open"], b["close"])) and
            (up_w(a) <= body(a) and lo_w(a) <= body(a) and
             up_w(b) <= body(b) and lo_w(b) <= body(b) and
             up_w(c) <= body(c) and lo_w(c) <= body(c))
        )
        if cond:
            out.append({"name":"Three White Soldiers","direction":"bull","confidence":0.62,"notes":"3 strong bull bodies","window":3})

    # --- Three Black Crows (robust) ---
    if len(d) >= 4:
        a,b,c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
        def bear(x): return x["close"] < x["open"]
        def body(x): return abs(x["close"] - x["open"])
        def up_w(x): return x["high"] - max(x["open"], x["close"])
        def lo_w(x): return min(x["open"], x["close"]) - x["low"]
        cond = (
            bear(a) and bear(b) and bear(c) and
            (c["close"] < b["close"] < a["close"]) and
            (body(b) >= 0.7 * body(a)) and (body(c) >= 0.7 * body(b)) and
            (b["open"] >= min(a["open"], a["close"]) and b["open"] <= max(a["open"], a["close"])) and
            (c["open"] >= min(b["open"], b["close"]) and c["open"] <= max(b["open"], b["close"])) and
            (up_w(a) <= body(a) and lo_w(a) <= body(a) and
             up_w(b) <= body(b) and lo_w(b) <= body(b) and
             up_w(c) <= body(c) and lo_w(c) <= body(c))
        )
        if cond:
            out.append({"name":"Three Black Crows","direction":"bear","confidence":0.62,"notes":"3 strong bear bodies","window":3})

    # --- Piercing Line (bull) ---
    if _color(o1,c1)=="bear" and _color(o0,c0)=="bull":
        if _gap_down(h1, l0) and c0 >= (o1 + (c1 - o1)*0.5):
            out.append({"name":"Piercing Line","direction":"bull","confidence":0.56,"notes":"gap down + 50% body recovery","window":2})

    # --- Dark Cloud Cover (bear) ---
    if _color(o1,c1)=="bull" and _color(o0,c0)=="bear":
        if _gap_up(l1, h0) and c0 <= (o1 + (c1 - o1)*0.5):
            out.append({"name":"Dark Cloud Cover","direction":"bear","confidence":0.56,"notes":"gap up + 50% body drop","window":2})

    # --- Tweezer Top / Bottom ---
    wick_tol = max((h0 - l0) * 0.002, 1e-9)
    if abs(h0 - h1) <= wick_tol and _color(o0,c0)=="bear":
        out.append({"name":"Tweezer Top","direction":"bear","confidence":0.50,"notes":"equal highs rejection","window":2})
    if abs(l0 - l1) <= wick_tol and _color(o0,c0)=="bull":
        out.append({"name":"Tweezer Bottom","direction":"bull","confidence":0.50,"notes":"equal lows rejection","window":2})

    # --- Doji (Dragonfly, Gravestone, Long-legged) ---
    body_small = _is_small_body(o0,h0,l0,c0,0.15)
    if body_small:
        upper = _upper_shadow(o0,h0,l0,c0)
        lower = _lower_shadow(o0,h0,l0,c0)
        name, direction, conf = None, "neutral", 0.45
        if lower >= _real_body(o0,c0)*2 and upper <= _real_body(o0,c0)*0.5:
            name, direction = "Dragonfly Doji", "bull"
        elif upper >= _real_body(o0,c0)*2 and lower <= _real_body(o0,c0)*0.5:
            name, direction = "Gravestone Doji", "bear"
        elif upper >= _real_body(o0,c0) and lower >= _real_body(o0,c0):
            name, direction = "Long-legged Doji", "neutral"
        if name:
            out.append({"name":name,"direction":direction,"confidence":conf,"notes":"indecision / potential reversal","window":1})

    # --- Filtri ATR/Volume (rafforza/filtra) ---
    last_idx = d.index[-1]
    atrv = float(atr.loc[last_idx]) if (atr is not None and last_idx in atr.index) else None
    volv = float(vol.loc[last_idx]) if (vol is not None and last_idx in vol.index) else None
    if atrv is not None or volv is not None:
        adjusted: List[Dict[str, Any]] = []
        for p in out:
            keep = True
            if "Doji" in p["name"] and atrv is not None:
                rng = max(1e-9, d.iloc[-1]["high"] - d.iloc[-1]["low"])
                if rng > 0 and (atrv / rng) < 0.25:  # scarta doji in micro-range
                    keep = False
            if volv is not None:
                p["confidence"] = min(0.95, p["confidence"] + 0.05)  # volume rafforza di poco
            if keep:
                adjusted.append(p)
        out = adjusted

    return out
