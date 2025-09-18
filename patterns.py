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
