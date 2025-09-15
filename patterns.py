# patterns.py
import pandas as pd
import numpy as np

# ------------------------------ #
# Helpers
# ------------------------------ #
def _pct(a, b):
    if b == 0 or b is None:
        return 0.0
    return (a - b) / b

def _rolling_argrelextrema(series: pd.Series, order: int = 3, mode: str = "min"):
    """
    Restituisce gli indici (labels) dei minimi/massimi locali univoci
    usando una finestra di ampiezza 2*order+1.
    """
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
    """
    Quota di barre che "toccano" il livello (entro una tolleranza relativa).
    """
    tol = abs(level) * tol_frac
    hits = ((series - level).abs() <= tol).sum()
    return hits / max(1, len(series))

def _pos_in_df(df: pd.DataFrame, label):
    """
    Converte un label di index nella POSIZIONE INTERA all’interno del df completo.
    Serve per allinearsi a _pattern_recent di analizza_trend.
    """
    loc = df.index.get_loc(label)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, (np.ndarray, list)):
        return int(loc[0])
    return int(loc)

# ------------------------------ #
# 1) Doppio Minimo (Double Bottom)
# ------------------------------ #
def detect_double_bottom(
    df: pd.DataFrame,
    lookback: int = 180,
    min_separation: int = 5,
    bottoms_tol: float = 0.02,
    neckline_confirm: bool = True
):
    """
    Rileva un Double Bottom nell'ultima finestra di lookback barre.

    Ritorna:
      {
        "found": bool,
        "pattern": "Double Bottom",
        "levels": {"neckline": float, "bottom_avg": float},
        "points": {"bottom1_idx": <label>, "bottom2_idx": <label>},
        # --- chiavi TOP-LEVEL per _pattern_recent ---
        "right_bottom_idx": int,        # posizione intera nel df completo
        "neckline_break_idx": int,      # posizione intera dell'ultima barra (se breakout ora)
        "last_index": int,              # fallback (ultima barra)
        # --- conferme/score ---
        "neckline_confirmed": bool,
        "neckline_breakout": bool,
        "confidence": float (0..1),
        "note": str
      }
    """
    if df is None or len(df) < max(2 * min_separation + 5, 40):
        return {"found": False, "pattern": "Double Bottom"}

    d = df.tail(lookback).copy()
    lows = d["low"]
    closes = d["close"]

    mins = _rolling_argrelextrema(lows, order=3, mode="min")
    if len(mins) < 2:
        return {"found": False, "pattern": "Double Bottom"}

    # Prova coppie di minimi adiacenti (più recenti prioritari)
    for i in range(len(mins) - 2, -1, -1):
        i1, i2 = mins[i], mins[i + 1]
        p1, p2 = lows.loc[i1], lows.loc[i2]

        # separazione minima in barre (sulla FINESTRA d)
        if (d.index.get_loc(i2) - d.index.get_loc(i1)) < min_separation:
            continue

        # minimi "simili"
        if abs(_pct(p1, p2)) > bottoms_tol:
            continue

        # neckline = massimo tra i due minimi
        seg = d.loc[i1:i2]
        neckline = seg["high"].max()

        close_now = closes.iloc[-1]
        neckline_breakout = close_now > neckline
        neckline_confirmed = (not neckline_confirm) or neckline_breakout
        if neckline_confirm and not neckline_breakout:
            # se chiedi conferma, esci se non c'è breakout della neckline
            continue

        # Score semplice: simmetria + forza breakout
        sym = 1 - min(1.0, abs(_pct(p1, p2)) / bottoms_tol)         # 0..1
        brk = max(0.0, _pct(close_now, neckline))                    # %
        conf = float(np.clip(0.5 * sym + 0.5 * min(brk / 0.02, 1.0), 0, 1))

        # POSIZIONI INTERE sul df completo (non sulla sola finestra d)
        i1_pos_full = _pos_in_df(df, i1)
        i2_pos_full = _pos_in_df(df, i2)
        breakout_pos_full = len(df) - 1

        return {
            "found": True,
            "pattern": "Double Bottom",
            "levels": {
                "neckline": float(neckline),
                "bottom_avg": float((p1 + p2) / 2.0),
            },
            "points": {"bottom1_idx": i1, "bottom2_idx": i2},
            # --- chiavi top-level per gating/recency ---
            "right_bottom_idx": int(i2_pos_full),
            "neckline_break_idx": int(breakout_pos_full),
            "last_index": int(breakout_pos_full),
            # --- flags di conferma + score ---
            "neckline_confirmed": bool(neckline_confirmed),
            "neckline_breakout": bool(neckline_breakout),
            "confidence": conf,
            "note": f"neckline ~{neckline:.6f}, sym={sym:.2f}, brk={brk:.2%}",
        }

    return {"found": False, "pattern": "Double Bottom"}

# ----------------------------------------- #
# 2) Triangolo Ascendente (rialzista)
# ----------------------------------------- #
def detect_ascending_triangle(
    df: pd.DataFrame,
    lookback: int = 200,
    touch_tol: float = 0.015,
    min_touches: int = 2,
    breakout_confirm: bool = True
):
    """
    Rileva un triangolo ascendente nell'ultima finestra di lookback barre.

    Ritorna:
      {
        "found": bool,
        "pattern": "Ascending Triangle",
        "levels": {"resistance": float},
        "points": {},
        # --- chiavi TOP-LEVEL per _pattern_recent ---
        "breakout_idx": int,     # posizione intera dell'ultima barra (se breakout ora)
        "last_index": int,
        # --- conferme/score ---
        "breakout_confirmed": bool,
        "confidence": float (0..1),
        "note": str
      }
    """
    if df is None or len(df) < 40:
        return {"found": False, "pattern": "Ascending Triangle"}

    d = df.tail(lookback).copy()
    highs = d["high"]
    lows = d["low"]
    closes = d["close"]

    # Resistenza "piatta" stimata (alto percentile)
    res = highs.quantile(0.90)

    # Numero relativo di tocchi alla resistenza
    hits_ratio = _level_touch_ratio(highs, res, touch_tol)
    # Soglia minima: almeno 'min_touches' su max(5, len(d)) barre
    if hits_ratio < (min_touches / max(5, len(d))):
        return {"found": False, "pattern": "Ascending Triangle"}

    # Trend dei minimi: deve essere crescente
    x = np.arange(len(lows))
    slope = np.polyfit(x, lows.values, 1)[0]
    if slope <= 0:
        return {"found": False, "pattern": "Ascending Triangle"}

    # Conferma breakout
    breakout_now = closes.iloc[-1] > res
    breakout_confirmed = (not breakout_confirm) or breakout_now
    if not breakout_confirmed:
        return {"found": False, "pattern": "Ascending Triangle"}

    # Confidenza: mix tra breakout % e pendenza dei minimi
    brk = max(0.0, _pct(closes.iloc[-1], res))
    conf = float(np.clip(
        0.6 * min(brk / 0.02, 1.0) + 0.4 * min(abs(slope) / (abs(res) * 0.002), 1.0),
        0, 1
    ))

    breakout_pos_full = len(df) - 1

    return {
        "found": True,
        "pattern": "Ascending Triangle",
        "levels": {"resistance": float(res)},
        "points": {},
        # --- chiavi top-level per gating/recency ---
        "breakout_confirmed": True,
        "breakout_idx": int(breakout_pos_full),
        "last_index": int(breakout_pos_full),
        # --- score/nota ---
        "confidence": conf,
        "note": f"res ~{res:.6f}, brk={brk:.2%}, slope={slope:.6g}",
    }
