import numpy as np
import pandas as pd

def _safe_close(x: float, eps: float = 1e-9) -> float:
    try:
        if x is None or not np.isfinite(x):
            return eps
        return x if x > eps else eps
    except Exception:
        return eps

def ema_quality_buy_gate(
    hist: pd.DataFrame,
    w: int = 8,
    # slope minimi (relativi al close)
    min_slope7: float = 0.00025,
    min_slope25: float = 0.00012,
    # EMA25 “retta”: curvatura massima
    max_curv25: float = 0.00010,
    # separazione 7-25 deve crescere
    min_sep_growth_7_25: float = 0.00020,
    # opzionale: separazione 25-99 deve crescere un minimo
    min_sep_growth_25_99: float = 0.00010,
    require_25_99: bool = True,
) -> tuple[bool, str, dict]:
    """
    Gate BUY netto basato su EMA:
    - pendenza positiva “pulita” (EMA7 e EMA25)
    - EMA25 non deve curvare/oscillare (curvatura bassa)
    - spread EMA7-EMA25 in aumento (divergenza)
    - opzionale: spread EMA25-EMA99 in aumento
    Ritorna: (ok, reason, metrics)
    """
    if hist is None or len(hist) < w + 3:
        return False, "hist too short", {}

    for col in ("close", "EMA_7", "EMA_25", "EMA_99"):
        if col not in hist.columns:
            return False, f"missing {col}", {}

    c = _safe_close(float(hist["close"].iloc[-1]))

    e7  = pd.to_numeric(hist["EMA_7"].tail(w), errors="coerce").values.astype(float)
    e25 = pd.to_numeric(hist["EMA_25"].tail(w), errors="coerce").values.astype(float)
    e99 = pd.to_numeric(hist["EMA_99"].tail(w), errors="coerce").values.astype(float)

    if not (np.isfinite(e7).all() and np.isfinite(e25).all() and np.isfinite(e99).all()):
        return False, "nan in EMA window", {}

    # slope semplice su finestra
    slope7  = (e7[-1]  - e7[0])  / c
    slope25 = (e25[-1] - e25[0]) / c

    # curvatura EMA25 (ultima variazione vs precedente)
    d25_now  = (float(hist["EMA_25"].iloc[-1]) - float(hist["EMA_25"].iloc[-2])) / c
    d25_prev = (float(hist["EMA_25"].iloc[-2]) - float(hist["EMA_25"].iloc[-3])) / c
    curv25   = abs(d25_now - d25_prev)

    # spread growth
    sep_7_25_now  = abs(e7[-1] - e25[-1]) / c
    sep_7_25_old  = abs(e7[0]  - e25[0])  / c
    sep_grow_7_25 = sep_7_25_now - sep_7_25_old

    sep_25_99_now  = abs(e25[-1] - e99[-1]) / c
    sep_25_99_old  = abs(e25[0]  - e99[0])  / c
    sep_grow_25_99 = sep_25_99_now - sep_25_99_old

    # EMA25 “retta” anche come progressione: evita zig-zag (rumore)
    d25 = np.diff(e25) / c
    # tolleranza: consenti 1 piccolo passo negativo (rumore)
    tol = 0.00002
    neg = int(np.sum(d25 < -tol))
    ema25_monotone = bool(neg <= 1)

    metrics = {
        "slope7": slope7,
        "slope25": slope25,
        "curv25": curv25,
        "sep_grow_7_25": sep_grow_7_25,
        "sep_grow_25_99": sep_grow_25_99,
        "ema25_monotone": ema25_monotone,
    }

    # check step-by-step (reason utile nei log)
    if slope7 < min_slope7:
        return False, "slope7 low", metrics
    if slope25 < min_slope25:
        return False, "slope25 low", metrics
    if not ema25_monotone:
        return False, "ema25 not monotone", metrics
    if curv25 > max_curv25:
        return False, "ema25 too curvy", metrics
    if sep_grow_7_25 < min_sep_growth_7_25:
        return False, "7-25 not separating", metrics
    if require_25_99 and (sep_grow_25_99 < min_sep_growth_25_99):
        return False, "25-99 not separating", metrics

    return True, "ok", metrics
    
__all__ = ["ema_quality_buy_gate"]
