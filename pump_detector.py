# pump_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class PrePumpResult:
    triggered: bool
    gain_5m: float = 0.0
    gain_10m: float = 0.0
    vol_mult: float = 0.0
    last_price: float = 0.0
    reason: str = ""

def detect_pre_pump_1m(df_1m: pd.DataFrame) -> PrePumpResult:
    """
    PRE-PUMP 1m: intercetta accelerazione precoce (prezzo + volume + candele piene).
    df_1m: DataFrame 1m con colonne open/high/low/close/volume.
    """
    if not isinstance(df_1m, pd.DataFrame) or df_1m.empty or len(df_1m) < 12:
        return PrePumpResult(triggered=False, reason="df_1m insufficiente")

    tail = df_1m.tail(10)

    p0 = float(tail["close"].iloc[0])
    p5 = float(tail["close"].iloc[5])
    p9 = float(tail["close"].iloc[-1])

    gain_10m = (p9 / max(p0, 1e-9)) - 1.0
    gain_5m  = (p9 / max(p5, 1e-9)) - 1.0

    v_last3 = float(tail["volume"].tail(3).mean())
    v_prev7 = float(tail["volume"].head(7).mean())
    vol_mult = v_last3 / max(v_prev7, 1e-9)

    # “candela piena” nelle ultime 3 (evita doji rumorose)
    last3 = tail.tail(3)
    for _, r in last3.iterrows():
        body = abs(float(r["close"]) - float(r["open"]))
        rng  = max(float(r["high"]) - float(r["low"]), 1e-9)
        if (body / rng) < 0.55:
            return PrePumpResult(
                triggered=False,
                gain_5m=gain_5m, gain_10m=gain_10m, vol_mult=vol_mult, last_price=p9,
                reason="body_frac troppo basso"
            )

    # soglie (anticipate)
    if (gain_5m >= 0.05 or gain_10m >= 0.08) and vol_mult >= 2.0:
        return PrePumpResult(
            triggered=True,
            gain_5m=gain_5m,
            gain_10m=gain_10m,
            vol_mult=vol_mult,
            last_price=p9,
            reason="pre-pump 1m (gain+volume)"
        )

    return PrePumpResult(
        triggered=False,
        gain_5m=gain_5m,
        gain_10m=gain_10m,
        vol_mult=vol_mult,
        last_price=p9,
        reason="soglie non superate"
    )

def make_hotassets_entry_pre_pump(symbol: str, price: float, volx: float, gain: float, display_name: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "segnali": 1,
        "trend": "BUY",
        "rsi": None,
        "ema7": 0.0, "ema25": 0.0, "ema99": 0.0,
        "prezzo": round(float(price), 4),
        "candele_trend": 1,
        "note": f"⚡ PRE-PUMP 1m (gain={gain*100:.1f}%, volx={volx:.1f}) • 🛈 {display_name}"
    }
