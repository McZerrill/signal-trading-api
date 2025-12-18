# pump_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
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

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_1m.columns)
    if missing:
        return PrePumpResult(triggered=False, reason=f"colonne mancanti: {sorted(missing)}")

    # forza numerico (evita stringhe / object)
    for c in required:
        df_1m[c] = pd.to_numeric(df_1m[c], errors="coerce")

    if df_1m[list(required)].isna().any().any():
        return PrePumpResult(triggered=False, reason="NaN in colonne OHLCV")

    tail = df_1m.tail(10)

    p0 = float(tail["close"].iloc[0])
    p5 = float(tail["close"].iloc[5])
    p9 = float(tail["close"].iloc[-1])

    gain_10m = (p9 / max(p0, 1e-9)) - 1.0
    gain_5m  = (p9 / max(p5, 1e-9)) - 1.0

    v_last3 = float(tail["volume"].tail(3).mean())
    v_prev7 = float(tail["volume"].head(7).mean())
    vol_mult = v_last3 / max(v_prev7, 1e-9)

    # “candela piena” nelle ultime 3 (evita doji rumorose) - vettoriale
    last3 = tail.tail(3).copy()
    body = (last3["close"] - last3["open"]).abs()
    rng  = (last3["high"] - last3["low"]).clip(lower=1e-9)

    if ((body / rng) < 0.55).any():
        return PrePumpResult(
            triggered=False,
            gain_5m=gain_5m,
            gain_10m=gain_10m,
            vol_mult=vol_mult,
            last_price=p9,
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


def make_hotassets_entry_pre_pump(
    symbol: str,
    price: float,
    volx: float,
    gain: float,
    display_name: str,
    reason: str = "",
    gain5: float = 0.0,
    gain10: float = 0.0,
) -> Dict[str, Any]:
    g5  = float(gain5) if gain5 else 0.0
    g10 = float(gain10) if gain10 else (float(gain) if gain else 0.0)

    extra = []
    if g5:
        extra.append(f"g5={g5*100:.1f}%")
    if g10:
        extra.append(f"g10={g10*100:.1f}%")
    extra.append(f"volx={float(volx):.1f}")
    if reason:
        extra.append(str(reason))

    return {
        "symbol": symbol,
        "segnali": 1,
        "trend": "BUY",
        "rsi": None,
        "ema7": 0.0, "ema25": 0.0, "ema99": 0.0,
        "prezzo": round(float(price), 4),
        "candele_trend": 1,
        "note": f"⚡ PRE-PUMP 1m ({' • '.join(extra)}) • 🛈 {display_name}"
    }
