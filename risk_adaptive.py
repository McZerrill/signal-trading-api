# risk_adaptive.py
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd

def compute_adaptive_tp_sl(
    df: pd.DataFrame,
    entry: float,
    direction: str,                 # "BUY" | "SELL"
    atr_mult: float = 2.2,
    swing_lookback: int = 30,
    risk_min: float = 0.005,        # 0.5%
    risk_max: float = 0.03,         # 3%
    rr_min: float = 1.25,
    be_trigger_R: float = 0.8,      # breakeven a 0.8R
    be_buffer_pct: float = 0.0005   # 5 bps
) -> Dict:
    notes: List[str] = []
    if df is None or len(df) < max(2, swing_lookback):
        return {
            'sl': None, 'tp': None, 'rr': 0.0, 'risk_pct': 0.0,
            'valid': False, 'be_enabled': False,
            'be_trigger_price': None, 'be_stop_price': None,
            'notes': ['df troppo corto o mancante per calcolo adattivo']
        }

    # ATR (usa colonna 'ATR' se presente, altrimenti calcolo veloce ATR14)
    if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]):
        atr_val = float(df['ATR'].iloc[-1])
        notes.append(f'ATR from df: {atr_val:.6f}')
    else:
        high = df['high']; low = df['low']; close = df['close']
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        notes.append(f'ATR fallback calc: {atr_val:.6f}')

    # Swing strutturali (ultime N barre)
    swing_slice = df.iloc[-swing_lookback:]
    swing_min = float(swing_slice['low'].min())
    swing_max = float(swing_slice['high'].max())
    notes.append(f'swing_min[{swing_lookback}]: {swing_min:.6f}, swing_max: {swing_max:.6f}')

    tiny_buf = entry * be_buffer_pct  # micro-buffer
    if direction == 'BUY':
        sl_atr   = entry - atr_mult * atr_val
        sl_swing = min(swing_min, entry) - tiny_buf
        sl_raw   = min(sl_atr, sl_swing)
    else:
        sl_atr   = entry + atr_mult * atr_val
        sl_swing = max(swing_max, entry) + tiny_buf
        sl_raw   = max(sl_atr, sl_swing)

    notes.append(f'sl_atr: {sl_atr:.6f}, sl_swing: {sl_swing:.6f}, sl_raw: {sl_raw:.6f}')

    # Vincoli rischio min/max in % su entry
    if direction == 'BUY':
        sl_min_risk = entry * (1 - risk_min)   # più vicino
        sl_max_risk = entry * (1 - risk_max)   # più lontano
        sl_clamped = max(sl_max_risk, min(sl_raw, sl_min_risk))
    else:
        sl_min_risk = entry * (1 + risk_min)
        sl_max_risk = entry * (1 + risk_max)
        sl_clamped = min(sl_max_risk, max(sl_raw, sl_min_risk))

    notes.append(f'sl_clamped: {sl_clamped:.6f} (risk_min={risk_min:.3%}, risk_max={risk_max:.3%})')

    # Distanza rischio e TP coerente con RR minimo + swing opposto
    if direction == 'BUY':
        risk_abs = entry - sl_clamped
        if risk_abs <= 0:
            return _invalid(notes + ['risk_abs<=0 BUY'])
        tp_rr   = entry + rr_min * risk_abs
        tp_swg  = max(entry, swing_max) - tiny_buf
        tp      = max(tp_rr, tp_swg)
        rr      = (tp - entry) / risk_abs
        risk_pct = risk_abs / entry
        be_trigger_price = entry + be_trigger_R * risk_abs
        be_stop_price    = entry + tiny_buf
    else:
        risk_abs = sl_clamped - entry
        if risk_abs <= 0:
            return _invalid(notes + ['risk_abs<=0 SELL'])
        tp_rr   = entry - rr_min * risk_abs
        tp_swg  = min(entry, swing_min) + tiny_buf
        tp      = min(tp_rr, tp_swg)
        rr      = (entry - tp) / risk_abs
        risk_pct = risk_abs / entry
        be_trigger_price = entry - be_trigger_R * risk_abs
        be_stop_price    = entry - tiny_buf

    notes.append(f'tp_rr: {tp_rr:.6f}, tp_struct: {tp_swg:.6f}, tp_final: {tp:.6f}, rr: {rr:.2f}, risk_pct: {risk_pct:.3%}')
    valid = rr >= rr_min
    if not valid:
        notes.append(f'RR {rr:.2f} < rr_min {rr_min:.2f} → invalida segnale')

    return {
        'sl': float(sl_clamped),
        'tp': float(tp),
        'rr': float(rr),
        'risk_pct': float(risk_pct),
        'valid': bool(valid),
        'be_enabled': True,
        'be_trigger_price': float(be_trigger_price),
        'be_stop_price': float(be_stop_price),
        'notes': notes,
    }

def _invalid(notes: List[str]) -> Dict:
    return {
        'sl': None, 'tp': None, 'rr': 0.0, 'risk_pct': 0.0,
        'valid': False, 'be_enabled': False,
        'be_trigger_price': None, 'be_stop_price': None,
        'notes': notes
    }

def maybe_activate_breakeven(pos: Dict, last_price: float, direction: str) -> Optional[str]:
    if not pos or not pos.get("be_enabled"):
        return None
    trigger = pos.get("be_trigger_price"); be_sl = pos.get("be_stop_price")
    if trigger is None or be_sl is None:
        return None
    if direction == "BUY" and last_price >= trigger and pos.get("sl", -float("inf")) < be_sl:
        pos["sl"] = be_sl
        return f"Breakeven attivato a {trigger:.6f} → SL spostato a {be_sl:.6f}"
    if direction == "SELL" and last_price <= trigger and pos.get("sl", float('inf')) > be_sl:
        pos["sl"] = be_sl
        return f"Breakeven attivato a {trigger:.6f} → SL spostato a {be_sl:.6f}"
    return None
