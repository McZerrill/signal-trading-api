#!/usr/bin/env python3
"""
Test minimale per yahoo_api.get_yahoo_df su macro + crypto.

Si lancia con:
    python3 test_yahoo_rate.py
"""

import time
import traceback

import yfinance as yf
from yahoo_api import get_yahoo_df, YAHOO_SYMBOL_MAP

# Simboli “logici” che usi nel backend
SYMBOLS = [
    "XAUUSD",  # Oro
    "XAGUSD",  # Argento
    "SP500",
    "NAS100",
    "DAX40",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
]


def main() -> None:
    print("=== Test YAHOO (via yahoo_api.get_yahoo_df) ===")
    print("yfinance version:", getattr(yf, "__version__", "unknown"))
    print("\nSymbol map:")
    for s in SYMBOLS:
        print(f"  {s:7s} -> {YAHOO_SYMBOL_MAP.get(s, s)}")

    for round_no in (1, 2):
        print(f"\n[ROUND #{round_no}]")
        for s in SYMBOLS:
            y_sym = YAHOO_SYMBOL_MAP.get(s, s)

            try:
                # usiamo direttamente il ticker Yahoo (GC=F, BTC-USD, ecc.)
                df = get_yahoo_df(y_sym, interval="15m")
            except Exception as e:
                print(f"{s:7s} ({y_sym}) -> ERRORE: {repr(e)}")
                traceback.print_exc()
                continue

            if df is None or df.empty:
                print(f"{s:7s} ({y_sym}) -> VUOTO (len=0)")
            else:
                last = df.iloc[-1]
                try:
                    last_close = float(last.get("close", 0.0))
                except Exception:
                    last_close = 0.0
                print(
                    f"{s:7s} ({y_sym}) -> OK len={len(df)} "
                    f"last_close={last_close:.6f}"
                )

        if round_no == 1:
            print("...aspetto 5 secondi prima del prossimo round...")
            time.sleep(5)


if __name__ == "__main__":
    main()

