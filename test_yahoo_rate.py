import yfinance as yf
import pandas as pd

print("yfinance version:", yf.__version__)
ticker = "GC=F"   # oro futures (lo stesso che hai testato con curl da Windows)

print(f"\nScarico dati raw per {ticker} (7d, 15m)...\n")

try:
    df = yf.download(
        tickers=ticker,
        period="7d",
        interval="15m",
        auto_adjust=False,
        prepost=False,
        progress=True,
        threads=False,
        # se la tua versione lo supporta, prova anche:
        # raise_errors=True,
    )
except Exception as e:
    print(f"\n❌ Eccezione da yfinance: {repr(e)}")
else:
    print("\n=== INFO DATAFRAME ===")
    print(df.info())
    print("\n=== PRIME 5 RIGHE ===")
    print(df.head())
    print("\n=== ULTIME 5 RIGHE ===")
    print(df.tail())
