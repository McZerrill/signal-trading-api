import time
from yahoo_api import get_yahoo_df, YAHOO_SYMBOL_MAP

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

print("=== Test YAHOO CRYPTO (via yahoo_api.get_yahoo_df) ===")
print("Symbol map:")
for s in CRYPTO_SYMBOLS:
    print(f"  {s} -> {YAHOO_SYMBOL_MAP.get(s, s)}")

for round_no in range(1, 3):
    print(f"\n[ROUND #{round_no}]")
    for s in CRYPTO_SYMBOLS:
        y_symbol = YAHOO_SYMBOL_MAP.get(s, s)
        try:
            # NB: qui passo il TICKER YAHOO (es. BTC-USD) perché
            # il tuo get_yahoo_df chiama direttamente yfinance su 'symbol'
            df = get_yahoo_df(y_symbol, interval="15m")

            if df is None or df.empty:
                print(f"{s} ({y_symbol}) -> VUOTO (len=0)")
                continue

            last = df.iloc[-1]
            ts = df.index[-1]
            print(
                f"{s} ({y_symbol}) -> len={len(df)} "
                f"last_close={last['close']:.4f} @ {ts}"
            )
        except Exception as e:
            print(f"{s} ({y_symbol}) -> ERRORE: {e}")

    if round_no < 2:
        print("...aspetto 5 secondi prima del prossimo round...\n")
        time.sleep(5)
