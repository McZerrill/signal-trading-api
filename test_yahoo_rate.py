import time

from yahoo_api import get_yahoo_df, get_yahoo_last_price, YAHOO_SYMBOL_MAP

SYMBOLS = ["XAUUSD", "XAGUSD", "SP500", "NAS100", "DAX40"]


def main():
    print("=== Test yfinance + cache interna ===")
    print("Symbol map:", {s: YAHOO_SYMBOL_MAP.get(s, s) for s in SYMBOLS})
    print()

    # Primo giro: dovrebbe fare le chiamate vere verso Yahoo
    for s in SYMBOLS:
        y_ticker = YAHOO_SYMBOL_MAP.get(s, s)
        try:
            df = get_yahoo_df(y_ticker, interval="15m")
            last_price = get_yahoo_last_price(y_ticker)
            print(
                f"[ROUND #1] {s} ({y_ticker}) -> len(df)={len(df)} "
                f"last_close={last_price}"
            )
        except Exception as e:
            print(f"[ROUND #1] {s} ({y_ticker}) -> ERRORE: {e}")

    print("\nAspetto 5 secondi e riprovo (dovrebbe usare quasi solo cache)...\n")
    time.sleep(5)

    # Secondo giro: se la cache funziona, non dovresti scatenare 429
    for s in SYMBOLS:
        y_ticker = YAHOO_SYMBOL_MAP.get(s, s)
        try:
            df = get_yahoo_df(y_ticker, interval="15m")
            last_price = get_yahoo_last_price(y_ticker)
            print(
                f"[ROUND #2] {s} ({y_ticker}) -> len(df)={len(df)} "
                f"last_close={last_price}"
            )
        except Exception as e:
            print(f"[ROUND #2] {s} ({y_ticker}) -> ERRORE: {e}")


if __name__ == "__main__":
    main()
