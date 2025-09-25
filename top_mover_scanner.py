import time
import requests
import logging
from threading import Thread

# Importa la tua funzione di analisi
from routes import analyze


def scan_top_movers(
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,  # +10% per coin già attive
    gain_threshold_listing: float = 1.00   # +100% per coin appena listate (1 sola candela)
):
    """
    Scanner top mover Binance (1m).
    - Se coin normale (>=2 barre 1m): allerta se ultima candela > gain_threshold_normale rispetto alla precedente.
    - Se nuovo listing (1 candela): allerta se candela > gain_threshold_listing rispetto all'open e candela piena.
    """
    url_tickers = "https://api.binance.com/api/v3/ticker/24hr"
    url_klines = "https://api.binance.com/api/v3/klines"

    while True:
        try:
            # 1) Prendi tutti i ticker
            tickers = requests.get(url_tickers, timeout=5).json()
            movers = [d for d in tickers if d["symbol"].endswith("USDT")]

            for d in movers:
                symbol = d["symbol"]

                try:
                    # 2) Prendi fino a 3 candele da 1m
                    kl = requests.get(
                        url_klines,
                        params={"symbol": symbol, "interval": "1m", "limit": 3},
                        timeout=3
                    ).json()

                    if not kl:
                        continue

                    # Caso A: almeno 2 barre disponibili
                    if len(kl) >= 2:
                        prev_close = float(kl[-2][4])
                        last_close = float(kl[-1][4])
                        if prev_close > 0:
                            gain = (last_close / prev_close) - 1
                            if gain >= gain_threshold_normale:
                                logging.info(f"🔥 Top mover rilevato: {symbol} {gain:.2%}")
                                try:
                                    result = analyze(symbol)
                                    logging.info(f"[TopMover] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[TopMover] errore analyze {symbol}: {e}")

                    # Caso B: nuovo listing (1 sola candela)
                    if len(kl) == 1:
                        o, h, l, c, v = float(kl[0][1]), float(kl[0][2]), float(kl[0][3]), float(kl[0][4]), float(kl[0][5])
                        if o > 0:
                            gain = (c / o) - 1
                            body_frac = abs(c - o) / max(h - l, 1e-9)
                            if gain >= gain_threshold_listing and body_frac >= 0.7:
                                logging.info(f"🚀 NUOVO LISTING sospetto pump: {symbol} {gain:.2%}")
                                try:
                                    result = analyze(symbol)
                                    logging.info(f"[ListingPump] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[ListingPump] errore analyze {symbol}: {e}")

                except Exception:
                    continue

        except Exception as e:
            logging.error(f"Errore scan_top_movers: {e}")

        time.sleep(interval_sec)


def start_top_mover_scanner(
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,
    gain_threshold_listing: float = 1.00
):
    """
    Avvia lo scanner in un thread separato.
    """
    t = Thread(
        target=scan_top_movers,
        kwargs={
            "interval_sec": interval_sec,
            "gain_threshold_normale": gain_threshold_normale,
            "gain_threshold_listing": gain_threshold_listing,
        },
        daemon=True
    )
    t.start()
    logging.info("✅ Thread Top Mover Scanner avviato")
    return t


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    start_top_mover_scanner()
    while True:
        time.sleep(3600)  # mantiene vivo il main thread
