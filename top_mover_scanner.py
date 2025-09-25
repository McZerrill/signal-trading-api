import time
import requests
import logging
from threading import Thread
from typing import Callable, Optional


def scan_top_movers(
    analyze_fn: Callable[[str], any],
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,  # +10% in 1m rispetto alla candela precedente
    gain_threshold_listing: float = 1.00   # +100% per coin appena listate (1 sola candela)
):
    """
    Scanner top mover Binance (1m).

    - Caso A (>=2 barre): allerta se l'ultima close è > +gain_threshold_normale rispetto alla precedente.
    - Caso B (1 sola barra = nuovo listing): allerta se (close/open) - 1 >= gain_threshold_listing
      e la candela è 'piena' (body >= 70% del range).
    """
    url_tickers = "https://api.binance.com/api/v3/ticker/24hr"
    url_klines = "https://api.binance.com/api/v3/klines"

    sess = requests.Session()

    while True:
        try:
            tickers = sess.get(url_tickers, timeout=5).json()
            # Se vuoi includere anche USDC, usa:  if d["symbol"].endswith(("USDT","USDC"))
            movers = [d for d in tickers if d.get("symbol","").endswith("USDC")]

            for d in movers:
                symbol = d["symbol"]

                try:
                    kl = sess.get(
                        url_klines,
                        params={"symbol": symbol, "interval": "1m", "limit": 3},
                        timeout=3
                    ).json()
                    if not kl:
                        continue

                    # ---- Caso A: almeno 2 barre disponibili
                    if len(kl) >= 2:
                        prev_close = float(kl[-2][4])
                        last_close = float(kl[-1][4])
                        if prev_close > 0:
                            gain = (last_close / prev_close) - 1.0
                            if gain >= gain_threshold_normale:
                                logging.info(f"🔥 Top mover rilevato: {symbol} {gain:.2%}")
                                try:
                                    result = analyze_fn(symbol)
                                    logging.info(f"[TopMover] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[TopMover] analyze error {symbol}: {e}")

                    # ---- Caso B: nuovo listing (1 sola candela)
                    if len(kl) == 1:
                        o, h, l, c, v = map(float, (kl[0][1], kl[0][2], kl[0][3], kl[0][4], kl[0][5]))
                        if o > 0:
                            gain = (c / o) - 1.0
                            body_frac = abs(c - o) / max(h - l, 1e-9)
                            if gain >= gain_threshold_listing and body_frac >= 0.70:
                                logging.info(f"🚀 NUOVO LISTING sospetto pump: {symbol} {gain:.2%}")
                                try:
                                    result = analyze_fn(symbol)
                                    logging.info(f"[ListingPump] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[ListingPump] analyze error {symbol}: {e}")

                except Exception as e:
                    logging.debug(f"[scan_loop] skip {symbol}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Errore scan_top_movers: {e}")

        time.sleep(interval_sec)


def start_top_mover_scanner(
    analyze_fn: Callable[[str], any],
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
            "analyze_fn": analyze_fn,  # <— passiamo la callback
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

    # Esempio di callback fittizia per test manuale:
    def _fake_analyze(sym: str):
        class R: segnale="HOLD"; commento="test"
        return R()
    start_top_mover_scanner(_fake_analyze)
    while True:
        time.sleep(3600)
