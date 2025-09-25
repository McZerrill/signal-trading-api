import time
import requests
import logging
from threading import Thread
from typing import Callable, Optional, Tuple

def scan_top_movers(
    analyze_fn: Callable[[str], any],
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,   # +10% in 1m vs candela precedente
    gain_threshold_listing: float = 1.00,   # +100% per coin appena listate (1 sola candela)
    quote_suffix: Tuple[str, ...] = ("USDC",),  # puoi mettere ("USDT",) o ("USDT","USDC")
    top_n_24h: int = 80,                   # prefiltra: analizza solo i top n per %change 24h
    cooldown_sec: int = 180                # evita spam: non ri-triggerare la stessa coin entro X secondi
):
    """
    Scanner top mover Binance (1m).

    - Caso A (>=2 barre 1m): allerta se last_close > prev_close * (1 + gain_threshold_normale).
    - Caso B (1 sola barra = nuovo listing): allerta se (close/open - 1) >= gain_threshold_listing
      e la candela è 'piena' (body >= 70% del range).
    - Prefiltro: considera solo i top 'top_n_24h' simboli per priceChangePercent nel 24h.
    - Cooldown: non richiama analyze per la stessa coin entro 'cooldown_sec'.
    """
    url_tickers = "https://api.binance.com/api/v3/ticker/24hr"
    url_klines = "https://api.binance.com/api/v3/klines"

    sess = requests.Session()
    last_alert = {}  # symbol -> last_ts

    while True:
        loop_started_at = time.time()
        try:
            # ===== 24h tickers (prefiltro) =====
            r = sess.get(url_tickers, timeout=5)
            r.raise_for_status()
            tickers = r.json()
            if not isinstance(tickers, list):
                logging.warning("[TopMover] payload 24hr non-list, skip ciclo")
                time.sleep(interval_sec)
                continue

            # Solo le quote desiderate (USDC/USDT) e ordina per %change 24h
            filtered = (d for d in tickers if str(d.get("symbol", "")).endswith(quote_suffix))
            try:
                movers = sorted(
                    filtered,
                    key=lambda x: float(x.get("priceChangePercent", "0") or 0.0),
                    reverse=True
                )[:max(1, top_n_24h)]
            except Exception:
                # se manca priceChangePercent, fallback senza sort
                movers = [d for d in filtered]

            for d in movers:
                symbol = d.get("symbol")
                if not symbol:
                    continue

                # cooldown per non spammare
                now = time.time()
                if now - last_alert.get(symbol, 0) < cooldown_sec:
                    continue

                try:
                    rr = sess.get(
                        url_klines,
                        params={"symbol": symbol, "interval": "1m", "limit": 3},
                        timeout=4
                    )
                    rr.raise_for_status()
                    kl = rr.json()
                    if not isinstance(kl, list) or not kl:
                        continue

                    # ---- Caso A: almeno 2 barre disponibili
                    if len(kl) >= 2:
                        prev_close = float(kl[-2][4])
                        last_close = float(kl[-1][4])
                        if prev_close > 0:
                            gain = (last_close / prev_close) - 1.0
                            if gain >= gain_threshold_normale:
                                logging.info(
                                    f"🔥 Top mover: {symbol} gain={gain:.2%} "
                                    f"(prev={prev_close:.6f} → last={last_close:.6f})"
                                )
                                try:
                                    result = analyze_fn(symbol)
                                    logging.info(f"[TopMover] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[TopMover] analyze error {symbol}: {e}")
                                else:
                                    last_alert[symbol] = now

                    # ---- Caso B: nuovo listing (1 sola candela)
                    if len(kl) == 1:
                        o, h, l, c, v = map(float, (kl[0][1], kl[0][2], kl[0][3], kl[0][4], kl[0][5]))
                        if o > 0:
                            gain = (c / o) - 1.0
                            body_frac = abs(c - o) / max(h - l, 1e-9)
                            if gain >= gain_threshold_listing and body_frac >= 0.70:
                                logging.info(
                                    f"🚀 NUOVO LISTING: {symbol} gain={gain:.2%} body={body_frac:.0%} "
                                    f"(open={o:.6f} high={h:.6f} low={l:.6f} close={c:.6f})"
                                )
                                try:
                                    result = analyze_fn(symbol)
                                    logging.info(f"[ListingPump] {symbol} → {result.segnale} | {result.commento}")
                                except Exception as e:
                                    logging.warning(f"[ListingPump] analyze error {symbol}: {e}")
                                else:
                                    last_alert[symbol] = now

                except requests.RequestException as e:
                    logging.debug(f"[scan_loop] HTTP err {symbol}: {e}")
                    continue
                except Exception as e:
                    logging.debug(f"[scan_loop] skip {symbol}: {e}")
                    continue

        except requests.RequestException as e:
            logging.error(f"[TopMover] HTTP errore 24hr: {e}")
        except Exception as e:
            logging.error(f"[TopMover] errore scan_top_movers: {e}")

        # attende il prossimo ciclo (compensando il tempo già speso)
        elapsed = time.time() - loop_started_at
        sleep_left = max(0.0, interval_sec - elapsed)
        time.sleep(sleep_left)


def start_top_mover_scanner(
    analyze_fn: Callable[[str], any],
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,
    gain_threshold_listing: float = 1.00,
    quote_suffix: Tuple[str, ...] = ("USDC",),
    top_n_24h: int = 80,
    cooldown_sec: int = 180
):
    """
    Avvia lo scanner in un thread separato.
    Mantiene la compatibilità: i primi 4 parametri sono gli stessi già usati in routes.py.
    """
    t = Thread(
        target=scan_top_movers,
        kwargs={
            "analyze_fn": analyze_fn,
            "interval_sec": interval_sec,
            "gain_threshold_normale": gain_threshold_normale,
            "gain_threshold_listing": gain_threshold_listing,
            "quote_suffix": quote_suffix,
            "top_n_24h": top_n_24h,
            "cooldown_sec": cooldown_sec,
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

    # Callback fittizia per test manuale
    def _fake_analyze(sym: str):
        class R:
            segnale = "HOLD"
            commento = "test"
        return R()

    start_top_mover_scanner(_fake_analyze)
    while True:
        time.sleep(3600)
