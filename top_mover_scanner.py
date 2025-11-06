# top_mover_scanner.py
# Scanner 1m per top movers Binance con:
# - trigger UP e DOWN (BUY/SELL)
# - rilevamento listing (1 sola candela, body ≥ 70%)
# - prefiltro 24h, cooldown anti-spam
# - hook extra_symbols_fn (es. /hotassets) per includere simboli aggiuntivi
# - compatibile con routes.py (start_top_mover_scanner)

import time
import random
import logging
from threading import Thread
from typing import Callable, Any, Tuple, Dict, List, Optional

import requests


# =========================
# Helper: HTTP con retry
# =========================
def _http_get_with_retry(
    sess: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = 6.0,
    max_retries: int = 3,
) -> Optional[requests.Response]:
    """
    GET con retry/backoff esponenziale semplice.
    Ritorna None se fallisce definitivamente.
    """
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries:
                logging.warning(f"[HTTP] GET fallito {url} (tentativo {attempt}/{max_retries}): {e}")
                return None
            sleep_s = backoff * (1.0 + 0.25 * random.random())
            time.sleep(sleep_s)
            backoff *= 2.0
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Scanner principale (1m)
# =========================
def scan_top_movers(
    analyze_fn: Callable[[str], Any],
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.10,     # +10% o -10% in 1 minuto
    gain_threshold_listing: float = 1.00,     # +100% o -100% su prima candela (listing)
    quote_suffix: Tuple[str, ...] = ("USDC", "USDT"),
    top_n_24h: int = 80,                      # prefiltra: analizza i migliori per %change 24h
    cooldown_sec: int = 180,                  # non ri-triggerare la stessa coin entro X sec
    extra_symbols_fn: Optional[Callable[[], List[str]]] = None,  # es. hotassets()
):
    """
    - Caso A (>=2 barre 1m): trigger se |last/prev - 1| >= gain_threshold_normale (UP/DOWN).
    - Caso B (1 sola barra = nuovo listing): trigger se body>=70% e |close/open -1|>=gain_threshold_listing.
    - Prefiltro: considera solo i top 'top_n_24h' simboli per priceChangePercent nel 24h.
    - Cooldown: non richiama analyze per la stessa coin entro 'cooldown_sec'.
    - extra_symbols_fn: callable che restituisce una lista di simboli extra (dedup automatico).
    """
    url_tickers = "https://api.binance.com/api/v3/ticker/24hr"
    url_klines = "https://api.binance.com/api/v3/klines"

    sess = requests.Session()
    sess.headers.update({"User-Agent": "top-mover-scanner/1.1"})

    last_alert: Dict[str, float] = {}  # symbol -> last_ts

    while True:
        loop_started_at = time.time()
        try:
            # ===== 24h tickers (prefiltro) =====
            r = _http_get_with_retry(sess, url_tickers, timeout=8.0)
            if r is None:
                raise RuntimeError("Impossibile ottenere /ticker/24hr")

            tickers = r.json()
            if not isinstance(tickers, list):
                logging.warning("[TopMover] payload 24hr non è una lista, skip ciclo")
                raise RuntimeError("Payload non valido per /ticker/24hr")

            # Solo le quote desiderate e ordina per %change 24h
            filtered = (d for d in tickers if str(d.get("symbol", "")).endswith(quote_suffix))
            try:
                movers = sorted(
                    filtered,
                    key=lambda x: _safe_float(x.get("priceChangePercent", 0.0)),
                    reverse=True
                )[:max(1, int(top_n_24h))]
            except Exception:
                movers = [d for d in filtered]

            # ===== Costruisci lista finale (movers + extra_symbols_fn) =====
            final_symbols: List[str] = [str(d.get("symbol") or "").strip().upper() for d in movers if d.get("symbol")]
            if extra_symbols_fn:
                try:
                    extras = [s.strip().upper() for s in (extra_symbols_fn() or []) if s]
                    # dedup preservando l'ordine
                    final_symbols = list(dict.fromkeys(final_symbols + extras))
                except Exception as _e:
                    logging.warning(f"[TopMover] extra_symbols_fn error: {_e}")

            # ===== Loop simboli selezionati =====
            for symbol in final_symbols:
                if not symbol:
                    continue

                now = time.time()
                # cooldown anti-spam
                if now - last_alert.get(symbol, 0.0) < cooldown_sec:
                    continue

                # klines 1m (limite 3 per avere prev/last e rilevare listing)
                rr = _http_get_with_retry(
                    sess,
                    url_klines,
                    params={"symbol": symbol, "interval": "1m", "limit": 3},
                    timeout=6.0,
                )
                if rr is None:
                    logging.debug(f"[scan_loop] HTTP klines fallito per {symbol}")
                    continue

                try:
                    kl = rr.json()
                except Exception:
                    continue

                if not isinstance(kl, list) or not kl:
                    continue

                triggered = False

                # ---- Caso A: almeno 2 barre 1m disponibili (UP o DOWN)
                if len(kl) >= 2:
                    prev_close = _safe_float(kl[-2][4])
                    last_close = _safe_float(kl[-1][4])
                    if prev_close > 0.0 and last_close > 0.0:
                        gain = (last_close / prev_close) - 1.0
                        if gain >= gain_threshold_normale:
                            logging.info(
                                f"🔥 Top mover UP: {symbol} +{gain:.2%} "
                                f"(prev={prev_close:.8f} → last={last_close:.8f})"
                            )
                            triggered = True
                        elif gain <= -gain_threshold_normale:
                            logging.info(
                                f"📉 Top mover DOWN: {symbol} {gain:.2%} "
                                f"(prev={prev_close:.8f} → last={last_close:.8f})"
                            )
                            triggered = True

                # ---- Caso B: nuovo listing (1 sola candela) – UP o DOWN con body “pieno”
                if not triggered and len(kl) == 1:
                    try:
                        o = _safe_float(kl[0][1])
                        h = _safe_float(kl[0][2])
                        l = _safe_float(kl[0][3])
                        c = _safe_float(kl[0][4])
                    except Exception:
                        o = h = l = c = 0.0

                    if o > 0.0 and h >= l:
                        gain = (c / o) - 1.0
                        body_frac = abs(c - o) / max(h - l, 1e-9)
                        if body_frac >= 0.70 and abs(gain) >= gain_threshold_listing:
                            direction = "UP" if gain > 0 else "DOWN"
                            logging.info(
                                f"🆕 LISTING {direction}: {symbol} gain={gain:.2%} body={body_frac:.0%} "
                                f"(open={o:.8f} high={h:.8f} low={l:.8f} close={c:.8f})"
                            )
                            triggered = True

                # ---- Se trigger, aggiorno il cooldown e invoco analyze_fn ----
                if triggered:
                    last_alert[symbol] = now
                    try:
                        result = analyze_fn(symbol)
                        segnale = getattr(result, "segnale", None) or getattr(result, "signal", "HOLD")
                        commento = getattr(result, "commento", None) or getattr(result, "comment", "")
                        logging.info(f"[TopMover→analyze] {symbol} → {segnale} | {commento}")
                    except Exception as e:
                        logging.warning(f"[TopMover] analyze() errore per {symbol}: {e}")

        except Exception as e:
            logging.error(f"[TopMover] errore scan_top_movers: {e}")

        # attende il prossimo ciclo (compensando il tempo già speso) + jitter anti-sync
        elapsed = time.time() - loop_started_at
        sleep_left = max(0.0, interval_sec - elapsed)
        time.sleep(sleep_left + random.uniform(0.0, 0.5))


def start_top_mover_scanner(
    analyze_fn: Callable[[str], Any],
    interval_sec: int = 60,
    gain_threshold_normale: float = 0.07,
    gain_threshold_listing: float = 0.80,
    quote_suffix: Tuple[str, ...] = ("USDC", "USDT"),
    top_n_24h: int = 80,
    cooldown_sec: int = 120,
    extra_symbols_fn: Optional[Callable[[], List[str]]] = None,
) -> Thread:
    """
    Avvia lo scanner in un thread separato.
    Mantiene la compatibilità con routes.py.
    """
    t = Thread(
        target=scan_top_movers,
        kwargs={
            "analyze_fn": analyze_fn,
            "interval_sec": int(interval_sec),
            "gain_threshold_normale": float(gain_threshold_normale),
            "gain_threshold_listing": float(gain_threshold_listing),
            "quote_suffix": tuple(quote_suffix),
            "top_n_24h": int(top_n_24h),
            "cooldown_sec": int(cooldown_sec),
            "extra_symbols_fn": extra_symbols_fn,
        },
        daemon=True
    )
    t.start()
    logging.info("✅ Thread Top Mover Scanner avviato")
    return t


# ===== Esecuzione manuale (facoltativa) =====
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Callback fittizia per test rapido
    def _fake_analyze(sym: str):
        class R:
            segnale = "HOLD"
            commento = f"test analyze su {sym}"
        return R()

    # Esempio: nessun extra_symbols_fn
    start_top_mover_scanner(
        analyze_fn=_fake_analyze,
        interval_sec=30,
        gain_threshold_normale=0.05,
        gain_threshold_listing=0.80,
        quote_suffix=("USDC", "USDT"),
        top_n_24h=120,
        cooldown_sec=90,
        extra_symbols_fn=None,
    )

    while True:
        time.sleep(3600)
