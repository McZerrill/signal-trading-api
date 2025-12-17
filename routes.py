from fastapi import APIRouter
from datetime import datetime, timezone as dt_timezone, timedelta
import time
import requests
import logging
import pandas as pd
# Thread di monitoraggio attivo ogni 5 secondi
import threading
from binance_api import get_binance_df, get_best_symbols, get_bid_ask, get_symbol_tick_step
# Supporto Yahoo Finance
from yahoo_api import get_yahoo_df, get_yahoo_last_price, YAHOO_SYMBOL_MAP


from trend_logic import analizza_trend, conta_candele_trend
from indicators import calcola_rsi, calcola_macd, calcola_atr
from models import SignalResponse

from top_mover_scanner import start_top_mover_scanner
import json
from pathlib import Path

try:
    CURRENT_COMMIT = Path(".current_commit").read_text().strip()
except Exception:
    CURRENT_COMMIT = "unknown"


logging.basicConfig(
    filename="log.txt",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
logging.debug("🧪 LOG DI TEST DEBUG all'avvio")
logging.warning("🚀 BACKEND RIAVVIATO — routes.py ricaricato")


router = APIRouter()
utc = dt_timezone.utc

# Yahoo: soglia freschezza dati (minuti)
YAHOO_STALE_MAX_MIN = 25   # 25m: tollera 15m + ritardo + jitter


# ------------------------------------------------------------------
# WHITELIST asset che devono comparire sempre in /hotassets
# ------------------------------------------------------------------
HOT_WHITELIST_BASE = {
    "BCH","ATOM","QTUM","AVAX","APT","DOGE","ETH","AXS","TRX",
    "XRP","SOL","A","MANA","LINK","DOT","BTC","NEAR","ETC","SAND",
    "LTC","DASH","BNB","ZEC","ADA",
}

QUOTES = ("USDT", "USDC")
SIM_LOG_PATH = Path("simulazioni_chiuse_log.jsonl")

# ------------------------------------------------------------------
# WHITELIST Yahoo (macro/indici + azioni) da mostrare in coda a /hotassets
# ------------------------------------------------------------------
# Whitelist Yahoo: sempre coerente con YAHOO_SYMBOL_MAP
YAHOO_HOT_LIST = [ "XAUUSD", "XAGUSD", "SP500", "NAS100", "DAX40", "AAPL", "MSFT", "NVDA", "TSLA", ]



# Nome leggibile per asset Binance (whitelist) + macro Yahoo + azioni
ASSET_NAME_MAP = {
    # --- Binance crypto (base symbol) ---
    "BTC":  "Bitcoin",
    "ETH":  "Ethereum",
    "SOL":  "Solana",
    "XRP":  "XRP",
    "ADA":  "Cardano",
    "BNB":  "BNB",
    "LTC":  "Litecoin",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche",
    "ATOM": "Cosmos",
    "MANA": "Decentraland",
    "SAND": "The Sandbox",
    "LINK": "Chainlink",
    "DOT":  "Polkadot",
    "NEAR": "NEAR Protocol",
    "ETC":  "Ethereum Classic",
    "DASH": "Dash",
    "ZEC":  "Zcash",
    "BCH":  "Bitcoin Cash",
    "AXS":  "Axie Infinity",
    "TRX":  "TRON",
    "QTUM": "Qtum",
    "APT":  "Aptos",
    "A":    "Vaulta",

    # --- Macro/Indici/FX (Yahoo) ---
    "DJI": "Dow Jones",
    "VIX": "VIX (Volatility Index)",
    "RUS2000": "Russell 2000",
    "OIL": "Crude Oil (WTI futures)",
    "NGAS": "Natural Gas (futures)",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "MAG7": "Magnificent 7 (ETF proxy)",

    # --- Azioni Yahoo (nuove) ---
    "ORCL": "Oracle",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "MU": "Micron Technology",
    "KO": "Coca-Cola",
    "MCD": "McDonald's",
    "COST": "Costco",
    "WMT": "Walmart",
    "V": "Visa",
    "MA": "Mastercard",
    "UPS": "United Parcel Service",
    "QCOM": "Qualcomm",
    "ADBE": "Adobe",
    "INTC": "Intel",
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly",
    "ABBV": "AbbVie",
    "ABT": "Abbott Laboratories",
    "BRKB": "Berkshire Hathaway (B)",
    "BABA": "Alibaba",
    "MELI": "MercadoLibre",
    "COIN": "Coinbase",
    "RDDT": "Reddit",
    "GME": "GameStop",
    "CAT": "Caterpillar",
    "FCX": "Freeport-McMoRan",
    "AA": "Alcoa",
    "GOLD": "Barrick Gold",
    "BYND": "Beyond Meat",
    "SPCE": "Virgin Galactic",
    "LYFT": "Lyft",
    "SEDG": "SolarEdge",
    "FSLR": "First Solar",
    "AAL": "American Airlines",
    "LUV": "Southwest Airlines",
    "CCL": "Carnival",
    "DAL": "Delta Air Lines",
    "AMC": "AMC Entertainment",
    "ACB": "Aurora Cannabis",
    "FERRARI": "Ferrari N.V.",

        # --- Macro / indici (Yahoo) ---
    "XAUUSD": "Oro (Gold futures)",
    "XAGUSD": "Argento (Silver futures)",
    "SP500":  "S&P 500",
    "NAS100": "Nasdaq 100",
    "DAX40":  "DAX 40",

    # --- Titoli azionari (Yahoo) ---
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",


}

def _asset_display_name(symbol: str) -> str:
    """
    Restituisce un nome leggibile per:
    - crypto Binance (BTCUSDT → BTC)
    - simboli Yahoo (AAPL, XAUUSD, ecc.)
    """
    if not symbol:
        return symbol

    base = symbol

    if symbol.endswith("USDT") or symbol.endswith("USDC"):
        base = symbol[:-4]

    # 🔴 FIX: prova SEMPRE prima con base, poi con symbol
    return ASSET_NAME_MAP.get(base) or ASSET_NAME_MAP.get(symbol) or base






def _augment_with_whitelist(symbols: list[str]) -> list[str]:
    """
    Garantisce che ogni asset base in HOT_WHITELIST_BASE
    sia presente come simbolo completo (es. BTCUSDT/BTCUSDC).
    Se uno dei due esiste già nei risultati del filtro, evita duplicati.
    """
    current = set(symbols)
    base_present = {
        s[:-4] if s.endswith("USDT") else (s[:-4] if s.endswith("USDC") else s)
        for s in symbols
    }

    extra = []

    for base in HOT_WHITELIST_BASE:
        if base in base_present:
            continue

        for q in QUOTES:
            full = f"{base}{q}"
            if full not in current:
                extra.append(full)
                current.add(full)
                break  # usa una sola coppia per non riempire troppo

    return symbols + extra


# Stato simulazioni attive
posizioni_attive = {}
_pos_lock = threading.Lock()

# Quanto tempo tenere in memoria le simulazioni chiuse (in ore)
CLEANUP_AGE_HOURS = 1


def cleanup_posizioni_attive():
    """
    Rimuove da posizioni_attive tutte le simulazioni chiuse
    da più di CLEANUP_AGE_HOURS.
    """
    now = datetime.now(dt_timezone.utc)
    cutoff = now - timedelta(hours=CLEANUP_AGE_HOURS)

    to_delete = []

    with _pos_lock:
        for symbol, pos in posizioni_attive.items():
            # tieni solo quelle effettivamente chiuse da backend
            if not pos.get("chiusa_da_backend", False):
                continue

            ora_chiusura = pos.get("ora_chiusura")
            if not ora_chiusura:
                continue

            try:
                dt = datetime.fromisoformat(ora_chiusura)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=dt_timezone.utc)
            except Exception:
                # se la data è strana, non rischiamo di cancellare
                continue

            if dt < cutoff:
                to_delete.append(symbol)

        # cancelliamo fuori dal ciclo principale
        for symbol in to_delete:
            posizioni_attive.pop(symbol, None)

    if to_delete:
        logging.info(f"🧹 Cleanup posizioni_attive: rimossi {len(to_delete)} simboli chiusi da oltre {CLEANUP_AGE_HOURS}h")

def append_simulazione_chiusa(symbol: str, sim: dict):
    """
    Appende una simulazione chiusa in formato JSONL.
    Ogni riga è un JSON indipendente -> facile da greppare/analizzare.
    """
    try:
        record = {
            "symbol": symbol,
            "tipo": sim.get("tipo"),
            "entry": float(sim.get("entry", 0.0)),
            "tp": float(sim.get("tp", 0.0)),
            "sl": float(sim.get("sl", 0.0)),
            "prezzo_chiusura": float(sim.get("prezzo_chiusura", 0.0)),
            "esito": sim.get("esito"),
            "variazione_pct": float(sim.get("variazione_pct", 0.0)),
            "motivo": sim.get("motivo"),
            "note_notifica": sim.get("note_notifica", ""),
            "ora_chiusura": sim.get("ora_chiusura"),
            "chiusa_da_backend": bool(sim.get("chiusa_da_backend", False)),
            "timestamp_log": datetime.now(tz=utc).isoformat(timespec="seconds"),
        }
        with SIM_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"❌ Errore durante append_simulazione_chiusa per {symbol}: {e}")


@router.get("/")
def read_root():
    return {
        "status": "API Segnali di Borsa attiva",
        "commit": CURRENT_COMMIT
    }


@router.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    logging.debug(f"📩 Richiesta /analyze per {symbol.upper()}")

    try:
        symbol = symbol.upper()

        # ----------------------------------------------------------------
        #  🔹 Se è un simbolo Yahoo (whitelist) → analizza via Yahoo
        # ----------------------------------------------------------------
        if symbol in YAHOO_SYMBOL_MAP:
            try:
                y_symbol = YAHOO_SYMBOL_MAP[symbol]
                spread = 0.0
                prezzo_live = get_yahoo_last_price(y_symbol)
                if prezzo_live == 0:
                    prezzo_live = None


                # Carica dati Yahoo (gli stessi timeframe)
                df_15m = get_yahoo_df(y_symbol, "15m")
                df_1h  = get_yahoo_df(y_symbol, "1h")
                df_1d  = get_yahoo_df(y_symbol, "1d")

                if df_15m is None or df_15m.empty:
                    raise ValueError(f"Nessun dato disponibile da Yahoo per {symbol}")

                # ----------------------------------------------------------------
                # ✅ FRESHNESS CHECK (Yahoo): se l'ultima candela è troppo vecchia,
                #    NON fare analisi e NON far partire simulazioni.
                # ----------------------------------------------------------------
                try:
                    last_ts = df_15m.index[-1]

                    # df index può essere Timestamp pandas
                    if hasattr(last_ts, "to_pydatetime"):
                        last_ts = last_ts.to_pydatetime()

                    # rendi timezone-aware
                    if getattr(last_ts, "tzinfo", None) is None:
                        last_ts = last_ts.replace(tzinfo=dt_timezone.utc)

                    now_utc = datetime.now(dt_timezone.utc)
                    age_min = (now_utc - last_ts).total_seconds() / 60.0

                    if age_min > YAHOO_STALE_MAX_MIN:
                        msg = (
                            f"⏸️ Yahoo dati non aggiornati per {symbol}: "
                            f"ultima candela {last_ts.isoformat()} (≈{age_min:.1f} min fa). "
                            "Mercato chiuso/ritardo feed → analisi e simulazione bloccate."
                        )
                        logging.warning(f"[YAHOO STALE] {symbol} age_min={age_min:.1f} last_ts={last_ts}")

                        # ritorna HOLD subito, senza analizza_trend
                        return SignalResponse(
                            symbol=symbol,
                            segnale="HOLD",
                            commento=msg,
                            prezzo=0.0,          # oppure close se preferisci mostrare l'ultimo close
                            take_profit=0.0,
                            stop_loss=0.0,
                            rsi=0.0,
                            macd=0.0,
                            macd_signal=0.0,
                            atr=0.0,
                            ema7=0.0,
                            ema25=0.0,
                            ema99=0.0,
                            timeframe="15m",
                            spread=0.0,
                            motivo=msg,
                            chiusa_da_backend=False
                        )
                except Exception as e:
                    logging.warning(f"[YAHOO STALE CHECK] fallito per {symbol}: {e}")
                    

                logging.info(f"[CALL analizza_trend] YAHOO symbol={symbol} ticker={y_symbol}")
                logging.info(f"[CALL analizza_trend] BINANCE symbol={symbol}")

                # Se i dati sono freschi → ok, fai analisi
                segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(
                    df_15m, spread, None,
                    asset_name=f"{_asset_display_name(symbol)} ({symbol})",
                    asset_class="yahoo"
                )




                note = note15.split("\n") if note15 else []

                # --- conferma 1h ---
                try:
                    if df_1h is not None and not df_1h.empty:
                        segnale_1h, _, _, note1h, *_ = analizza_trend(df_1h, spread, None, asset_class="yahoo")
                        note.append(f"🕒 1h: {segnale_1h}")
                except Exception as e:
                    note.append(f"⚠️ Analisi 1h fallita: {e}")

                # --- controllo daily ---
                try:
                    from trend_logic import enrich_indicators
                    if df_1d is not None and not df_1d.empty:
                        df_1d_chk = enrich_indicators(df_1d.copy())
                        e7d  = float(df_1d_chk["EMA_7"].iloc[-1])
                        e25d = float(df_1d_chk["EMA_25"].iloc[-1])
                        e99d = float(df_1d_chk["EMA_99"].iloc[-1])
                        if e7d > e25d > e99d:
                            note.append("📅 1d: BUY")
                        elif e7d < e25d < e99d:
                            note.append("📅 1d: SELL")
                        else:
                            note.append("📅 1d: HOLD")
                except Exception as e:
                    note.append(f"⚠️ Analisi daily fallita: {e}")

                commento = "\n".join(note)

                # --- estrai ultimi indicatori ---
                close = rsi = ema7 = ema25 = ema99 = atr = macd = macd_signal = 0.0
                try:
                    src = hist if isinstance(hist, pd.DataFrame) and not hist.empty else df_15m
                    ultimo = src.iloc[-1]
                    close = round(float(ultimo.get("close", 0.0)), 4)
                    rsi = round(float(ultimo.get("RSI", 0.0)), 2)
                    ema7 = round(float(ultimo.get("EMA_7", 0.0)), 2)
                    ema25 = round(float(ultimo.get("EMA_25", 0.0)), 2)
                    ema99 = round(float(ultimo.get("EMA_99", 0.0)), 2)
                    atr = round(float(ultimo.get("ATR", 0.0)), 6)
                    macd = round(float(ultimo.get("MACD", 0.0)), 4)
                    macd_signal = round(float(ultimo.get("MACD_SIGNAL", 0.0)), 4)
                except Exception as e:
                    note.append(f"⚠️ Errore estrazione tecnici: {e}")

                prezzo_output = round(prezzo_live or close, 4)

                logging.info(f"📊 Yahoo analyze {symbol} – segnale={segnale}, prezzo={prezzo_output}")

                return SignalResponse(
                    symbol=symbol,
                    segnale=segnale,
                    commento=commento,
                    prezzo=prezzo_output,
                    take_profit=tp,
                    stop_loss=sl,
                    rsi=rsi,
                    macd=macd,
                    macd_signal=macd_signal,
                    atr=atr,
                    ema7=ema7,
                    ema25=ema25,
                    ema99=ema99,
                    timeframe="15m",
                    spread=spread,
                    motivo=commento,
                    chiusa_da_backend=False
                )

            except Exception as e:
                logging.error(f"❌ Errore analisi Yahoo per {symbol}: {e}")
                return SignalResponse(
                    symbol=symbol,
                    segnale="HOLD",
                    commento=f"Errore Yahoo: {e}",
                    prezzo=0.0,
                    take_profit=0.0,
                    stop_loss=0.0,
                    rsi=0.0,
                    macd=0.0,
                    macd_signal=0.0,
                    atr=0.0,
                    ema7=0.0,
                    ema25=0.0,
                    ema99=0.0,
                    timeframe="15m",
                    spread=0.0,
                    motivo="Errore Yahoo",
                    chiusa_da_backend=False
                )



                
        # BINANCE: accetta solo simboli completi (USDT/USDC)
        if not (symbol.endswith("USDT") or symbol.endswith("USDC")):
            logging.warning(f"[BINANCE BLOCK] symbol non USDT/USDC: {symbol}")
            return SignalResponse(
                symbol=symbol,
                segnale="HOLD",
                commento=f"Symbol non valido per Binance: {symbol}",
                prezzo=0.0,
                take_profit=0.0,
                stop_loss=0.0,
                rsi=0.0,
                macd=0.0,
                macd_signal=0.0,
                atr=0.0,
                ema7=0.0,
                ema25=0.0,
                ema99=0.0,
                timeframe="15m",
                spread=0.0,
                motivo="BINANCE BLOCK: symbol non USDT/USDC",
                chiusa_da_backend=False
            )

        with _pos_lock:
            posizione = posizioni_attive.get(symbol)
            motivo_attuale = (posizione or {}).get("motivo", "")


        # Tick size reale di Binance (serve al monitor 15m)
        try:
            tick_info = get_symbol_tick_step(symbol)

            if isinstance(tick_info, dict):
                tick_size = float(tick_info.get("tickSize", 0.0))
            elif isinstance(tick_info, (list, tuple)) and len(tick_info) >= 1:
                tick_size = float(tick_info[0])
            else:
                tick_size = float(tick_info)

        except Exception as e:
            logging.warning(f"⚠️ Impossibile ottenere tick_size per {symbol}: {e}")
            tick_size = 0.0



        # 1) Spread PRIMA (early return se eccessivo)
        book = get_bid_ask(symbol)
        bid = float(book.get("bid", 0.0))
        ask = float(book.get("ask", 0.0))
        spread = book["spread"]

        # Prezzo live (mid tra bid e ask) per allineare l'app a Binance
        prezzo_live = round(((bid + ask) / 2), 6) if bid > 0 and ask > 0 else 0.0

        logging.debug(
            f"[SPREAD] {symbol} – Spread attuale: {spread:.4f}% | "
            f"bid={bid:.6f} ask={ask:.6f} mid={prezzo_live:.6f}"
        )
        
        if spread > 5.0:
            try:
                df_15m_tmp = get_binance_df(symbol, "15m", 50)

                # calcolo prudente del close per eventuale notifica
                close_tmp = 0.0
                if not df_15m_tmp.empty:
                    close_tmp = round(float(df_15m_tmp.iloc[-1]["close"]), 6)

                
                # --- eccezione: listing pump rilevato -> notifica comunque con warning ---
                try:
                    n = len(df_15m_tmp)
                    if n >= 1:
                        u = df_15m_tmp.iloc[-1]
                        corpo   = abs(float(u["close"]) - float(u["open"]))
                        range_c = float(u["high"]) - float(u["low"])
                        body_frac = corpo / max(range_c, 1e-9)

                        # riferimenti robusti se abbiamo almeno 3 barre, altrimenti fallback
                        if n >= 3:
                            corpo_ref  = (df_15m_tmp["close"] - df_15m_tmp["open"]).iloc[:-1].abs().median()
                            volume_ref = df_15m_tmp["volume"].iloc[:-1].median() if "volume" in df_15m_tmp.columns else 0.0
                        elif n == 2:
                            p = df_15m_tmp.iloc[-2]
                            corpo_ref  = abs(float(p["close"]) - float(p["open"]))
                            volume_ref = float(p["volume"]) if "volume" in df_15m_tmp.columns else 0.0
                        else:  # n == 1 → listing appena partito
                            corpo_ref  = max(corpo, 1e-9)
                            volume_ref = float(u["volume"]) if "volume" in df_15m_tmp.columns else 0.0

                        # condizioni
                        COND_LISTING = (float(u["close"]) / max(float(u["open"]), 1e-9)) >= 2.0 and body_frac >= 0.70
                        COND_CORPO   = corpo >= 3.0 * max(corpo_ref, 1e-9)
                        COND_VOLUME  = ("volume" in df_15m_tmp.columns) and (float(u["volume"]) >= 2.5 * max(volume_ref, 1e-9))

                        # se 1 candela → usa COND_LISTING; se >=2 candele → accetta anche corpo/volume esplosi
                        trigger = (n == 1 and COND_LISTING) or (n >= 2 and (COND_LISTING or (body_frac >= 0.70 and (COND_CORPO or COND_VOLUME))))

                        if trigger:
                            prezzo_notifica = close_tmp if close_tmp > 0 else round(float(u["close"]), 6)
                            return SignalResponse(
                                symbol=symbol,
                                segnale="BUY",
                                commento="🚀 Listing/Vertical pump rilevato • ⚠️ Spread elevato: operazione ad altissimo rischio",
                                prezzo=prezzo_notifica,
                                take_profit=0.0,
                                stop_loss=0.0,
                                rsi=0.0, macd=0.0, macd_signal=0.0, atr=0.0,
                                ema7=0.0, ema25=0.0, ema99=0.0,
                                timeframe="15m",
                                spread=spread,
                                motivo="Listing pump (override spread)",
                                chiusa_da_backend=False
                            )
                except Exception:
                    pass


                if df_15m_tmp.empty:
                    raise ValueError("DataFrame vuoto")
                ultimo_tmp = df_15m_tmp.iloc[-1]
                close_tmp = round(ultimo_tmp["close"], 6)
            except Exception as e:
                logging.warning(f"⚠️ Errore nel recupero del prezzo per {symbol} con spread alto: {e}")
                close_tmp = 0.0  # fallback

            if close_tmp == 0.0:
                logging.warning(f"⛔ Nessun prezzo disponibile per {symbol} (spread alto), risposta ignorata")
                return SignalResponse(
                    symbol=symbol,
                    segnale="HOLD",
                    commento=f"⚠️ Nessun prezzo disponibile per {symbol}.",
                    prezzo=0.0,
                    take_profit=0.0,
                    stop_loss=0.0,
                    rsi=0.0,
                    macd=0.0,
                    macd_signal=0.0,
                    atr=0.0,
                    ema7=0.0,
                    ema25=0.0,
                    ema99=0.0,
                    timeframe="",
                    spread=spread,
                    motivo="Prezzo non disponibile",
                    chiusa_da_backend=False
                )

            return SignalResponse(
                symbol=symbol,
                segnale="HOLD",
                commento=f"Simulazione ignorata per {symbol} a causa di spread eccessivo.\nSpread: {spread:.2f}%",
                prezzo=close_tmp,
                take_profit=0.0,
                stop_loss=0.0,
                rsi=0.0,
                macd=0.0,
                macd_signal=0.0,
                atr=0.0,
                ema7=0.0,
                ema25=0.0,
                ema99=0.0,
                timeframe="15m",
                spread=spread,
                motivo="Spread eccessivo",
                chiusa_da_backend=False
            )

        # 2) Dati Binance (dopo lo spread) + analisi
        df_15m = get_binance_df(symbol, "15m", 300)
        df_1h  = get_binance_df(symbol, "1h", 300)
        df_1d  = get_binance_df(symbol, "1d", 300)
        df_1m  = get_binance_df(symbol, "1m", 100)

        try:
            segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(
                df_15m, spread, df_1m,
                asset_name=f"{_asset_display_name(symbol)} ({symbol})"
            )

        except KeyError as e:
            # 👉 Non facciamo più fallire /analyze: trasformiamo in HOLD di sicurezza
            cols = list(df_15m.columns) if isinstance(df_15m, pd.DataFrame) else []
            logging.warning(
                f"[analizza_trend 15m] KeyError {e!r} per {symbol} - "
                f"colonne df_15m: {cols} (len={len(df_15m) if hasattr(df_15m, '__len__') else 'NA'})"
            )
            segnale, hist, distanza_ema, note15, tp, sl, supporto = (
                "HOLD",
                df_15m if isinstance(df_15m, pd.DataFrame) else pd.DataFrame(),
                0.0,
                "Analisi 15m non disponibile (KeyError su dati)",
                0.0,
                0.0,
                None,
            )
        except Exception as e:
            cols = list(df_15m.columns) if isinstance(df_15m, pd.DataFrame) else []
            logging.error(
                f"❌ Errore imprevisto in analizza_trend 15m per {symbol}: {e} - "
                f"colonne df_15m: {cols} (len={len(df_15m) if hasattr(df_15m, '__len__') else 'NA'})"
            )
            segnale, hist, distanza_ema, note15, tp, sl, supporto = (
                "HOLD",
                df_15m if isinstance(df_15m, pd.DataFrame) else pd.DataFrame(),
                0.0,
                "Analisi 15m non disponibile (errore interno)",
                0.0,
                0.0,
                None,
            )

        note = note15.split("\n") if note15 else []



        # 3) Gestione posizione già attiva (UNA SOLA VOLTA QUI)
        if posizione:
            logging.info(
                f"⏳ Simulazione già attiva su {symbol} – tipo: "
                f"{posizione['tipo']} @ {posizione['entry']}$"
            )

            # se l’analisi ha “annullato” il segnale → marca la simulazione e restituisci HOLD annotato
            if segnale == "HOLD" and note15 and "Segnale annullato" in note15:
                with _pos_lock:
                    posizione["tipo"] = "HOLD"
                    posizione["esito"] = "Annullata"
                    posizione["chiusa_da_backend"] = True
                    posizione["motivo"] = note15
                return SignalResponse(
                    symbol=symbol,
                    segnale="HOLD",
                    commento=note15,
                    prezzo=posizione["entry"],
                    take_profit=posizione["tp"],
                    stop_loss=posizione["sl"],
                    rsi=0.0, macd=0.0, macd_signal=0.0, atr=0.0,
                    ema7=0.0, ema25=0.0, ema99=0.0,
                    timeframe="15m",
                    spread=posizione.get("spread", 0.0),
                    motivo=note15,
                    chiusa_da_backend=True
                )

            # altrimenti ritorna lo stato della simulazione attiva
            return SignalResponse(
                symbol=symbol,
                segnale="HOLD",
                commento=(
                    f"\u23f3 Simulazione gi\u00e0 attiva su {symbol} - tipo: {posizione['tipo']} @ {posizione['entry']}$\n"
                    f"🎯 TP: {posizione['tp']} | 🛡 SL: {posizione['sl']}"
                ),
                prezzo=posizione["entry"],
                take_profit=posizione["tp"],
                stop_loss=posizione["sl"],
                rsi=0.0, macd=0.0, macd_signal=0.0, atr=0.0,
                ema7=0.0, ema25=0.0, ema99=0.0,
                timeframe="15m",
                spread=posizione.get("spread", 0.0),
                motivo=motivo_attuale,
                chiusa_da_backend=posizione.get("chiusa_da_backend", False)
            )

        # 4) Estrai sempre i tecnici più recenti (anche se HOLD)
        close = rsi = ema7 = ema25 = ema99 = atr = macd = macd_signal = 0.0

        try:
            # scegli la sorgente: hist se valido, altrimenti df_15m
            src = hist if isinstance(hist, pd.DataFrame) and not hist.empty and "close" in hist.columns else df_15m

            if not isinstance(src, pd.DataFrame) or src.empty:
                raise ValueError("DataFrame tecnico vuoto")

            if "close" not in src.columns:
                raise KeyError("colonna 'close' mancante nel DataFrame tecnico")

            # recupera l’ultima riga e applica valori fallback per sicurezza
            ultimo = src.iloc[-1]
            close = round(float(ultimo.get("close", 0.0)), 4)
            rsi = round(float(ultimo.get("RSI", 0.0)), 2)
            ema7 = round(float(ultimo.get("EMA_7", 0.0)), 2)
            ema25 = round(float(ultimo.get("EMA_25", 0.0)), 2)
            ema99 = round(float(ultimo.get("EMA_99", 0.0)), 2)
            atr = round(float(ultimo.get("ATR", 0.0)), 6)
            macd = round(float(ultimo.get("MACD", 0.0)), 4)
            macd_signal = round(float(ultimo.get("MACD_SIGNAL", 0.0)), 4)

        except Exception as e:
            logging.warning(f"⚠️ Errore nell’estrazione dei dati tecnici per {symbol}: {e}")
            close = rsi = ema7 = ema25 = ema99 = atr = macd = macd_signal = 0.0



        # Prezzo da esporre all'app:
        # – se disponibile, usa il mid live bid/ask
        # – altrimenti fallback al close 15m
        prezzo_output = round(prezzo_live, 4) if "prezzo_live" in locals() and prezzo_live > 0 else close


        # 5) Logging timeframe e analisi di conferma
        logging.debug(f"[BINANCE] {symbol} – 15m: {len(df_15m)} | 1h: {len(df_1h)} | 1d: {len(df_1d)}")
        logging.debug(f"[15m] {symbol} – Segnale: {segnale}, Note: {note15.replace(chr(10), ' | ')}")

        # supporto può essere None → niente formato .6f diretto
        try:
            dist_val = float(distanza_ema) if distanza_ema is not None else 0.0
            tp_val   = float(tp) if tp is not None else 0.0
            sl_val   = float(sl) if sl is not None else 0.0
            sup_val  = float(supporto) if isinstance(supporto, (int, float)) else 0.0
            logging.debug(
                "[15m DETTAGLI] distEMA={:.6f}, TP={:.6f}, SL={:.6f}, supporto={:.6f}".format(
                    dist_val, tp_val, sl_val, sup_val
                )
            )
        except Exception as e_log:
            logging.debug(f"[15m DETTAGLI] impossibile loggare TP/SL/supporto: {e_log}")


        # 1h: ok usare ancora analizza_trend come conferma "soft"
        try:
            segnale_1h, hist_1h, _, note1h, *_ = analizza_trend(df_1h, spread)
        except KeyError as e:
            logging.error(
                f"[analizza_trend 1h] KeyError {e} per {symbol} – colonne df_1h: {list(df_1h.columns)}"
            )
            from trend_logic import enrich_indicators
            hist_1h = enrich_indicators(df_1h.copy()) if isinstance(df_1h, pd.DataFrame) and not df_1h.empty else df_1h
            segnale_1h = "HOLD"
            note1h = f"Errore analisi 1h: {e}"

        logging.debug(f"[1h] {symbol} – Segnale: {segnale_1h}")


        # 1d: controllo RIGOROSO solo su EMA (niente recupero/RSI/MACD)
        try:
            from trend_logic import enrich_indicators  # se non già importato altrove
            df_1d_chk = enrich_indicators(df_1d.copy())
            e7d  = float(df_1d_chk["EMA_7"].iloc[-1])
            e25d = float(df_1d_chk["EMA_25"].iloc[-1])
            e99d = float(df_1d_chk["EMA_99"].iloc[-1])

            daily_ok_buy  = (e7d > e25d > e99d)
            daily_ok_sell = (e7d < e25d < e99d)
            daily_state   = "BUY" if daily_ok_buy else ("SELL" if daily_ok_sell else "HOLD")
            logging.debug(f"[1d STRICT] {symbol} – EMA7={e7d:.4f} EMA25={e25d:.4f} EMA99={e99d:.4f} -> {daily_state}")
        except Exception as _err:
            logging.warning(f"[daily-check] impossibile validare 1D per {symbol}: {_err}")
            daily_state = "NA"  # neutro


        # 6) Conferma / descrizione 1h
        # ---------------------------------------------
        try:
            e7h  = float(hist_1h["EMA_7"].iloc[-1])
            e25h = float(hist_1h["EMA_25"].iloc[-1])
            e99h = float(hist_1h["EMA_99"].iloc[-1])

            trend_up_1h   = e7h > e25h > e99h
            trend_down_1h = e7h < e25h < e99h
        except Exception as e:
            logging.warning(f"⚠️ Errore EMA 1h: {e}")
            trend_up_1h = trend_down_1h = False

        # Caso BUY/SELL → vera conferma multi-timeframe
        if segnale in ("BUY", "SELL"):
            ema_confirm = (
                (segnale == "BUY"  and trend_up_1h) or
                (segnale == "SELL" and trend_down_1h)
            )

            if ema_confirm:
                note.append("🧭 1h✓ (EMA)")
            else:
                # Se le EMA non confermano, usa la conferma 'soft' con analizza_trend su 1h
                if segnale != segnale_1h:
                    logging.info(f"🧭 {symbol} – 1h NON conferma {segnale} (1h = {segnale_1h})")
                    try:
                        ultimo_1h = hist_1h.iloc[-1]
                        macd_1h   = float(ultimo_1h['MACD'])
                        signal_1h = float(ultimo_1h['MACD_SIGNAL'])
                        rsi_1h    = float(ultimo_1h['RSI'])

                        if segnale == "SELL" and macd_1h < 0 and (macd_1h - signal_1h) < 0.005 and rsi_1h < 45:
                            note.append("ℹ️ 1h non confermato, ma MACD/RSI coerenti con SELL")
                        elif segnale == "BUY" and macd_1h > 0 and (macd_1h - signal_1h) > -0.005 and rsi_1h > 50:
                            note.append("ℹ️ 1h non confermato, ma MACD/RSI coerenti con BUY")
                        else:
                            note.append(f"⚠️ {segnale} non confermato su 1h")

                        # qui il conteggio è coerente col tipo di segnale
                        trend_1h = conta_candele_trend(hist_1h, rialzista=(segnale == "BUY"))
                        if trend_1h < 2:
                            note.append(f"⚠️ Trend su 1h debole ({trend_1h} candele)")
                    except Exception as e:
                        logging.warning(f"⚠️ Errore dati 1h: {e}")
                else:
                    note.append("🧭 1h✓")

        # Caso HOLD → non chiediamo "conferma", ma descriviamo onestamente l'1h
        else:  # segnale == "HOLD"
            try:
                if trend_up_1h:
                    c_up = conta_candele_trend(hist_1h, rialzista=True)
                    note.append(f"📡 1h rialzista ({c_up} candele)")
                elif trend_down_1h:
                    c_down = conta_candele_trend(hist_1h, rialzista=False)
                    note.append(f"📡 1h ribassista ({c_down} candele)")
                else:
                    note.append("📡 1h laterale / incerto")
            except Exception as e:
                logging.warning(f"⚠️ Errore descrizione trend 1h: {e}")



        # 7) Note 1d e possibile apertura simulazione
        if segnale in ["BUY", "SELL"]:
            if daily_state == "NA":
                note.append("📅 1d - Check fallito")  # dati non disponibili / check fallito
            else:
                ok_daily = (
                    (segnale == "BUY"  and daily_state == "BUY") or
                    (segnale == "SELL" and daily_state == "SELL")
                )
                if ok_daily:
                    note.append("📅 1d✓")
                else:
                    note.append(f"⚠️ Daily in conflitto ({daily_state})")

        # (REMOVED) Denominazione ASSET già inclusa in trend_logic nella riga "📊 Trend score ... | 🛈 ...".
        # Evita duplicati nelle notifiche/card.

        
        # testo completo della notifica (per app + per log simulazioni)
        commento = "\n".join(note) if note else "Nessuna nota"


        # apertura simulazione SOLO se c'è un vero segnale
        if segnale in ["BUY", "SELL"]:
            logging.info(
                f"✅ Nuova simulazione {segnale} per {symbol} @ {close}$ – "
                f"TP: {tp}, SL: {sl}, spread: {spread:.2f}%, tick_size={tick_size}"
            )
            with _pos_lock:
                posizioni_attive[symbol] = {
                    "tipo": segnale,
                    "entry": close,
                    "tp": tp,
                    "sl": sl,
                    "spread": spread,
                    "tick_size": tick_size,
                    "chiusa_da_backend": False,
                    "motivo": " | ".join(note),
                    "note_notifica": commento,  
                }

        return SignalResponse(
            symbol=symbol,
            segnale=segnale,
            commento=commento,
            prezzo=prezzo_output,
            take_profit=tp,
            stop_loss=sl,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            atr=atr,
            ema7=ema7,
            ema25=ema25,
            ema99=ema99,
            timeframe="15m",
            spread=spread,
            motivo=" | ".join(note),
            chiusa_da_backend=False
        )


    except Exception as e:
        logging.error(f"❌ Errore durante /analyze per {symbol}: {e}")
        try:
            df_15m = get_binance_df(symbol, "15m", 50)
            close = round(df_15m.iloc[-1]["close"], 6)
        except Exception as e2:
            logging.warning(f"⚠️ Fallito anche il recupero prezzo fallback: {e2}")
            close = 0.0

        return SignalResponse(
            symbol=symbol,
            segnale="HOLD",
            commento=(
                f"Errore durante l'analisi di {symbol}.\n"
                f"Tentativo di recupero prezzo: {'Riuscito' if close > 0 else 'Fallito'}\n"
                f"Errore originale: {e}"
            ),
            prezzo=close,
            take_profit=0.0,
            stop_loss=0.0,
            rsi=0.0,
            macd=0.0,
            macd_signal=0.0,
            atr=0.0,
            ema7=0.0,
            ema25=0.0,
            ema99=0.0,
            timeframe="",
            spread=0.0,
            motivo="Errore interno",
            chiusa_da_backend=False
        )

        # <-- PAUSA -->

# ===== Avvio scanner Top Movers (evita doppio avvio) =====
_SCANNER_STARTED = False

def _ensure_scanner():
    global _SCANNER_STARTED
    if _SCANNER_STARTED:
        return
    try:
        # puoi cambiare gli intervalli/soglie come preferisci
        start_top_mover_scanner(
            analyze_fn=analyze,
            interval_sec=30,
            gain_threshold_normale=0.10,
            gain_threshold_listing=1.00,
            quote_suffix=("USDC","USDT"),
            top_n_24h=80,
            cooldown_sec=90,
        )

        _SCANNER_STARTED = True
        logging.info("✅ Top Mover Scanner avviato da routes.py")
    except Exception as e:
        logging.error(f"❌ Impossibile avviare lo scanner: {e}")

_ensure_scanner()
# =========================================================


@router.get("/price")
def get_price(symbol: str):
    
    start = time.time()
    symbol = symbol.upper()

    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        response = requests.get(url, timeout=3)
        data = response.json()

        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        if bid <= 0 or ask <= 0:
            raise ValueError(f"Prezzo non valido: bid={bid}, ask={ask}")

        spread = (ask - bid) / ((ask + bid) / 2) * 100
        prezzo = round((bid + ask) / 2, 4)

        elapsed = round(time.time() - start, 3)
        with _pos_lock:
            pos = posizioni_attive.get(symbol, {})       # <-- lookup una sola volta

        return {
            "symbol": symbol,
            "prezzo": prezzo,
            "spread": round(spread, 4),
            "tempo": elapsed,
            "motivo": pos.get("motivo", ""),
            "takeProfit": pos.get("tp", 0.0),
            "stopLoss":  pos.get("sl", 0.0),
            "chiusaDaBackend": pos.get("chiusa_da_backend", False)  # <-- nuovo campo
        }

    except Exception as e:
        logging.error(f"❌ Errore durante l'analisi di {symbol.upper()}: {e}")

        elapsed = round(time.time() - start, 3)
        return {
            "symbol": symbol,
            "prezzo": 0.0,
            "spread": 0.0,
            "errore": str(e),
            "tempo": elapsed
        }

# Cache e log filtri
_hot_cache = {"time": 0, "data": []}
_filtro_log = {
    "totali": 0,
    "atr": 0,
    "ema_flat": 0,
    "volume_basso": 0,
    "prezzo_piattissimo": 0,
    "macd_rsi_neutri": 0
}
MODALITA_TEST = True

@router.get("/hotassets")
def hot_assets():
    now = time.time()
    if (now - _hot_cache["time"]) < 180:
        return _hot_cache["data"]

    symbols = get_best_symbols(limit=120)
    symbols = _augment_with_whitelist(symbols)

    risultati = []

    # --- Soglie radar HOT vs trend_logic (imbuto crescente) ---
    if MODALITA_TEST:
        # Radar più permissivo di trend_logic:
        # trend_logic TEST: volume_soglia=120, atr_minimo=0.0005, distanza_minima=0.0008,
        #                   macd_rsi_range=(45,55), macd_signal_threshold=0.00015
        volume_soglia = 50          # quantità scambiata minima per non essere "morta"
        atr_minimo = 0.0003         # un po' sotto 0.0005 → passa roba più "timida"
        distanza_minima = 0.0004    # metà della soglia di analizza_trend
        macd_rsi_range = (47, 53)   # zona neutra più stretta del (45,55)
        macd_signal_threshold = 0.00010  # MACD davvero incollato al segnale
        distanza_flat_max = 0.0012       # EMA 7–99 quasi sovrapposte = mercato piatto
    else:
        # Radar PROD, sempre più permissivo di trend_logic PROD:
        # trend_logic PROD: volume_soglia=300, atr_minimo=0.0009, distanza_minima=0.0012,
        #                   macd_rsi_range=(47,53), macd_signal_threshold=0.0006
        volume_soglia = 200
        atr_minimo = 0.0007
        distanza_minima = 0.0008
        macd_rsi_range = (48, 52)   # ancora più centrale
        macd_signal_threshold = 0.00040
        distanza_flat_max = 0.0010


    for symbol in symbols:
        try:
            # --- Flag whitelist per simbolo corrente ---
            base = symbol[:-4] if (symbol.endswith("USDT") or symbol.endswith("USDC")) else symbol
            is_whitelist = base in HOT_WHITELIST_BASE
            logging.debug(f"[HOTASSETS] consider {symbol} base={base} whitelist={is_whitelist}")

            added = False  # diventa True quando il symbol entra in risultati
            

            df = get_binance_df(symbol, "15m", 100)

            # --- Fast-path per LISTING PUMP anche con storico corto (<40 barre) ---
            if len(df) >= 2 and len(df) < 40:
                ultimo = df.iloc[-1]
                prev   = df.iloc[-2]

                # spike verticale su 1 candela
                corpo     = abs(ultimo["close"] - ultimo["open"])
                range_c   = ultimo["high"] - ultimo["low"]
                body_frac = corpo / max(range_c, 1e-9)

                # riferimenti minimi robusti
                corpo_ref  = (df["close"] - df["open"]).iloc[:-1].abs().median() if len(df) > 3 else abs(prev["close"] - prev["open"])
                volume_ref = df["volume"].iloc[:-1].median() if len(df) > 3 else max(prev["volume"], 1e-9)

                # criteri listing pump (tolleranti, niente ATR/EMA richiesti)
                COND_GAIN   = (ultimo["close"] / max(ultimo["open"], 1e-9)) >= 2.0    # +100% in barra
                COND_BODY   = body_frac >= 0.70                                      # candela "piena"
                COND_CORPO  = corpo >= 3.0 * max(corpo_ref, 1e-9)                    # corpo enorme
                COND_VOLUME = ultimo["volume"] >= 2.5 * max(volume_ref, 1e-9)        # volume esploso

                if (COND_GAIN and COND_BODY and (COND_CORPO or COND_VOLUME)):
                    trend_pump = "BUY" if ultimo["close"] >= ultimo["open"] else "SELL"
                    risultati.append({
                        "symbol": symbol,
                        "segnali": 1,
                        "trend": trend_pump,
                        "rsi": None,
                        "ema7": 0.0, "ema25": 0.0, "ema99": 0.0,
                        "prezzo": round(float(ultimo["close"]), 4),
                        "candele_trend": 1,
                        "note": f"🚀 Listing pump (storico corto) • 🛈 {_asset_display_name(symbol)}"
                    })
                    added = True
                    continue  # salta i filtri classici e marca come hot

            # --- filtro storico minimo: per whitelist salta solo se df è proprio vuoto ---
            if df.empty or (len(df) < 40 and not is_whitelist):
                continue

            _filtro_log["totali"] += 1

            prezzo = float(df["close"].iloc[-1])
            volume_quote_medio = float((df["volume"].tail(20) * df["close"].tail(20)).mean())
            soglia_quote = 50_000 if MODALITA_TEST else 250_000  # es.: 50k$/candela su 15m

            if pd.isna(volume_quote_medio) or volume_quote_medio < soglia_quote:
                _filtro_log["volume_basso"] += 1
                if not is_whitelist:
                    continue  # i whitelist non vengono scartati per volume basso

            df["EMA_7"] = df["close"].ewm(span=7).mean()
            df["EMA_25"] = df["close"].ewm(span=25).mean()
            df["EMA_99"] = df["close"].ewm(span=99).mean()
            df["RSI"] = calcola_rsi(df["close"])
            df["MACD"], df["MACD_SIGNAL"] = calcola_macd(df["close"])
            df["ATR"] = calcola_atr(df)

            ema7 = df["EMA_7"].iloc[-1]
            ema25 = df["EMA_25"].iloc[-1]
            ema99 = df["EMA_99"].iloc[-1]
            rsi = df["RSI"].iloc[-1]
            macd = df["MACD"].iloc[-1]
            macd_signal = df["MACD_SIGNAL"].iloc[-1]
            atr = df["ATR"].iloc[-1]
            prezzo = df["close"].iloc[-1]

            if (prezzo <= 0 or pd.isna(atr) or atr < atr_minimo) and not is_whitelist:
                _filtro_log["atr"] += 1
                continue

            # --- Pump fast-track: non blocca le hot durante spike ---
            ultimo = df.iloc[-1]
            corpo_candela = abs(ultimo["close"] - ultimo["open"])
            range_candela = ultimo["high"] - ultimo["low"]
            corpo_medio   = (df["close"] - df["open"]).iloc[-21:-1].abs().mean()
            volume_medio_20  = df["volume"].iloc[-21:-1].mean()

            upper_wick = ultimo["high"] - max(ultimo["open"], ultimo["close"])
            lower_wick = min(ultimo["open"], ultimo["close"]) - ultimo["low"]
            wick_ratio = (upper_wick + lower_wick) / max(range_candela, 1e-9)

            cond_range  = range_candela > 2.0 * atr
            cond_corpo  = corpo_candela > 3.0 * max(corpo_medio, 1e-9)
            cond_volume = df["volume"].iloc[-1] > 2.0 * max(volume_medio_20, 1e-9)
            cond_wick   = wick_ratio < 0.35

            if (cond_corpo and cond_volume) or (cond_range and cond_volume and cond_wick):
                trend_pump = "BUY" if ultimo["close"] >= ultimo["open"] else "SELL"
                candele_trend = conta_candele_trend(df, rialzista=(trend_pump == "BUY"))
                logging.info(f"[HOTASSETS BINANCE ADD] {symbol} reason=pump trend={trend_pump}")
                risultati.append({
                    "symbol": symbol,
                    "segnali": 1,
                    "trend": trend_pump,
                    "rsi": round(rsi, 2),
                    "ema7": round(ema7, 2),
                    "ema25": round(ema25, 2),
                    "ema99": round(ema99, 2),
                    "prezzo": round(prezzo, 4),
                    "candele_trend": candele_trend,
                    "note": f"🛈 {_asset_display_name(symbol)}"
                })
                added = True
                continue  # salta i filtri successivi: la coin è "hot" per pump


            distanza_relativa = abs(ema7 - ema99) / max(abs(ema99), 1e-9)
            if distanza_relativa < distanza_minima and prezzo < 1000 and not is_whitelist:
                _filtro_log["ema_flat"] += 1
                continue

            oscillazione = df["close"].diff().abs().tail(10).sum()
            if oscillazione < 0.001 and prezzo < 50 and not is_whitelist:
                _filtro_log["prezzo_piattissimo"] += 1
                continue

            # --- Kill switch solo per asset DAVVERO piatti (MACD/RSI neutri + EMA schiacciate) ---
            delta_macd = abs(macd - macd_signal)
            if (
                delta_macd < macd_signal_threshold           # MACD quasi sovrapposto al segnale
                and macd_rsi_range[0] <= rsi <= macd_rsi_range[1]  # RSI molto centrale
                and distanza_relativa < distanza_flat_max    # EMA 7–99 troppo vicine
            ):
                if not is_whitelist:
                    _filtro_log["macd_rsi_neutri"] += 1
                    continue


            recenti_rialzo = all(df["EMA_7"].iloc[-i] > df["EMA_25"].iloc[-i] > df["EMA_99"].iloc[-i] for i in range(1, 4))
            recenti_ribasso = all(df["EMA_7"].iloc[-i] < df["EMA_25"].iloc[-i] < df["EMA_99"].iloc[-i] for i in range(1, 4))

            trend_buy = recenti_rialzo and rsi > 50 and macd > macd_signal
            trend_sell = recenti_ribasso and rsi < 50 and macd < macd_signal

            # Permissivo per presegnali: MACD > signal o vicino
            macd_ok = macd > macd_signal or abs(macd - macd_signal) < 0.01

            presegnale_buy = (
                df["EMA_7"].iloc[-2] < df["EMA_25"].iloc[-2]
                and ema7 > ema25
                and ema25 < ema99
                and distanza_relativa < 0.015
                and rsi > 50
                and macd_ok
            )

            presegnale_sell = (
                df["EMA_7"].iloc[-2] > df["EMA_25"].iloc[-2]
                and ema7 < ema25
                and ema25 > ema99
                and distanza_relativa < 0.015
                and rsi < 50
                and (macd < macd_signal or abs(macd - macd_signal) < 0.01)
            )

            if trend_buy or trend_sell or presegnale_buy or presegnale_sell:
                segnale = "BUY" if (trend_buy or presegnale_buy) else "SELL"
                candele_trend = conta_candele_trend(df, rialzista=(segnale == "BUY"))
                risultati.append({
                    "symbol": symbol,
                    "segnali": 1,
                    "trend": segnale,
                    "rsi": round(rsi, 2),
                    "ema7": round(ema7, 2),
                    "ema25": round(ema25, 2),
                    "ema99": round(ema99, 2),
                    "prezzo": round(prezzo, 4),
                    "candele_trend": candele_trend,
                    "note": f"🛈 {_asset_display_name(symbol)}"
                })
                added = True

            # --- Fallback: i simboli in whitelist devono comparire comunque ---
            if is_whitelist and not added:
                candele_trend = conta_candele_trend(df, rialzista=True)
                logging.info(f"[HOTASSETS BINANCE ADD] {symbol} reason=whitelist_fallback trend=HOLD")
                risultati.append({
                    "symbol": symbol,
                    "segnali": 0,
                    "trend": "HOLD",
                    "rsi": round(float(rsi), 2) if not pd.isna(rsi) else None,
                    "ema7": round(float(ema7), 2),
                    "ema25": round(float(ema25), 2),
                    "ema99": round(float(ema99), 2),
                    "prezzo": round(float(prezzo), 4),
                    "candele_trend": candele_trend,
                    "note": f"🎯 Whitelist asset (monitoraggio) • 🛈 {_asset_display_name(symbol)}"
                })

        except Exception:
            continue


    # --- Aggiunta asset YAHOO Finance in coda alla lista HOT ---
    # Scansiona TUTTI i simboli supportati da YAHOO_SYMBOL_MAP (macro + indici + azioni),
    # con priorità a YAHOO_HOT_LIST.
    yahoo_scan_list = []
    try:
        # 1) prima la hot list (ordine preservato)
        yahoo_scan_list.extend([s for s in YAHOO_HOT_LIST if s in YAHOO_SYMBOL_MAP])

        # 2) poi tutti gli altri in mappa (senza duplicati)
        for s in YAHOO_SYMBOL_MAP.keys():
            if s not in yahoo_scan_list:
                yahoo_scan_list.append(s)
    except Exception:
        yahoo_scan_list = list(YAHOO_HOT_LIST)

    for y_sym in yahoo_scan_list:
        try:
            # piccolo throttle (molto più leggero di 1.5s)
            time.sleep(0.25)

            # Mappa simbolo “logico” -> ticker Yahoo reale
            y_ticker = YAHOO_SYMBOL_MAP.get(y_sym, y_sym)

            logging.debug(f"[YAHOO hotassets] scanning {y_sym} -> {y_ticker}")

            # Dati 15m da Yahoo, compatibili con trend_logic
            df_y = get_yahoo_df(y_ticker, interval="15m")
            if df_y is None or df_y.empty:
                logging.warning(f"[YAHOO hotassets] Nessun dato per {y_sym} ({y_ticker})")
                continue

            # ✅ FRESHNESS CHECK (Yahoo) anche in /hotassets: se è stale -> skip
            try:
                last_ts = df_y.index[-1]
                if hasattr(last_ts, "to_pydatetime"):
                    last_ts = last_ts.to_pydatetime()
                if getattr(last_ts, "tzinfo", None) is None:
                    last_ts = last_ts.replace(tzinfo=dt_timezone.utc)

                now_utc = datetime.now(dt_timezone.utc)
                age_min = (now_utc - last_ts).total_seconds() / 60.0

                if age_min > YAHOO_STALE_MAX_MIN:
                    logging.debug(f"[YAHOO hotassets] skip STALE {y_sym} age_min={age_min:.1f} last_ts={last_ts}")
                    continue
            except Exception as e:
                logging.debug(f"[YAHOO hotassets] freshness check fail {y_sym}: {e}")



            
            # Analisi con la stessa funzione del backend crypto
            try:
                logging.info(f"[CALL analizza_trend] HOTASSETS YAHOO symbol={y_sym} ticker={y_ticker}")
                segnale_y, hist_y, _, note15_y, *_ = analizza_trend(
                    df_y, 0.0, None,
                    asset_name=f"{_asset_display_name(y_sym)} ({y_sym})",
                    asset_class="yahoo"
                )
            except Exception as e_y:
                logging.warning(f"[YAHOO hotassets] analizza_trend fallita per {y_sym}: {e_y}")
                continue

            # Sorgente dati per gli ultimi indicatori
            src_y = hist_y if isinstance(hist_y, pd.DataFrame) and not hist_y.empty else df_y
            if src_y is None or src_y.empty:
                continue

            ultimo_y = src_y.iloc[-1]

            prezzo_y = float(ultimo_y.get("close", 0.0))
            ema7_y   = float(ultimo_y.get("EMA_7", 0.0))  if "EMA_7" in src_y.columns  else 0.0
            ema25_y  = float(ultimo_y.get("EMA_25", 0.0)) if "EMA_25" in src_y.columns else 0.0
            ema99_y  = float(ultimo_y.get("EMA_99", 0.0)) if "EMA_99" in src_y.columns else 0.0
            rsi_y    = float(ultimo_y.get("RSI", 0.0))    if "RSI" in src_y.columns    else 0.0

            candele_trend_y = conta_candele_trend(src_y, rialzista=(segnale_y == "BUY"))

            logging.info(f"[HOTASSETS YAHOO ADD] {y_sym} trend={segnale_y}")
            risultati.append({
                "symbol": y_sym,
                "segnali": 1 if segnale_y in ("BUY", "SELL") else 0,
                "trend": segnale_y if segnale_y in ("BUY", "SELL") else "HOLD",
                "rsi": round(rsi_y, 2),
                "ema7": round(ema7_y, 2),
                "ema25": round(ema25_y, 2),
                "ema99": round(ema99_y, 2),
                "prezzo": round(prezzo_y, 4),
                "candele_trend": candele_trend_y,
                "note": f"🌍 Yahoo Finance • 🛈 {_asset_display_name(y_sym)}",
            })

        except Exception as e_yahoo:
            logging.warning(f"[YAHOO hotassets] Errore per {y_sym}: {e_yahoo}")
            continue




    _hot_cache["time"] = now
    _hot_cache["data"] = risultati
    return risultati




# ------------------------------------------------------------------
# Flag globale: mettilo a True quando vorrai riattivare la gestione
# ------------------------------------------------------------------
GESTIONE_ATTIVA        = True     # lascia False se vuoi solo log, True per aggiornare TP
CHECK_INTERVAL_SEC     = 60       # ogni minuto
TIMEFRAME_TREND        = "15m"
CANDLE_LIMIT           = 50       # ~12 h
EMA_DIST_MIN_PERC      = 0.0008   # 0,08 %
EMA_DIST_MAX_PERC      = 0.0030   # 0,30 %
SL_BUFFER_PERC         = 0.25     # retracement: se prezzo torna entro il 25 % fra entry e SL
TP_TRAIL_FACTOR        = 1.20     # TP = entry + (prezzo-entry)*1.20  (BUY); viceversa SELL


def verifica_posizioni_attive():
    counter = 0  # per chiamare il cleanup ogni N cicli

    # ===== PROTEZIONE MASTER LOOP =====
    while True:
        try:
            time.sleep(CHECK_INTERVAL_SEC)

            with _pos_lock:
                _keys_snapshot = list(posizioni_attive.keys())

            for symbol in _keys_snapshot:
                with _pos_lock:
                    sim = posizioni_attive.get(symbol)
                if sim is None or sim.get("esito") in ("Profitto", "Perdita"):
                    continue

                tipo = sim["tipo"]            # "BUY" | "SELL"
                try:
                    # ===== prezzi live bid/ask =====
                    book      = get_bid_ask(symbol)
                    entry     = float(sim["entry"])
                    tp        = float(sim["tp"])
                    sl        = float(sim["sl"])

                    # prezzo live usato solo per progress/stato
                    prezzo_live = float(book["ask"] if tipo == "BUY" else book["bid"])
                    distanza_entry = abs(prezzo_live - entry)
                    progresso = abs(prezzo_live - entry) / abs(tp - entry) if tp != entry else 0.0

                    # ===== ultima candela timeframe trend =====
                    df = get_binance_df(symbol, TIMEFRAME_TREND, limit=CANDLE_LIMIT)
                    if df.empty:
                        sim["motivo"] = f"⚠️ Dati insufficienti ({TIMEFRAME_TREND})"
                        continue

                    last = df.iloc[-1]
                    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])

                    # ===== EMA per stato/trailing =====
                    df["EMA_7"]  = df["close"].ewm(span=7).mean()
                    df["EMA_25"] = df["close"].ewm(span=25).mean()
                    ema7  = float(df["EMA_7"].iloc[-1])
                    ema25 = float(df["EMA_25"].iloc[-1])
                    dist_ema = abs(ema7 - ema25)

                    in_range = (EMA_DIST_MIN_PERC * c) <= dist_ema <= (EMA_DIST_MAX_PERC * c)
                    trend_ok = (
                        (tipo == "BUY"  and c >= ema7 and ema7 > ema25 and in_range) or
                        (tipo == "SELL" and c <= ema7 and ema7 < ema25 and in_range)
                    )

                    # ===== verifiche TP/SL su candela corrente =====
                    TICK = float(sim.get("tick_size", 0.0)) if isinstance(sim.get("tick_size", 0.0), (int, float)) else 0.0
                    eps = TICK

                    fill_price = None
                    exit_reason = None

                    if tipo == "BUY":
                        if sl and (l <= sl + eps):
                            fill_price = sl
                            exit_reason = "SL"
                        elif tp and (h >= tp - eps):
                            fill_price = tp
                            exit_reason = "TP"
                    else:  # SELL
                        if sl and (h >= sl - eps):
                            fill_price = sl
                            exit_reason = "SL"
                        elif tp and (l <= tp + eps):
                            fill_price = tp
                            exit_reason = "TP"

                    if fill_price is not None:
                        with _pos_lock:
                            sim["prezzo_chiusura"]  = round(float(fill_price), 10)
                            sim["chiusa_da_backend"] = True
                            sim["ora_chiusura"]     = datetime.now(tz=utc).isoformat(timespec="seconds")
                            if tipo == "BUY":
                                var_pct = (fill_price - entry) / entry * 100.0
                            else:
                                var_pct = (entry - fill_price) / entry * 100.0
                            sim["variazione_pct"] = round(float(var_pct), 4)
                            sim["esito"] = "Profitto" if exit_reason == "TP" else "Perdita"
                            sim["motivo"] = ("🎯 TP colpito" if exit_reason == "TP" else "🛡️ SL colpito")

                            logging.info(
                                f"🔚 CLOSE {symbol} {tipo}: entry={entry:.6f} fill={fill_price:.6f} "
                                f"tp={tp:.6f} sl={sl:.6f} esito={sim['esito']} var={sim['variazione_pct']:.3f}%"
                            )
                            append_simulazione_chiusa(symbol, sim)

                        continue

                    # ===== retracement verso SL? =====
                    if sl != 0:
                        if tipo == "BUY":
                            verso_sl = c <= entry - SL_BUFFER_PERC * abs(entry - sl)
                        else:
                            verso_sl = c >= entry + SL_BUFFER_PERC * abs(entry - sl)
                    else:
                        verso_sl = False

                    # ===== trailing TP dinamico =====
                    if GESTIONE_ATTIVA and trend_ok and not verso_sl:
                        with _pos_lock:
                            sim.setdefault("tp_esteso", 1)
                        if tipo == "BUY":
                            nuovo_tp = round(entry + distanza_entry * TP_TRAIL_FACTOR, 6)
                            if nuovo_tp > tp:
                                with _pos_lock:
                                    sim["tp"] = nuovo_tp
                                    sim["motivo"] = "📈 TP aggiornato (trend 15m)"
                        else:
                            nuovo_tp = round(entry - distanza_entry * TP_TRAIL_FACTOR, 6)
                            if nuovo_tp < tp:
                                with _pos_lock:
                                    sim["tp"] = nuovo_tp
                                    sim["motivo"] = "📈 TP aggiornato (trend 15m)"

                    # ===== REVERSAL detector veloce su 1m (solo per cambiare motivo) =====
                    reversal_msg = None
                    try:
                        df_1m_rev = get_binance_df(symbol, "1m", limit=60)
                        if isinstance(df_1m_rev, pd.DataFrame) and len(df_1m_rev) >= 30:
                            df_1m_rev["EMA_7"]  = df_1m_rev["close"].ewm(span=7).mean()
                            df_1m_rev["EMA_25"] = df_1m_rev["close"].ewm(span=25).mean()

                            c0    = float(df_1m_rev["close"].iloc[-1])
                            c1    = float(df_1m_rev["close"].iloc[-2])
                            e7_0  = float(df_1m_rev["EMA_7"].iloc[-1])
                            e7_1  = float(df_1m_rev["EMA_7"].iloc[-2])
                            e25_0 = float(df_1m_rev["EMA_25"].iloc[-1])
                            e25_1 = float(df_1m_rev["EMA_25"].iloc[-2])

                            # 2 barre consecutive opposte + prezzo oltre EMA7
                            if tipo == "SELL":
                                opp = (e7_0 > e25_0 and e7_1 > e25_1 and c0 > e7_0 and c1 > e7_1)
                                if opp:
                                    reversal_msg = "🔄 Inversione 1m contro SELL"
                            else:  # BUY
                                opp = (e7_0 < e25_0 and e7_1 < e25_1 and c0 < e7_0 and c1 < e7_1)
                                if opp:
                                    reversal_msg = "🔄 Inversione 1m contro BUY"
                    except Exception:
                        pass

                    # ===== messaggio di stato =====
                    if reversal_msg:
                        with _pos_lock:
                            sim["motivo"] = reversal_msg
                    elif not trend_ok:
                        with _pos_lock:
                            sim["motivo"] = "⚠️ Trend 15m incerto"
                    elif verso_sl:
                        with _pos_lock:
                            sim["motivo"] = "⏸️ Ritracciamento, TP stabile"
                    elif "TP aggiornato" not in sim.get("motivo", ""):
                        with _pos_lock:
                            sim["motivo"] = "✅ Trend 15m in linea"

                    logging.info(
                        f"[15m] {symbol} {tipo} Entry={entry:.6f} Price={c:.6f} "
                        f"TP={sim['tp']:.6f} SL={sl:.6f} Prog={progresso:.2f} Motivo={sim['motivo']}"
                    )

                except Exception as e:
                    with _pos_lock:
                        sim["motivo"] = f"❌ Errore monitor 15m: {e}"
                    logging.error(f"[ERRORE] {symbol}: {e}")

            # ===== cleanup periodico =====
            counter += 1
            if counter >= 10:
                try:
                    cleanup_posizioni_attive()
                except Exception as e:
                    logging.error(f"❌ Errore nel cleanup_posizioni_attive: {e}")
                counter = 0

        # ===== CATCH GLOBALE: il monitor NON può morire =====
        except Exception as fatal:
            logging.error(f"💥 ERRORE FATALE nel monitor (protetto): {fatal}")
            time.sleep(1)   # evita loop immediato


# Thread monitor
monitor_thread = threading.Thread(target=verifica_posizioni_attive, daemon=True)
monitor_thread.start()

    
@router.get("/simulazioni_attive")
def simulazioni_attive():
    """
    Restituisce SOLO le simulazioni ancora aperte (non chiuse da backend).
    Quelle chiuse restano in memoria max CLEANUP_AGE_HOURS e
    vengono poi eliminate da cleanup_posizioni_attive().
    """
    with _pos_lock:
        return {
            symbol: data
            for symbol, data in posizioni_attive.items()
            if not data.get("chiusa_da_backend", False)
        }


__all__ = ["router"]
