from fastapi import APIRouter
from pytz import timezone
from datetime import datetime, timezone as dt_timezone
import time
import requests
import logging
import pandas as pd

from binance_api import get_binance_df, get_best_symbols, get_bid_ask
from trend_logic import analizza_trend, conta_candele_trend, riconosci_pattern_candela
from indicators import calcola_rsi, calcola_macd, calcola_atr
from models import SignalResponse

logging.basicConfig(
    filename="log.txt",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
logging.debug("🧪 LOG DI TEST DEBUG all'avvio")


router = APIRouter()
utc = dt_timezone.utc

# Stato simulazioni attive
posizioni_attive = {}

@router.get("/")
def read_root():
    return {"status": "API Segnali di Borsa attiva"}

@router.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    logging.debug(f"📩 Richiesta /analyze per {symbol.upper()}")

    try:
        symbol = symbol.upper()
        motivo_attuale = posizioni_attive.get(symbol, {}).get("motivo", "")

        # 1) Spread PRIMA (early return se eccessivo)
        book = get_bid_ask(symbol)
        spread = book["spread"]
        logging.debug(f"[SPREAD] {symbol} – Spread attuale: {spread:.4f}%")
        if spread > 5.0:
            try:
                df_15m_tmp = get_binance_df(symbol, "15m", 50)
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

        segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(df_15m, spread, df_1m)
        note = note15.split("\n") if note15 else []

        # 3) Gestione posizione già attiva (UNA SOLA VOLTA QUI)
        if symbol in posizioni_attive:
            logging.info(
                f"⏳ Simulazione già attiva su {symbol} – tipo: "
                f"{posizioni_attive[symbol]['tipo']} @ {posizioni_attive[symbol]['entry']}$"
            )
            posizione = posizioni_attive[symbol]

            # se l’analisi ha “annullato” il segnale → marca la simulazione e restituisci HOLD annotato
            if segnale == "HOLD" and note15 and "Segnale annullato" in note15:
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
        try:
            ultimo = hist.iloc[-1]
            close = round(ultimo['close'], 4)
            rsi = round(ultimo['RSI'], 2)
            ema7 = round(ultimo['EMA_7'], 2)
            ema25 = round(ultimo['EMA_25'], 2)
            ema99 = round(ultimo['EMA_99'], 2)
            atr = round(ultimo['ATR'], 2)
            macd = round(ultimo['MACD'], 4)
            macd_signal = round(ultimo['MACD_SIGNAL'], 4)
        except Exception as e:
            logging.warning(f"⚠️ Errore nell’estrazione dei dati tecnici: {e}")
            close = rsi = ema7 = ema25 = ema99 = atr = macd = macd_signal = 0.0

        # 5) Logging timeframe e analisi di conferma
        logging.debug(f"[BINANCE] {symbol} – 15m: {len(df_15m)} | 1h: {len(df_1h)} | 1d: {len(df_1d)}")
        logging.debug(f"[15m] {symbol} – Segnale: {segnale}, Note: {note15.replace(chr(10), ' | ')}")
        logging.debug(f"[15m DETTAGLI] distEMA={distanza_ema:.6f}, TP={tp:.6f}, SL={sl:.6f}, supporto={supporto:.6f}")

        # 1h: ok usare ancora analizza_trend come conferma "soft"
        segnale_1h, hist_1h, _, note1h, *_ = analizza_trend(df_1h, spread)
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


        # 6) Conferma 1h
        # Prima: conferma "strutturale" solo-EMA su 1h (più aderente a quello che vedi sul grafico)
        try:
            e7h  = float(hist_1h["EMA_7"].iloc[-1])
            e25h = float(hist_1h["EMA_25"].iloc[-1])
            e99h = float(hist_1h["EMA_99"].iloc[-1])
            ema_confirm = (segnale == "BUY"  and e7h > e25h > e99h) or \
                  (segnale == "SELL" and e7h < e25h < e99h)
        except Exception as e:
            logging.warning(f"⚠️ Errore EMA 1h: {e}")
            ema_confirm = False

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
                        note.append(f"⚠️ {segnale} non confermato su 1h (1h = {segnale_1h})")

                    trend_1h = conta_candele_trend(hist_1h, rialzista=(segnale == "BUY"))
                    if trend_1h < 2:
                        note.append(f"⚠️ Trend su 1h debole ({trend_1h} candele)")
                except Exception as e:
                    logging.warning(f"⚠️ Errore dati 1h: {e}")
            else:
                note.append("🧭 1h✓")



        # 7) Note 1d e possibile apertura simulazione
        if segnale in ["BUY", "SELL"]:
            if daily_state == "NA":
                note.append("📅 1d - Check fallito")  # dati non disponibili / check fallito
            else:
                ok_daily = (segnale == "BUY" and daily_state == "BUY") or \
                   (segnale == "SELL" and daily_state == "SELL")
                if ok_daily:
                    note.append("📅 1d✓")
                else:
                    note.append(f"⚠️ 1d in conflitto (daily={daily_state})")

            logging.info(f"✅ Nuova simulazione {segnale} per {symbol} @ {close}$ – TP: {tp}, SL: {sl}, spread: {spread:.2f}%")
            posizioni_attive[symbol] = {
                "tipo": segnale,
                "entry": close,
                "tp": tp,
                "sl": sl,
                "spread": spread,
                "chiusa_da_backend": False,
                "motivo": " | ".join(note)
            }


        commento = "\n".join(note) if note else "Nessuna nota"

        return SignalResponse(
            symbol=symbol,
            segnale=segnale,
            commento=commento,
            prezzo=close,
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
@router.get("/price")
def get_price(symbol: str):
    import time
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
_hot_cache = {"time": 0, "data": [], "valid_until": 0}
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
    if now < _hot_cache["valid_until"] and now - _hot_cache["time"] < 180:
        return _hot_cache["data"]

    symbols = get_best_symbols(limit=50)
    risultati = []

    volume_soglia = 50 if MODALITA_TEST else 300
    atr_minimo = 0.0005 if MODALITA_TEST else 0.0009
    distanza_minima = 0.0006 if MODALITA_TEST else 0.0012
    macd_rsi_range = (43, 57) if MODALITA_TEST else (47.5, 52.5)
    macd_signal_threshold = 0.0003 if MODALITA_TEST else 0.0005

    for symbol in symbols:
        try:
            df = get_binance_df(symbol, "15m", 100)
            if df.empty or len(df) < 60:
                continue

            _filtro_log["totali"] += 1

            volume_medio = df["volume"].tail(20).mean()
            if pd.isna(volume_medio) or volume_medio < volume_soglia:
                _filtro_log["volume_basso"] += 1
                continue

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

            if prezzo <= 0 or pd.isna(atr) or atr < atr_minimo:
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
                risultati.append({
                    "symbol": symbol,
                    "segnali": 1,
                    "trend": trend_pump,
                    "rsi": round(rsi, 2),
                    "ema7": round(ema7, 2),
                    "ema25": round(ema25, 2),
                    "ema99": round(ema99, 2),
                    "prezzo": round(prezzo, 4),
                    "candele_trend": candele_trend
                })
                continue  # salta i filtri successivi: la coin è "hot" per pump


            distanza_relativa = abs(ema7 - ema99) / max(abs(ema99), 1e-9)
            if distanza_relativa < distanza_minima and prezzo < 1000:
                _filtro_log["ema_flat"] += 1
                continue

            oscillazione = df["close"].diff().abs().tail(10).sum()
            if oscillazione < 0.001 and prezzo < 50:
                _filtro_log["prezzo_piattissimo"] += 1
                continue

            if (
                abs(macd - macd_signal) < macd_signal_threshold
                and macd_rsi_range[0] < rsi < macd_rsi_range[1]
                and distanza_relativa < 0.0015
            ):
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
                    "candele_trend": candele_trend
                })
        except Exception:
            continue

    _hot_cache["time"] = now
    _hot_cache["valid_until"] = now + 3600
    _hot_cache["data"] = risultati
    return risultati

# Thread di monitoraggio attivo ogni 5 secondi
import threading

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
    while True:
        time.sleep(CHECK_INTERVAL_SEC)

        for symbol in list(posizioni_attive.keys()):
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

                # ===== verifiche TP/SL su candela corrente (fill al livello, non a mercato) =====
                # tolleranza tick (se disponibile)
                TICK = float(sim.get("tick_size", 0.0)) if isinstance(sim.get("tick_size", 0.0), (int, float)) else 0.0
                eps = TICK

                fill_price = None
                exit_reason = None

                if tipo == "BUY":
                    # conservativo: prima SL, poi TP se SL non toccato
                    if sl and (l <= sl + eps):
                        fill_price = sl
                        exit_reason = "SL"
                    elif tp and (h >= tp - eps):
                        fill_price = tp
                        exit_reason = "TP"
                else:  # SELL
                    # per SELL, priorità SL (high) poi TP (low)
                    if sl and (h >= sl - eps):
                        fill_price = sl
                        exit_reason = "SL"
                    elif tp and (l <= tp + eps):
                        fill_price = tp
                        exit_reason = "TP"

                if fill_price is not None:
                    sim["prezzo_chiusura"]  = round(float(fill_price), 10)
                    sim["chiusa_da_backend"] = True
                    sim["ora_chiusura"]     = datetime.now(tz=utc).isoformat(timespec="seconds")
                    # variazione % rispetto a entry
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
                    # passa al prossimo symbol (questa è chiusa)
                    continue

                # ===== retracement verso SL? =====
                if sl != 0:
                    if tipo == "BUY":
                        verso_sl = c <= entry - SL_BUFFER_PERC * abs(entry - sl)
                    else:  # SELL
                        verso_sl = c >= entry + SL_BUFFER_PERC * abs(entry - sl)
                else:
                    verso_sl = False

                # ===== trailing TP dinamico (solo se trend OK e non verso SL) =====
                if GESTIONE_ATTIVA and trend_ok and not verso_sl:
                    sim.setdefault("tp_esteso", 1)
                    if tipo == "BUY":
                        nuovo_tp = round(entry + distanza_entry * TP_TRAIL_FACTOR, 6)
                        if nuovo_tp > tp:       # sposta solo più in là
                            sim["tp"] = nuovo_tp
                            sim["motivo"] = "📈 TP aggiornato (trend 15m)"
                    else:  # SELL
                        nuovo_tp = round(entry - distanza_entry * TP_TRAIL_FACTOR, 6)
                        if nuovo_tp < tp:
                            sim["tp"] = nuovo_tp
                            sim["motivo"] = "📈 TP aggiornato (trend 15m)"

                # ===== messaggio di stato =====
                if not trend_ok:
                    sim["motivo"] = "⚠️ Trend 15m incerto"
                elif verso_sl:
                    sim["motivo"] = "⏸️ Ritracciamento, TP stabile"
                elif "TP aggiornato" not in sim.get("motivo", ""):
                    sim["motivo"] = "✅ Trend 15m in linea"

                logging.info(
                    f"[15m] {symbol} {tipo} Entry={entry:.6f} Price={c:.6f} "
                    f"TP={sim['tp']:.6f} SL={sl:.6f} Prog={progresso:.2f} Motivo={sim['motivo']}"
                )

            except Exception as e:
                sim["motivo"] = f"❌ Errore monitor 15m: {e}"
                logging.error(f"[ERRORE] {symbol}: {e}")


# Thread monitor
monitor_thread = threading.Thread(target=verifica_posizioni_attive, daemon=True)
monitor_thread.start()

@router.get("/debuglog")
def get_debug_log():
    return _filtro_log
    
@router.get("/simulazioni_attive")
def simulazioni_attive():
    return posizioni_attive

__all__ = ["router"]
