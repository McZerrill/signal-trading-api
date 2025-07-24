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

        if symbol in posizioni_attive:
            logging.info(f"⏳ Simulazione già attiva su {symbol} – tipo: {posizioni_attive[symbol]['tipo']} @ {posizioni_attive[symbol]['entry']}$")
            posizione = posizioni_attive[symbol]
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
                rsi=0.0,
                macd=0.0,
                macd_signal=0.0,
                atr=0.0,
                ema7=0.0,
                ema25=0.0,
                ema99=0.0,
                timeframe="15m",
                spread=posizione.get("spread", 0.0),
                motivo=motivo_attuale,
                chiusa_da_backend=posizione.get("chiusa_da_backend", False)
            )

        # Spread
        book = get_bid_ask(symbol)
        spread = book["spread"]
        logging.debug(f"[SPREAD] {symbol} – Spread attuale: {spread:.4f}%")

        if spread > 5.0:
            return SignalResponse(
                symbol=symbol,
                segnale="HOLD",
                commento=f"Simulazione ignorata per {symbol} a causa di spread eccessivo.\nSpread: {spread:.2f}%",
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
                spread=spread,
                motivo="Spread eccessivo",
                chiusa_da_backend=False
            )

        # Dati Binance
        df_15m = get_binance_df(symbol, "15m", 300)
        df_1h = get_binance_df(symbol, "1h", 300)
        df_1d = get_binance_df(symbol, "1d", 300)

        segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(df_15m, spread)

        note = note15.split("\n") if note15 else []
        


        # 📌 Anche se HOLD, recupera sempre gli ultimi dati tecnici
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

        # Logging e analisi timeframe superiori
        logging.debug(f"[BINANCE] {symbol} – 15m: {len(df_15m)} | 1h: {len(df_1h)} | 1d: {len(df_1d)}")
        logging.debug(f"[15m] {symbol} – Segnale: {segnale}, Note: {note15.replace(chr(10), ' | ')}")
        logging.debug(f"[15m DETTAGLI] distEMA={distanza_ema:.6f}, TP={tp:.6f}, SL={sl:.6f}, supporto={supporto:.6f}")

        segnale_1h, *_ = analizza_trend(df_1h, spread)
        segnale_1d, *_ = analizza_trend(df_1d, spread)

        logging.debug(f"[1h] {symbol} – Segnale: {segnale_1h}")
        logging.debug(f"[1d] {symbol} – Segnale: {segnale_1d}")

        # Verifica 1h
        if segnale != segnale_1h:
            logging.info(f"🧭 {symbol} – 1h NON conferma {segnale} (1h = {segnale_1h})")
            try:
                ultimo_1h = df_1h.iloc[-1]
                macd_1h = ultimo_1h['MACD']
                signal_1h = ultimo_1h['MACD_SIGNAL']
                rsi_1h = ultimo_1h['RSI']

                if segnale == "SELL" and macd_1h < 0 and (macd_1h - signal_1h) < 0.005 and rsi_1h < 45:
                    note.append("ℹ️ Timeframe 1h non confermato, ma MACD e RSI coerenti con SELL")
                elif segnale == "BUY" and macd_1h > 0 and (macd_1h - signal_1h) > -0.005 and rsi_1h > 50:
                    note.append("ℹ️ Timeframe 1h non confermato, ma MACD e RSI coerenti con BUY")
                else:
                    note.append(f"⚠️ Segnale {segnale} non confermato su 1h (1h = {segnale_1h})")

                trend_1h = conta_candele_trend(df_1h, rialzista=(segnale == "BUY"))
                if trend_1h < 2:
                    note.append(f"⚠️ Trend su 1h debole ({trend_1h} candele)")
            except Exception as e:
                logging.warning(f"⚠️ Errore dati 1h: {e}")
        else:
            note.append("🧭 1h✓")

        # Verifica 1d
        if segnale in ["BUY", "SELL"]:
            if (segnale == "BUY" and segnale_1d == "SELL") or (segnale == "SELL" and segnale_1d == "BUY"):
                note.append(f"⚠️ Timeframe 1d in conflitto con il segnale ({segnale_1d})")
            else:
                note.append("📅 1d✓")

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
        return SignalResponse(
            symbol=symbol,
            segnale="HOLD",
            commento=f"Errore durante l'analisi: {e}",
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
            spread=0.0,
            motivo="Errore interno",
            chiusa_da_backend=False
        )
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

            distanza_relativa = abs(ema7 - ema99) / ema99
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

def verifica_posizioni_attive():
    while True:
        time.sleep(5)

        for symbol in list(posizioni_attive.keys()):
            simulazione = posizioni_attive.get(symbol)
            if simulazione is None or simulazione.get("esito") in ("Profitto", "Perdita"):
                continue

            tipo = simulazione["tipo"]
            try:
                book = get_bid_ask(symbol)
                prezzo_corrente = book["ask"] if tipo == "BUY" else book["bid"]
                entry = simulazione["entry"]
                tp = simulazione["tp"]
                sl = simulazione["sl"]
                spread = simulazione.get("spread", 0.0)
                tp_esteso = simulazione.get("tp_esteso", 0)
                tp_esteso2 = simulazione.get("tp_esteso2", 0)

                progresso = abs(prezzo_corrente - entry) / abs(tp - entry)

                df_1m = get_binance_df(symbol, "1m", limit=50)
                if df_1m.empty:
                    simulazione["motivo"] = "⚠️ Dati insufficienti (1m)"
                    continue

                df_1m["EMA_7"] = df_1m["close"].ewm(span=7).mean()
                df_1m["EMA_25"] = df_1m["close"].ewm(span=25).mean()
                df_1m["RSI"] = calcola_rsi(df_1m["close"])
                df_1m["MACD"], df_1m["MACD_SIGNAL"] = calcola_macd(df_1m["close"])

                ema7 = df_1m["EMA_7"].iloc[-1]
                ema25 = df_1m["EMA_25"].iloc[-1]
                rsi = df_1m["RSI"].iloc[-1]
                macd, macd_sig = df_1m[["MACD", "MACD_SIGNAL"]].iloc[-1]

                if any(pd.isna(v) for v in (ema7, ema25, rsi, macd, macd_sig)):
                    simulazione["motivo"] = "⚠️ Dati 1m non validi"
                    continue

                motivi = []
                chiusura = False

                if tipo == "BUY":
                    cond1 = ema7 < ema25 and rsi < 50
                    cond2 = ema7 < ema25 and macd < macd_sig - 0.003
                    cond3 = ema7 < ema25 and rsi < 50 and macd < macd_sig - 0.003

                    if cond1 or cond2 or cond3:
                        if ema7 < ema25: motivi.append("EMA7 < EMA25")
                        if rsi < 50: motivi.append(f"RSI {rsi:.1f} < 50")
                        if macd < macd_sig - 0.003: motivi.append(f"MACD {macd:.4f} ≪ Segnale {macd_sig:.4f}")
                        chiusura = True

                else:  # SELL
                    cond1 = ema7 > ema25 and rsi > 57
                    cond2 = ema7 > ema25 and macd > macd_sig + 0.003
                    cond3 = ema7 > ema25 and rsi > 57 and macd > macd_sig + 0.003

                    if cond1 or cond2 or cond3:
                        if ema7 > ema25: motivi.append("EMA7 > EMA25")
                        if rsi > 57: motivi.append(f"RSI {rsi:.1f} > 57")
                        if macd > macd_sig + 0.003: motivi.append(f"MACD {macd:.4f} ≫ Segnale {macd_sig:.4f}")
                        chiusura = True

                if chiusura:
                    simulazione["sl"] = prezzo_corrente
                    simulazione["esito"] = "Perdita"
                    simulazione["motivo"] = "📉 Inversione 1m: " + ", ".join(motivi)
                    simulazione["chiusa_da_backend"] = True
                    logging.info(f"[STOPLOSS FORZATO] {symbol} – {simulazione['motivo']} @ {prezzo_corrente}")
                    continue

                # microtrend coerente
                microtrend_ok = (
                    (tipo == "BUY" and ema7 > ema25 and rsi >= 55 and macd >= macd_sig) or
                    (tipo == "SELL" and ema7 < ema25 and rsi <= 52 and macd <= macd_sig)
                )

                if tp_esteso == 0 and progresso > 0.8 and microtrend_ok:
                    nuovo_tp = round(entry + (tp - entry) * 1.5, 6) if tipo == "BUY" else round(entry - (entry - tp) * 1.5, 6)
                    simulazione["tp"] = nuovo_tp
                    simulazione["tp_esteso"] = 1
                    simulazione["motivo"] = "📈 TP esteso automaticamente (trend forte)"
                    logging.info(f"[TP ESTESO] {symbol} – Nuovo TP: {nuovo_tp}")
                    continue

                if tp_esteso == 1 and progresso > 0.8:
                    microtrend_sfavorevole = (
                        (tipo == "BUY" and (ema7 < ema25 or rsi < 50 or macd < macd_sig)) or
                        (tipo == "SELL" and (ema7 > ema25 or rsi > 57 or macd > macd_sig))
                    )
                    if microtrend_sfavorevole:
                        simulazione["tp"] = prezzo_corrente
                        simulazione["esito"] = "Profitto"
                        simulazione["motivo"] = "🛑 Uscita anticipata con TP esteso (microtrend indebolito)"
                        simulazione["chiusa_da_backend"] = True
                        logging.info(f"[CHIUSURA ANTICIPATA] {symbol} – TP1 esteso ma trend indebolito @ {prezzo_corrente}")
                        continue

                if tp_esteso == 1 and tp_esteso2 == 0 and progresso > 0.9 and microtrend_ok:
                    nuovo_tp = round(entry + (tp - entry) * 1.3, 6) if tipo == "BUY" else round(entry - (entry - tp) * 1.3, 6)
                    simulazione["tp"] = nuovo_tp
                    simulazione["tp_esteso2"] = 2
                    simulazione["motivo"] = "📈 TP esteso ulteriormente (trend ancora forte)"
                    logging.info(f"[TP ESTESO x2] {symbol} – Nuovo TP: {nuovo_tp}")
                    continue

                if tp_esteso2 == 2 and progresso > 0.9:
                    microtrend_sfavorevole = (
                        (tipo == "BUY" and (ema7 < ema25 or rsi < 50 or macd < macd_sig)) or
                        (tipo == "SELL" and (ema7 > ema25 or rsi > 57 or macd > macd_sig))
                    )
                    if microtrend_sfavorevole:
                        simulazione["tp"] = prezzo_corrente
                        simulazione["esito"] = "Profitto"
                        simulazione["motivo"] = "🛑 Uscita anticipata dopo secondo TP esteso (microtrend indebolito)"
                        simulazione["chiusa_da_backend"] = True
                        logging.info(f"[CHIUSURA ANTICIPATA] {symbol} – TP2 esteso ma trend indebolito @ {prezzo_corrente}")
                        continue

                if microtrend_ok:
                    simulazione["motivo"] = "✅ Microtrend 1m in linea col trend principale"
                elif len(motivi) >= 1:
                    simulazione["motivo"] = "👀 Possibile inversione in avvicinamento"
                else:
                    simulazione["motivo"] = "⚠️ Microtrend 1m incerto"

                logging.info(f"[STATO] {symbol} – Entry={entry:.6f}, Prezzo={prezzo_corrente:.6f}, TP={tp:.6f}, SL={sl:.6f}, Progresso={progresso:.2f}, Motivo={simulazione['motivo']}")

            except Exception as err:
                simulazione["motivo"] = f"❌ Errore microtrend 1m: {err}"
                logging.error(f"[ERRORE] Verifica {symbol}: {err}")


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
