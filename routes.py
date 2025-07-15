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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True
)

router = APIRouter()
utc = dt_timezone.utc

# Stato simulazioni attive
posizioni_attive = {}

@router.get("/")
def read_root():
    return {"status": "API Segnali di Borsa attiva"}

@router.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    try:
        if symbol in posizioni_attive:
            posizione = posizioni_attive[symbol]
            return SignalResponse(
                segnale="HOLD",
                commento=(
                    f"⏳ Simulazione già attiva su {symbol.upper()} - tipo: {posizione['tipo']} @ {posizione['entry']}$\n"
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
                motivo=posizione.get("motivo", "")
            )

        book = get_bid_ask(symbol)
        spread = book["spread"]
        if spread > 5.0:
            return SignalResponse(
                segnale="HOLD",
                commento=f"Simulazione ignorata per {symbol.upper()} a causa di spread eccessivo.\nSpread: {spread:.2f}%",
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
                motivo="Spread eccessivo"
            )

        df_15m = get_binance_df(symbol, "15m", 300)
        df_1h = get_binance_df(symbol, "1h", 300)
        df_1d = get_binance_df(symbol, "1d", 300)

        segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(df_15m, spread)
        note = note15.split("\n") if note15 else []

        segnale_1h, *_ = analizza_trend(df_1h, spread)
        segnale_1d, *_ = analizza_trend(df_1d, spread)

        if segnale != segnale_1h:
            ultimo_1h = df_1h.iloc[-1]
            macd_1h = ultimo_1h['MACD']
            signal_1h = ultimo_1h['MACD_SIGNAL']
            rsi_1h = ultimo_1h['RSI']

            if segnale == "SELL" and macd_1h < 0 and (macd_1h - signal_1h) < 0.005 and rsi_1h < 45:
                note.append("ℹ️ Timeframe 1h non confermato, ma MACD e RSI coerenti con SELL")
            elif segnale == "BUY" and macd_1h > 0 and (macd_1h - signal_1h) > -0.005 and rsi_1h > 50:
                note.append("ℹ️ Timeframe 1h non confermato, ma MACD e RSI coerenti con BUY")
            else:
                note.append(f"ℹ️ Segnale {segnale} non confermato su 1h (1h = {segnale_1h})")

            trend_1h = conta_candele_trend(df_1h, rialzista=(segnale == "BUY"))
            if trend_1h < 2:
                note.append(f"ℹ️ Trend su 1h debole ({trend_1h} candele)")
        else:
            note.append("🧭 1h✓")

        if segnale in ["BUY", "SELL"]:
            if (segnale == "BUY" and segnale_1d == "SELL") or (segnale == "SELL" and segnale_1d == "BUY"):
                note.append(f"ℹ️ Timeframe 1d in conflitto con il segnale attuale ({segnale_1d})")
            else:
                note.append("📅 1d✓")

        ultimo = hist.iloc[-1]
        close = round(ultimo['close'], 4)
        if close <= 0:
            raise ValueError(f"Prezzo non valido: {close}")

        rsi = round(ultimo['RSI'], 2)
        ema7 = round(ultimo['EMA_7'], 2)
        ema25 = round(ultimo['EMA_25'], 2)
        ema99 = round(ultimo['EMA_99'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)

        base_dati = f"RSI {rsi} | MACD {macd}/{macd_signal} | EMA {ema7}/{ema25}/{ema99} | ATR {atr}"

        if segnale in ["BUY", "SELL"]:
            entry_price = close
            tp = round(tp, 4)
            sl = round(sl, 4)

            posizioni_attive[symbol] = {
                "tipo": segnale,
                "entry": entry_price,
                "tp": tp,
                "sl": sl,
                "ora_apertura": time.time(),
                "spread": spread,
                "motivo": ""
            }

            tp_pct = round(abs((tp - close) / close) * 100, 2)
            sl_pct = round(abs((sl - close) / close) * 100, 2)

            if any("💥" in riga for riga in note):
                base_dati = "💥 BREAKOUT rilevato\n" + base_dati

            header = "🟢 BUY confermato" if segnale == "BUY" else "🔴 SELL confermato"

            commento = (
                f"{header} | {symbol.upper()} @ {close}$\n"
                f"🎯 TP: {tp} ({tp_pct}%)   🛡 SL: {sl} ({sl_pct}%)\n"
                f"{base_dati}\n" + "\n".join(note)
            )

            motivo_attuale = posizioni_attive[symbol].get("motivo", "")

            return SignalResponse(
                segnale=segnale,
                commento=commento,
                prezzo=entry_price,
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
                motivo=motivo_attuale
            )

        header = f"🚱 HOLD | {symbol.upper()} @ {close}$"
        corpo = f"{base_dati}\n📉 Supporto: {supporto}$\n" + "\n".join(note)

        motivo_attuale = posizioni_attive.get(symbol, {}).get("motivo", "")

        return SignalResponse(
            segnale="HOLD",
            commento=f"{header}\n{corpo}",
            prezzo=close,
            take_profit=0.0,
            stop_loss=0.0,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            atr=atr,
            ema7=ema7,
            ema25=ema25,
            ema99=ema99,
            timeframe="15m",
            spread=spread,
            motivo=motivo_attuale
        )

    except Exception as e:
        return SignalResponse(
            segnale="ERROR",
            commento=f"Errore durante l'analisi di {symbol.upper()}: {e}",
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
            motivo=f"Errore durante l'analisi di {symbol.upper()}: {e}"
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
        return {
            "symbol": symbol,
            "prezzo": prezzo,
            "spread": round(spread, 4),
            "tempo": elapsed,
            "motivo": posizioni_attive.get(symbol, {}).get("motivo", ""),
        }

    except Exception as e:
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

    volume_soglia = 20 if MODALITA_TEST else 300
    atr_minimo = 0.00005 if MODALITA_TEST else 0.0008
    distanza_minima = 0.00005 if MODALITA_TEST else 0.0012
    macd_rsi_range = (40, 60) if MODALITA_TEST else (48, 52)
    macd_signal_threshold = 0.00001 if MODALITA_TEST else 0.0005

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
            simulazione_attiva = posizioni_attive[symbol]
            tipo = simulazione_attiva["tipo"]
            entry = simulazione_attiva["entry"]
            tp = simulazione_attiva["tp"]
            sl = simulazione_attiva["sl"]
            spread = simulazione_attiva["spread"]
            investimento = simulazione_attiva.get("investimento", 100.0)
            commissione = simulazione_attiva.get("commissione", 0.1)

            try:
                # 🔹 Prezzo corrente
                book = get_bid_ask(symbol)
                prezzo_corrente = book["ask"] if tipo == "BUY" else book["bid"]

                # 🔹 Guadagno netto simulato
                prezzo_uscita = (
                    prezzo_corrente * (1 - spread / 100) if tipo == "BUY"
                    else prezzo_corrente * (1 + spread / 100)
                )
                prezzo_ingresso = (
                    entry * (1 + spread / 100) if tipo == "BUY"
                    else entry * (1 - spread / 100)
                )
                rendimento = (
                    prezzo_uscita / prezzo_ingresso if tipo == "BUY"
                    else prezzo_ingresso / prezzo_uscita
                )
                guadagno_netto = round(
                    investimento * rendimento - investimento - investimento * 2 * (commissione / 100), 4
                )
                simulazione_attiva["guadagno_netto"] = guadagno_netto

                # 🔹 TP/SL raggiunto
                chiudere = (
                    (tipo == "BUY" and (prezzo_corrente >= tp or prezzo_corrente <= sl)) or
                    (tipo == "SELL" and (prezzo_corrente <= tp or prezzo_corrente >= sl))
                )

                # 🔹 Microtrend 1m
                df_1m = get_binance_df(symbol, "1m", 40)
                df_1m["EMA_7"] = df_1m["close"].ewm(span=7).mean()
                df_1m["EMA_25"] = df_1m["close"].ewm(span=25).mean()
                df_1m["RSI"] = calcola_rsi(df_1m["close"])
                df_1m["MACD"], df_1m["MACD_SIGNAL"] = calcola_macd(df_1m["close"])

                ema7 = df_1m["EMA_7"].iloc[-1]
                ema25 = df_1m["EMA_25"].iloc[-1]
                rsi_1m = df_1m["RSI"].iloc[-1]
                macd_1m = df_1m["MACD"].iloc[-1]
                macd_signal_1m = df_1m["MACD_SIGNAL"].iloc[-1]

                # 🔹 Verifica condizioni di inversione microtrend
                motivi = []
                if tipo == "BUY":
                    if ema7 < ema25:
                        motivi.append("EMA↓")
                    if rsi_1m < 48:
                        motivi.append("RSI<48")
                    if macd_1m < macd_signal_1m:
                        motivi.append("MACD↓")
                else:  # tipo == "SELL"
                    if ema7 > ema25:
                        motivi.append("EMA↑")
                    if rsi_1m > 52:
                        motivi.append("RSI>52")
                    if macd_1m > macd_signal_1m:
                        motivi.append("MACD↑")

                # 🔹 Motivo aggiornato in tempo reale
                if motivi:
                    simulazione_attiva["motivo"] = f"📉 Microtrend 1m invertito ({', '.join(motivi)})"
                    chiudere = True  # protezione attiva anche se guadagno negativo
                else:
                    simulazione_attiva["motivo"] = (
                        f"📊 1m ema7={ema7:.4f} ema25={ema25:.4f} | "
                        f"rsi={rsi_1m:.1f} | macd={macd_1m:.4f}/{macd_signal_1m:.4f}"
                    )

                # 🔹 Chiusura simulazione
                if chiudere:
                    simulazione_attiva["attiva"] = False
                    simulazione_attiva["chiusa"] = time.time()
                    logging.info(f"[CLOSE] {symbol} - {simulazione_attiva['motivo']}")
                    del posizioni_attive[symbol]

            except Exception as err:
                logging.error(f"Verifica {symbol}: {err}")

            
monitor_thread = threading.Thread(target=verifica_posizioni_attive, daemon=True)
monitor_thread.start()

@router.get("/debuglog")
def get_debug_log():
    return _filtro_log
    
@router.get("/simulazioni_attive")
def simulazioni_attive():
    return posizioni_attive

__all__ = ["router"]
