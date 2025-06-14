# routes.py

from fastapi import APIRouter
from pytz import timezone
from datetime import datetime, timezone as dt_timezone
import time

from binance_api import get_binance_df, get_best_symbols
from trend_logic import analizza_trend, conta_candele_trend, riconosci_pattern_candela
from indicators import calcola_rsi, calcola_macd, calcola_atr  # se usi anche questi esplicitamente
from models import SignalResponse
import pandas as pd
from binance_api import get_bid_ask
import requests

# Stato simulazioni attive
posizioni_attive = {}  # Esempio: { "ADAUSDC": {"tipo": "BUY", "prezzo": 0.45, "ora_apertura": time.time()} }

router = APIRouter()
utc = dt_timezone.utc

@router.get("/")
def read_root():
    return {"status": "API Segnali di Borsa attiva"}
    
@router.get("/analyze", response_model=SignalResponse)
def analyze(symbol: str):
    try:
        # 🔒 Blocco: posizione già attiva
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
                spread=0.0
            )

        # 🧮 Spread check
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
                spread=spread
            )

        # 📊 Dati da Binance
        df_15m = get_binance_df(symbol, "15m", 300)
        df_1h = get_binance_df(symbol, "1h", 300)
        df_1d = get_binance_df(symbol, "1d", 300)

        segnale_15m, h15, dist_15m, note15, tp15, sl15, supporto15 = analizza_trend(df_15m, spread)
        segnale_1h, h1h, *_ = analizza_trend(df_1h, spread)
        segnale_1d, *_ = analizza_trend(df_1d, spread)

        segnale, hist, distanza, note, tp, sl, supporto = segnale_15m, h15, dist_15m, note15, tp15, sl15, supporto15

        # ⏸ Conferme multi-timeframe
        if segnale != segnale_1h:
            note += f"\n⚠️ Segnale {segnale} non confermato su 1h (1h = {segnale_1h})"
            segnale = "HOLD"
        else:
            note += "\n🧭 Segnale confermato anche su 1h"

        if segnale in ["BUY", "SELL"]:
            if (segnale == "BUY" and segnale_1d == "SELL") or (segnale == "SELL" and segnale_1d == "BUY"):
                note += f"\n⚠️ Segnale {segnale} non confermato su 1d (1d = {segnale_1d})"
                segnale = "HOLD"
            else:
                note += "\n📅 Timeframe 1d non contrario: segnale permesso"

        # 📈 Indicatori attuali
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

        base_dati = f"RSI: {rsi}  |  EMA: {ema7}/{ema25}/{ema99}\nMACD: {macd}/{macd_signal}  |  ATR: {atr}"

        # ✅ Simulazione
        if segnale in ["BUY", "SELL"]:
            entry_price = close
            tp = round(tp, 4)
            sl = round(sl, 4)

            posizioni_attive[symbol] = {
                "tipo": segnale,
                "entry": entry_price,
                "tp": tp,
                "sl": sl,
                "ora_apertura": time.time()
            }

            tp_pct = round(abs((tp - close) / close) * 100, 2)
            sl_pct = round(abs((sl - close) / close) * 100, 2)

            if "💥" in note.lower():
                base_dati = "💥 BREAKOUT rilevato\n" + base_dati

            header = "🟢 BUY confermato" if segnale == "BUY" else "🔴 SELL confermato"

            commento = (
                f"{header} | {symbol.upper()} @ {close}$\n"
                f"🎯 TP: {tp} ({tp_pct}%)   🛡 SL: {sl} ({sl_pct}%)\n"
                f"{base_dati}\n{note}"
            )

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
                spread=spread
            )

        # 🛁 HOLD
        header = f"🛁 HOLD | {symbol.upper()} @ {close}$"
        corpo = f"{base_dati}\n📉 Supporto: {supporto15}$\n{note}"

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
            spread=spread
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
            spread=0.0
        )
        
@router.get("/price")
def get_price(symbol: str):
    import time
    start = time.time()
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
        response = requests.get(url, timeout=3)
        data = response.json()

        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        # Protezione contro valori non validi
        if bid <= 0 or ask <= 0:
            raise ValueError(f"Prezzo non valido: bid={bid}, ask={ask}")

        spread = (ask - bid) / ((ask + bid) / 2) * 100
        prezzo = round((bid + ask) / 2, 4)

        elapsed = round(time.time() - start, 3)
        print(f"/price {symbol} ➜ prezzo: {prezzo}, spread: {spread:.4f}% (risposto in {elapsed}s)")

        return {
            "symbol": symbol,
            "prezzo": prezzo,
            "spread": round(spread, 4),
            "tempo": elapsed
        }

    except Exception as e:
        elapsed = round(time.time() - start, 3)
        print(f"/price {symbol} ERRORE: {e} (in {elapsed}s)")
        return {
            "symbol": symbol,
            "prezzo": 0.0,
            "spread": 0.0,
            "errore": str(e),
            "tempo": elapsed
        }

_hot_cache = {"time": 0, "data": []}

_filtro_log = {
    "totali": 0,
    "atr": 0,
    "ema_flat": 0,
    "volume_basso": 0,
    "prezzo_piattissimo": 0,
    "macd_rsi_neutri": 0
}

_hot_cache = {"time": 0, "data": [], "valid_until": 0}

MODALITA_TEST = True

@router.get("/hotassets")
def hot_assets():
    now = time.time()

    # Mantieni asset hot validi per 60 minuti, aggiorna ogni 3 minuti
    if now < _hot_cache["valid_until"] and now - _hot_cache["time"] < 180:
        return _hot_cache["data"]

    symbols = get_best_symbols(limit=50)
    risultati = []

    # Parametri adattivi in base alla modalità
    volume_soglia = 100 if MODALITA_TEST else 300
    atr_minimo = 0.0005 if MODALITA_TEST else 0.0008
    distanza_minima = 0.0008 if MODALITA_TEST else 0.0012
    macd_rsi_range = (46, 54) if MODALITA_TEST else (48, 52)
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
            raw_atr = df["ATR"].iloc[-1]
            prezzo = df["close"].iloc[-1]
            if prezzo <= 0:
                continue

            if pd.isna(raw_atr) or raw_atr < atr_minimo:
                _filtro_log["atr"] += 1
                continue

            atr = round(raw_atr, 4)

            distanza_relativa = abs(ema7 - ema99) / ema99
            if distanza_relativa < distanza_minima and prezzo < 1000:
                _filtro_log["ema_flat"] += 1
                continue

            oscillazione = df["close"].diff().abs().tail(10).sum()
            if oscillazione < 0.001 and prezzo < 50:
                _filtro_log["prezzo_piattissimo"] += 1
                continue

            if abs(macd - macd_signal) < macd_signal_threshold and macd_rsi_range[0] < rsi < macd_rsi_range[1] and distanza_relativa < 0.0015:
                _filtro_log["macd_rsi_neutri"] += 1
                continue

            recenti_rialzo = all(df["EMA_7"].iloc[-i] > df["EMA_25"].iloc[-i] > df["EMA_99"].iloc[-i] for i in range(1, 4))
            recenti_ribasso = all(df["EMA_7"].iloc[-i] < df["EMA_25"].iloc[-i] < df["EMA_99"].iloc[-i] for i in range(1, 4))

            trend_buy = recenti_rialzo and rsi > 50 and macd > macd_signal
            trend_sell = recenti_ribasso and rsi < 50 and macd < macd_signal

            presegnale_buy = (
                df["EMA_7"].iloc[-2] < df["EMA_25"].iloc[-2] and ema7 > ema25 and ema25 < ema99
                and distanza_relativa < 0.015 and rsi > 50 and macd > macd_signal
            )
            presegnale_sell = (
                df["EMA_7"].iloc[-2] > df["EMA_25"].iloc[-2] and ema7 < ema25 and ema25 > ema99
                and distanza_relativa < 0.015 and rsi < 50 and macd < macd_signal
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

        except Exception as e:
            print(f"❌ Errore con {symbol}: {e}")
            continue

    _hot_cache["time"] = now
    _hot_cache["valid_until"] = now + 3600  # validi per 60 minuti
    _hot_cache["data"] = risultati
    return risultati


import threading

def verifica_posizioni_attive():
    while True:
        time.sleep(5)
        da_rimuovere = []

        for symbol, posizione in list(posizioni_attive.items()):
            df = get_binance_df(symbol, "1m", 300)
            if df.empty or len(df) < 50:
                continue

            # 1. Analisi trend attuale
            segnale_corrente, hist, *_ = analizza_trend(df)
            candele_attive = conta_candele_trend(hist, rialzista=(posizione["tipo"] == "BUY"))

            # 2. Prezzo attuale
            book = get_bid_ask(symbol)
            prezzo_attuale = round((book["bid"] + book["ask"]) / 2, 4)

            # 3. TP / SL raggiunti?
            entry = posizione["entry"]
            tp = posizione["tp"]
            sl = posizione["sl"]
            tipo = posizione["tipo"]

            if tipo == "BUY" and prezzo_attuale >= tp:
                motivo = "🎯 TP raggiunto"
            elif tipo == "BUY" and prezzo_attuale <= sl:
                motivo = "🛡 SL colpito"
            elif tipo == "SELL" and prezzo_attuale <= tp:
                motivo = "🎯 TP raggiunto"
            elif tipo == "SELL" and prezzo_attuale >= sl:
                motivo = "🛡 SL colpito"
            elif segnale_corrente != tipo and candele_attive < 2:
                motivo = "📉 Chiusura anticipata: cambio segnale e trend debole"
            else:
                continue  # Posizione ancora valida

            # 4. Calcolo PnL simulato
            pnl = round(prezzo_attuale - entry, 4) if tipo == "BUY" else round(entry - prezzo_attuale, 4)

            print(f"🔔 CHIUSURA: {symbol} @ {prezzo_attuale} | {motivo} | PnL: {pnl}")
            da_rimuovere.append(symbol)

            # 5. Logging su file
            with open("log.txt", "a") as f:
                f.write(f"[{symbol}] Posizione chiusa @ {prezzo_attuale} | {motivo} | PnL: {pnl}\n")

        # Rimuovi posizioni chiuse
        for s in da_rimuovere:
            posizioni_attive.pop(s, None)

# Avvia il thread all'avvio del backend
monitor_thread = threading.Thread(target=verifica_posizioni_attive, daemon=True)
monitor_thread.start()


@router.get("/debuglog")
def get_debug_log():
    return _filtro_log

__all__ = ["router"]
