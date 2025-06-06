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
        df_1m = get_binance_df(symbol, "1m", 300)
        
        df_5m = get_binance_df(symbol, "5m", 300)
        
        df_15m = get_binance_df(symbol, "15m", 200)

        segnale_1m, h1, dist_1m, note1, tp1, sl1, supporto1 = analizza_trend(df_1m)
        segnale_5m, h5, dist_5m, note5, tp5, sl5, supporto5 = analizza_trend(df_5m)
        segnale_15m, h15, *_ = analizza_trend(df_15m)

        segnale, hist, distanza, note, tp, sl, supporto = segnale_1m, h1, dist_1m, note1, tp1, sl1, supporto1

        book = get_bid_ask(symbol)
        spread = book["spread"]
        
        if segnale == "SELL":
            return SignalResponse(
                segnale="HOLD",
                commento=f"🚫 Simulazione SELL disattivata temporaneamente per {symbol.upper()}.\n{note}",
                prezzo=hist['close'].iloc[-1],
                take_profit=0.0,
                stop_loss=0.0,
                rsi=round(hist['RSI'].iloc[-1], 2),
                macd=round(hist['MACD'].iloc[-1], 4),
                macd_signal=round(hist['MACD_SIGNAL'].iloc[-1], 4),
                atr=round(hist['ATR'].iloc[-1], 2),
                ema7=round(hist['EMA_7'].iloc[-1], 2),
                ema25=round(hist['EMA_25'].iloc[-1], 2),
                ema99=round(hist['EMA_99'].iloc[-1], 2),
                timeframe="",
                spread=spread
            )

        timeframe = "1m"

        if segnale in ["BUY", "SELL"]:
            if segnale_5m != segnale:
                note += f"\n⚠️ Segnale {segnale} non confermato su 5m (5m = {segnale_5m})"
                segnale = "HOLD"
            else:
                note += "\n🧭 Segnale confermato anche su 5m"

        if segnale in ["BUY", "SELL"]:
            trend_1m = sum(1 for i in range(-10, 0) if h1['EMA_7'].iloc[i] > h1['EMA_25'].iloc[i] > h1['EMA_99'].iloc[i] or h1['EMA_7'].iloc[i] < h1['EMA_25'].iloc[i] < h1['EMA_99'].iloc[i])
            trend_5m = sum(1 for i in range(-10, 0) if h5['EMA_7'].iloc[i] > h5['EMA_25'].iloc[i] > h5['EMA_99'].iloc[i] or h5['EMA_7'].iloc[i] < h5['EMA_25'].iloc[i] < h5['EMA_99'].iloc[i])
            if trend_5m > trend_1m:
                segnale, hist, distanza, note, tp, sl, supporto = segnale_5m, h5, dist_5m, note5, tp5, sl5, supporto5
                timeframe = "5m"

        if segnale in ["BUY", "SELL"]:
            if (segnale == "BUY" and segnale_15m == "SELL") or (segnale == "SELL" and segnale_15m == "BUY"):
                note += f"\n⚠️ Segnale {segnale} non confermato su 15m (15m = {segnale_15m})"
                segnale = "HOLD"
            elif segnale_15m == segnale:
                note += "\n🧭 Segnale confermato anche su 15m"

        ultima_candela = hist.index[-1].to_pydatetime().replace(second=0, microsecond=0, tzinfo=utc)
        orario_utc = ultima_candela.strftime("%H:%M UTC")
        orario_roma = ultima_candela.astimezone(timezone("Europe/Rome")).strftime("%H:%M ora italiana")
        data_candela = ultima_candela.strftime("(%d/%m)")
        ritardo = f"🕒 Dati riferiti alla candela chiusa alle {orario_utc} / {orario_roma} {data_candela}"

        ultimo = hist.iloc[-1]
        close = round(ultimo['close'], 4)
        if close <= 0:
            raise ValueError(f"Prezzo di chiusura nullo o non valido per {symbol}: close={close}")

        
        if spread > 5.0:
            note += f"\n⚠️ Spread troppo elevato: {spread:.2f}% — segnale ignorato"
            return SignalResponse(
                segnale="HOLD",
                commento=f"Simulazione ignorata per {symbol.upper()} a causa di spread eccessivo.\nSpread: {spread:.2f}%",
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
                spread=spread
            )

        with open("log.txt", "a") as f:
            f.write(f"📊 Spread calcolato per {symbol}: {spread}\n")

        rsi = round(ultimo['RSI'], 2)
        ema7 = round(ultimo['EMA_7'], 2)
        ema25 = round(ultimo['EMA_25'], 2)
        ema99 = round(ultimo['EMA_99'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)

        base_dati = f"RSI: {rsi}  |  EMA: {ema7}/{ema25}/{ema99}\nMACD: {macd}/{macd_signal}  |  ATR: {atr}"

        # ✅ BLOCCO MIGLIORATO
        if segnale in ["BUY", "SELL"]:
            commissione = 0.1
            profitto_minimo = 0.3
            margine_fisso = spread + 2 * commissione + profitto_minimo
            entry_price = close  


            atr = max(atr, 0.0008)
            volatilita_pct = (atr / entry_price) * 100
            rapporto_rr = 1.2 if atr < 0.002 else 1.8 if atr > 0.05 else 1.5
            rischio_pct = max(volatilita_pct * 1.1, 0.8)

            # Target massimo adattivo in base al timeframe
            max_tp_pct = 1.5 if timeframe == "1m" else 2.5 if timeframe == "5m" else 3.5
            tp_pct = min(rischio_pct * rapporto_rr + margine_fisso, max_tp_pct)
            sl_pct = tp_pct / rapporto_rr

            if segnale == "BUY":
                sl = round(entry_price * (1 - sl_pct / 100), 4)
                tp = round(entry_price * (1 + tp_pct / 100), 4)
            else:
                sl = round(entry_price * (1 + sl_pct / 100), 4)
                tp = round(entry_price * (1 - tp_pct / 100), 4)
        else:
            tp = sl = 0.0

        # ✅ Percentuali corrette anche per SELL
        if tp and sl:
            if segnale == "BUY":
                tp_pct = round(((tp - entry_price) / entry_price) * 100, 1)
                sl_pct = round(((sl - entry_price) / entry_price) * 100, 1)
            else:
                tp_pct = round(((entry_price - tp) / entry_price) * 100, 1)
                sl_pct = round(((sl - entry_price) / entry_price) * 100, 1)
        else:
            tp_pct = sl_pct = 0.0
            
        # 🧿 Registrazione della simulazione attiva
        if segnale in ["BUY"]:  # o anche "SELL" se vorrai riattivarli
            posizioni_attive[symbol] = {
                "tipo": segnale,
                "entry": close,
                "tp": tp,
                "sl": sl,
                "ora_apertura": time.time()
            }

        note_str = note.lower() if isinstance(note, str) else "\n".join(note).lower()
        if "💥" in note_str:
            base_dati = "💥 BREAKOUT rilevato\n" + base_dati

        if segnale == "BUY":
            header = "🟢 BUY confermato" if "anticipato" not in note_str else "⚡ BUY anticipato"
        elif segnale == "SELL":
            header = "🔴 SELL confermato" if "anticipato" not in note_str else "⚡ SELL anticipato"
        else:
            header = f"🛁 HOLD | {symbol.upper()} @ {close}$"
            corpo = f"{base_dati}\n📉 Supporto: {supporto}$\n{note}\n{ritardo}"
            
            return SignalResponse(
                segnale=segnale,
                commento="\n".join([header, corpo]),
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
                timeframe=timeframe,
                spread=spread
            )

        commento = (
            f"{header} | {symbol.upper()} @ {close}$\n"
            f"🎯 TP: {tp} ({tp_pct}%)   🛡 SL: {sl} ({sl_pct}%)\n"
            f"{base_dati}\n{note}\n{ritardo}"
        )

        return SignalResponse(
            segnale=segnale,
            commento="\n".join([r.strip() for r in commento.splitlines() if r.strip()]),
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
            timeframe=timeframe,
            spread=spread
        )

    except Exception as e:
        print(f"Errore: {e}")
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

@router.get("/hotassets")
def hot_assets():
    now = time.time()
    if now - _hot_cache["time"] < 30:
        return _hot_cache["data"]

    symbols = get_best_symbols(limit=50)
    risultati = []

    for symbol in symbols:
        try:
            df = get_binance_df(symbol, "1m", 100)
            if df.empty or len(df) < 60:
                continue

            _filtro_log["totali"] += 1

            # FILTRO VOLUME
            volume_medio = df["volume"].tail(20).mean()
            if pd.isna(volume_medio) or volume_medio < 500:
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
                continue  # Ignora asset con prezzo nullo o negativo

            if pd.isna(raw_atr) or raw_atr < 0.001:
                _filtro_log["atr"] += 1
                continue
            atr = round(raw_atr, 4)

            # Filtro su distanza EMA: più tollerante per asset ad alto prezzo
            distanza_relativa = abs(ema7 - ema99) / ema99
            if distanza_relativa < 0.0015 and prezzo < 1000:
                _filtro_log["ema_flat"] += 1
                continue
            # Filtro su variazione prezzo: salta se il prezzo è alto
            oscillazione = df["close"].diff().abs().tail(10).sum()
            if oscillazione < 0.001 and prezzo < 50:
                _filtro_log["prezzo_piattissimo"] += 1
                continue
            # Filtro su MACD e RSI neutri: solo se EMA sono piatte e tutto è "piatto"
            if abs(macd - macd_signal) < 0.0005 and 48 < rsi < 52 and distanza_relativa < 0.0015:
                _filtro_log["macd_rsi_neutri"] += 1
                continue

            distanza_percentuale = abs(ema7 - ema99) / ema99
            recenti_rialzo = all(df["EMA_7"].iloc[-i] > df["EMA_25"].iloc[-i] > df["EMA_99"].iloc[-i] for i in range(1, 4))
            recenti_ribasso = all(df["EMA_7"].iloc[-i] < df["EMA_25"].iloc[-i] < df["EMA_99"].iloc[-i] for i in range(1, 4))

            trend_buy = recenti_rialzo and rsi > 50 and macd > macd_signal
            trend_sell = recenti_ribasso and rsi < 50 and macd < macd_signal

            presegnale_buy = (
                df["EMA_7"].iloc[-2] < df["EMA_25"].iloc[-2] and ema7 > ema25 and ema25 < ema99
                and distanza_percentuale < 0.015 and rsi > 50 and macd > macd_signal
            )
            presegnale_sell = (
                df["EMA_7"].iloc[-2] > df["EMA_25"].iloc[-2] and ema7 < ema25 and ema25 > ema99
                and distanza_percentuale < 0.015 and rsi < 50 and macd < macd_signal
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

            segnale_corrente, hist, *_ = analizza_trend(df)
            candele_attive = conta_candele_trend(hist, rialzista=(posizione["tipo"] == "BUY"))

            # Chiudi se il segnale è cambiato o il trend si è indebolito troppo
            if segnale_corrente != posizione["tipo"] or candele_attive < 2:
                book = get_bid_ask(symbol)
                prezzo_attuale = round((book["bid"] + book["ask"]) / 2, 4)
                pnl = (
                    round(prezzo_attuale - posizione["entry"], 4)
                    if posizione["tipo"] == "BUY"
                    else round(posizione["entry"] - prezzo_attuale, 4)
                )

                print(f"📉 CHIUSURA ANTICIPATA: {symbol} @ {prezzo_attuale} | PnL simulato: {pnl}")
                da_rimuovere.append(symbol)

                with open("log.txt", "a") as f:
                    f.write(
                        f"[{symbol}] Posizione chiusa anticipatamente @ {prezzo_attuale} "
                        f"per {'cambio segnale' if segnale_corrente != posizione['tipo'] else 'trend debole (<2 candele)'}.\n"
                    )

        for s in da_rimuovere:
            posizioni_attive.pop(s, None)

# Avvia il thread all'avvio del backend
monitor_thread = threading.Thread(target=verifica_posizioni_attive, daemon=True)
monitor_thread.start()


@router.get("/debuglog")
def get_debug_log():
    return _filtro_log

__all__ = ["router"]
