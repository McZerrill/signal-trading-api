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

        def conta_trend_attivo(hist):
            return sum(1 for i in range(-10, 0) if (
                hist['EMA_7'].iloc[i] > hist['EMA_25'].iloc[i] > hist['EMA_99'].iloc[i] or
                hist['EMA_7'].iloc[i] < hist['EMA_25'].iloc[i] < hist['EMA_99'].iloc[i]
            ))

        trend_1m = conta_trend_attivo(h1)
        trend_5m = conta_trend_attivo(h5)

        if trend_5m > trend_1m:
            segnale, hist, distanza, note, _, _, supporto = segnale_5m, h5, dist_5m, note5, tp5, sl5, supporto5
            timeframe = "5m"
        else:
            segnale, hist, distanza, note, _, _, supporto = segnale_1m, h1, dist_1m, note1, tp1, sl1, supporto1
            timeframe = "1m"

        # Conferma 15m
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
        ritardo = f"\U0001F552 Dati riferiti alla candela chiusa alle {orario_utc} / {orario_roma} {data_candela}"

        ultimo = hist.iloc[-1]
        close = round(ultimo['close'], 4)
        book = get_bid_ask(symbol)
        spread = book["spread"]
        with open("log.txt", "a") as f:
            f.write(f"📊 Spread calcolato per {symbol}: {spread}\n")

        rsi = round(ultimo['RSI'], 2)
        ema7 = round(ultimo['EMA_7'], 2)
        ema25 = round(ultimo['EMA_25'], 2)
        ema99 = round(ultimo['EMA_99'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)

        base_dati = (
            f"RSI: {rsi}  |  EMA: {ema7}/{ema25}/{ema99}\n"
            f"MACD: {macd}/{macd_signal}  |  ATR: {atr}"
        )

        # Calcolo TP e SL solo se il segnale è ancora BUY o SELL
        tp = sl = 0.0
        print(f"[DEBUG] Entrato in calcolo TP/SL con segnale={segnale} e conferma15m={segnale_15m}")
        if segnale in ["BUY", "SELL"]:
            print(f"[✅] Segnale confermato ({segnale}) su {timeframe}, calcolo TP/SL attivo")
            commissione = 0.1
            profitto_minimo = 0.5
            margine_totale = spread + (2 * commissione) + profitto_minimo
            tp = round(close * (1 + margine_totale / 100), 4) if segnale == "BUY" else round(close * (1 - margine_totale / 100), 4)

            if segnale_15m == segnale:
                ultime3 = h15.tail(3)
                if segnale == "BUY":
                    min_candele = ultime3['low'].min()
                    sl_ema = min(ultimo['EMA_25'], ultimo['EMA_99'])
                    sl = round(min(min_candele, sl_ema), 4)
                else:
                    max_candele = ultime3['high'].max()
                    sl_ema = max(ultimo['EMA_25'], ultimo['EMA_99'])
                    sl = round(max(max_candele, sl_ema), 4)
            else:
                sl = round(close - atr * 1.2, 4) if segnale == "BUY" else round(close + atr * 1.2, 4)
                note += "\n⏳ SL in attesa: nessuna conferma su 15m"

        tp_pct = round(((tp - close) / close) * 100, 1) if tp else 0.0
        sl_pct = round(((sl - close) / close) * 100, 1) if sl else 0.0

        note_str = note.lower() if isinstance(note, str) else "\n".join(note).lower()
        if "💥" in note_str:
            base_dati = "💥 BREAKOUT rilevato\n" + base_dati

        if segnale == "BUY":
            if "anticipato" in note_str:
                commento = (
                    f"\u26a1 BUY anticipato | {symbol.upper()} @ {close}$\n"
                    f"\U0001F3AF Target stimato: {tp} ({tp_pct}%)   \U0001F6E1 Stop: {sl} ({sl_pct}%)\n"
                    f"{base_dati}\n{note}\n{ritardo}"
                )
            else:
                commento = (
                    f"🟢 BUY confermato | {symbol.upper()} @ {close}$\n"
                    f"🎯 TP: {tp} ({tp_pct}%)   🛡 SL: {sl} ({sl_pct}%)\n"
                    f"{base_dati}\n{note}\n{ritardo}"
                )

        elif segnale == "SELL":
            if "anticipato" in note_str:
                commento = (
                    f"⚡ SELL anticipato | {symbol.upper()} @ {close}$\n"
                    f"\U0001F3AF Target stimato: {tp} ({tp_pct}%)   \U0001F6E1 Stop: {sl} ({sl_pct}%)\n"
                    f"{base_dati}\n{note}\n{ritardo}"
                )
            else:
                commento = (
                    f"🔴 SELL confermato | {symbol.upper()} @ {close}$\n"
                    f"🎯 TP: {tp} ({tp_pct}%)   🛡 SL: {sl} ({sl_pct}%)\n"
                    f"{base_dati}\n{note}\n{ritardo}"
                )

        else:
            if isinstance(note, list):
                note = "\n".join(note)

            note_str = note.lower() if isinstance(note, str) else ""
            header = f"🛁 HOLD | {symbol.upper()} @ {close}$"
            corpo = (
                f"{base_dati}\n"
                f"📉 Supporto: {supporto}$\n"
                f"{'\u26a0\ufe0f Nessuna condizione forte rilevata' if 'trend' not in note_str else note}\n"
                f"{ritardo}"
            )
            commento = "\n".join([header, corpo])
        print(f"[DEBUG] TP calcolato: {tp}, SL calcolato: {sl}")
        print(f"✅ RESTITUZIONE → TP: {tp}, SL: {sl}, Segnale: {segnale}, Timeframe: {timeframe}")
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
            if pd.isna(volume_medio) or volume_medio < 1000:
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

            if pd.isna(raw_atr) or raw_atr < 0.001:
                _filtro_log["atr"] += 1
                continue
            atr = round(raw_atr, 4)

            if abs(ema7 - ema99) / ema99 < 0.002:
                _filtro_log["ema_flat"] += 1
                continue

            if df["close"].diff().abs().tail(10).sum() < 0.001:
                _filtro_log["prezzo_piattissimo"] += 1
                continue

            if abs(macd - macd_signal) < 0.0005 and 48 < rsi < 52:
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

@router.get("/debuglog")
def get_debug_log():
    return _filtro_log

__all__ = ["router"]
