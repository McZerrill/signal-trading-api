# routes.py

from fastapi import APIRouter
from pytz import timezone
from datetime import datetime, timezone as dt_timezone
import time

from binance_api import get_binance_df, get_best_symbols
from trend_logic import analizza_trend, conta_candele_trend, riconosci_pattern_candela
from indicators import calcola_rsi, calcola_macd, calcola_atr  # se usi anche questi esplicitamente
from models import SignalResponse


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

        segnale_1m, h1, dist_1m, note1, tp1, sl1, supporto1 = analizza_trend(df_1m)
        segnale_5m, h5, dist_5m, note5, tp5, sl5, supporto5 = analizza_trend(df_5m)

        def conta_trend_attivo(hist):
            return sum(1 for i in range(-10, 0) if (
                hist['EMA_7'].iloc[i] > hist['EMA_25'].iloc[i] > hist['EMA_99'].iloc[i] or
                hist['EMA_7'].iloc[i] < hist['EMA_25'].iloc[i] < hist['EMA_99'].iloc[i]
            ))

        trend_1m = conta_trend_attivo(h1)
        trend_5m = conta_trend_attivo(h5)

        # Scegli timeframe dominante
        if trend_5m > trend_1m:
            segnale, hist, distanza, note, tp, sl, supporto = segnale_5m, h5, dist_5m, note5, tp5, sl5, supporto5
            timeframe = "5m"
        else:
            segnale, hist, distanza, note, tp, sl, supporto = segnale_1m, h1, dist_1m, note1, tp1, sl1, supporto1
            timeframe = "1m"

        ultima_candela = hist.index[-1].to_pydatetime().replace(second=0, microsecond=0, tzinfo=utc)
        orario_utc = ultima_candela.strftime("%H:%M UTC")
        orario_roma = ultima_candela.astimezone(timezone("Europe/Rome")).strftime("%H:%M ora italiana")
        data_candela = ultima_candela.strftime("(%d/%m)")
        ritardo = f"🕒 Dati riferiti alla candela chiusa alle {orario_utc} / {orario_roma} {data_candela}"

        ultimo = hist.iloc[-1]
        close = round(ultimo['close'], 4)
        rsi = round(ultimo['RSI'], 2)
        ema7 = round(ultimo['EMA_7'], 2)
        ema25 = round(ultimo['EMA_25'], 2)
        ema99 = round(ultimo['EMA_99'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)

        if segnale in ["BUY", "SELL"]:
            tp_pct = round(((tp - close) / close) * 100, 1)
            sl_pct = round(((sl - close) / close) * 100, 1)
            commento = (
                f"{'🟢 BUY' if segnale == 'BUY' else '🔴 SELL'} | {symbol.upper()} @ {close}$\n"
                f"🎯 {tp} ({tp_pct}%)   🛡 {sl} ({sl_pct}%)\n"
                f"RSI: {rsi}  |  EMA: {ema7}/{ema25}/{ema99}\n"
                f"MACD: {macd}/{macd_signal}  |  ATR: {atr}\n"
                f"{note}\n{ritardo}"
            )
        else:
            commento = (
                f"⚠️ Nessun segnale confermato tra timeframe 1m e 5m\n"
                f"RSI: {rsi}  |  EMA: {ema7}/{ema25}/{ema99}\n"
                f"MACD: {macd}/{macd_signal}  |  ATR: {atr}\n"
                f"📉 Supporto: {supporto}$\n"
                f"{note}\n{ritardo}"
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
            ema9=ema7,
            ema21=ema25,
            ema100=ema99,
            timeframe=timeframe
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
            ema9=0.0,
            ema21=0.0,
            ema100=0.0,
            timeframe=""
        )

_hot_cache = {"time": 0, "data": []}

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

            # Fase 1: Filtra per volatilità nelle ultime 30 candele
            ultimi_30 = df[-30:]
            min_price = ultimi_30["low"].min()
            max_price = ultimi_30["high"].max()
            variazione_pct = (max_price - min_price) / min_price

            if variazione_pct < 0.03:  # almeno 3% di oscillazione
                continue  # scarta se non abbastanza volatile

            # Calcolo medie
            df["EMA_7"] = df["close"].ewm(span=7).mean()
            df["EMA_25"] = df["close"].ewm(span=25).mean()
            df["EMA_99"] = df["close"].ewm(span=99).mean()
            df["RSI"] = calcola_rsi(df["close"])

            ema7 = df["EMA_7"].iloc[-1]
            ema25 = df["EMA_25"].iloc[-1]
            ema99 = df["EMA_99"].iloc[-1]
            vicino_ema99 = abs(ema7 - ema99) / ema99 < 0.015

            incrocio_buy = any(
                df["EMA_7"].iloc[-i] > df["EMA_25"].iloc[-i] and df["EMA_7"].iloc[-i - 1] < df["EMA_25"].iloc[-i - 1]
                for i in range(1, 6)
            )
            incrocio_sell = any(
                df["EMA_7"].iloc[-i] < df["EMA_25"].iloc[-i] and df["EMA_7"].iloc[-i - 1] > df["EMA_25"].iloc[-i - 1]
                for i in range(1, 6)
            )

            presegnale_buy = incrocio_buy and ema25 < ema99 and vicino_ema99
            presegnale_sell = incrocio_sell and ema25 > ema99 and vicino_ema99

            if presegnale_buy or presegnale_sell:
                segnale = "BUY" if presegnale_buy else "SELL"
                candele_buy = conta_candele_trend(df, rialzista=True)
                candele_sell = conta_candele_trend(df, rialzista=False)
                ultimo = df.iloc[-1]

                risultati.append({
                    "symbol": symbol,
                    "segnali": 1,
                    "trend": segnale,
                    "rsi": round(ultimo["RSI"], 2),
                    "ema9": round(ema7, 2),     # mappati su EMA7
                    "ema21": round(ema25, 2),   # mappati su EMA25
                    "ema100": round(ema99, 2),  # mappati su EMA99
                    "candele_trend": max(candele_buy, candele_sell)
                })

        except Exception as e:
            print(f"❌ Errore con {symbol}: {e}")
            continue

    _hot_cache["time"] = now
    _hot_cache["data"] = risultati
    return risultati
    
__all__ = ["router"]
