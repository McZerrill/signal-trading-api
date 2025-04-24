# routes.py

from fastapi import APIRouter
from pytz import timezone
from datetime import datetime, timezone as dt_timezone
import time

from binance_api import get_binance_df, get_best_symbols
from trend_logic import analizza_trend, conta_candele_trend
from trend_logic import riconosci_pattern_candela
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
                hist['EMA_9'].iloc[i] > hist['EMA_21'].iloc[i] > hist['EMA_100'].iloc[i] or
                hist['EMA_9'].iloc[i] < hist['EMA_21'].iloc[i] < hist['EMA_100'].iloc[i]
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

        # Valori numerici
        close = round(ultimo['close'], 4)
        rsi = round(ultimo['RSI'], 2)
        ema9 = round(ultimo['EMA_9'], 2)
        ema21 = round(ultimo['EMA_21'], 2)
        ema100 = round(ultimo['EMA_100'], 2)
        atr = round(ultimo['ATR'], 2)
        macd = round(ultimo['MACD'], 4)
        macd_signal = round(ultimo['MACD_SIGNAL'], 4)

        if segnale in ["BUY", "SELL"]:
            tp_pct = round(((tp - close) / close) * 100, 1)
            sl_pct = round(((sl - close) / close) * 100, 1)
            commento = (
                f"{'🟢 BUY' if segnale == 'BUY' else '🔴 SELL'} | {symbol.upper()} @ {close}$\n"
                f"🎯 {tp} ({tp_pct}%)   🛡 {sl} ({sl_pct}%)\n"
                f"RSI: {rsi}  |  EMA: {ema9}/{ema21}/{ema100}\n"
                f"MACD: {macd}/{macd_signal}  |  ATR: {atr}\n"
                f"{note}\n{ritardo}"
            )
        else:
            commento = (
                f"⚠️ Nessun segnale confermato tra timeframe 1m e 5m\n"
                f"RSI: {rsi}  |  EMA: {ema9}/{ema21}/{ema100}\n"
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
            ema9=ema9,
            ema21=ema21,
            ema100=ema100,
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
    print("✅ Funzione /hotassets chiamata")
    return [{"test": "Funziona"}]

    
__all__ = ["router"]
