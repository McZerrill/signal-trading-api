from fastapi import APIRouter
import logging
import pandas as pd

from models import SignalResponse
from trend_logic import analizza_trend, enrich_indicators
from yahoo_api import (
    get_yahoo_df,
    get_yahoo_last_price,
    YAHOO_SYMBOL_MAP,
)

router = APIRouter()


@router.get("/analyze_yahoo", response_model=SignalResponse)
def analyze_yahoo(symbol: str):
    """
    Analisi stile /analyze ma con dati Yahoo (oro, argento, indici, azioni).
    NON apre simulazioni: solo analisi.
    """
    logging.debug(f"📩 Richiesta /analyze_yahoo per {symbol.upper()}")

    try:
        symbol = symbol.upper()
        y_symbol = YAHOO_SYMBOL_MAP.get(symbol, symbol)

        spread = 0.0  # niente order-book su Yahoo
        prezzo_live = get_yahoo_last_price(y_symbol)

        df_15m = get_yahoo_df(y_symbol, "15m")
        df_1h  = get_yahoo_df(y_symbol, "1h")
        df_1d  = get_yahoo_df(y_symbol, "1d")

        if df_15m is None or df_15m.empty:
            raise ValueError("Nessun dato Yahoo disponibile (15m).")

        # Analisi principale 15m (stessa trend_logic delle crypto)
        segnale, hist, distanza_ema, note15, tp, sl, supporto = analizza_trend(
            df_15m,
            spread,
            None   # per ora niente 1m
        )

        note = note15.split("\n") if note15 else []

        # ---- Indicatori tecnici più recenti ----
        close = rsi = ema7 = ema25 = ema99 = atr = macd = macd_signal = 0.0
        try:
            src = hist if isinstance(hist, pd.DataFrame) and not hist.empty and "close" in hist.columns else df_15m
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
            logging.warning(f"⚠️ Errore dati tecnici Yahoo per {symbol}: {e}")

        prezzo_output = round(prezzo_live, 4) if prezzo_live > 0 else close

        # ---- Conferma 1h semplificata ----
        try:
            if df_1h is not None and not df_1h.empty:
                segnale_1h, hist_1h, _, note1h, *_ = analizza_trend(df_1h, spread, None)
                note.append(f"🕒 1h: {segnale_1h}")
        except Exception as e:
            logging.warning(f"⚠️ Errore analisi 1h Yahoo per {symbol}: {e}")

        # ---- Check daily su EMA (stile Binance) ----
        try:
            if df_1d is not None and not df_1d.empty:
                df_1d_chk = enrich_indicators(df_1d.copy())
                e7d  = float(df_1d_chk["EMA_7"].iloc[-1])
                e25d = float(df_1d_chk["EMA_25"].iloc[-1])
                e99d = float(df_1d_chk["EMA_99"].iloc[-1])

                if e7d > e25d > e99d:
                    daily_state = "BUY"
                elif e7d < e25d < e99d:
                    daily_state = "SELL"
                else:
                    daily_state = "HOLD"

                note.append(f"📅 1d: {daily_state}")
        except Exception as e:
            logging.warning(f"⚠️ Errore analisi daily Yahoo per {symbol}: {e}")

        commento = "\n".join(note) if note else "Nessuna nota"

        # Nessuna simulazione per Yahoo → chiusa_da_backend sempre False
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
        logging.error(f"❌ Errore durante /analyze_yahoo per {symbol}: {e}")
        return SignalResponse(
            symbol=symbol,
            segnale="HOLD",
            commento=f"Errore analisi Yahoo per {symbol}: {e}",
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
            motivo="Errore Yahoo",
            chiusa_da_backend=False
        )


@router.get("/price_yahoo")
def get_price_yahoo(symbol: str):
    """
    Ultimo prezzo da Yahoo (close più recente). Nessuno spread.
    """
    import time
    start = time.time()

    try:
        symbol = symbol.upper()
        y_symbol = YAHOO_SYMBOL_MAP.get(symbol, symbol)

        prezzo = get_yahoo_last_price(y_symbol)
        elapsed = round(time.time() - start, 3)

        return {
            "symbol": symbol,
            "prezzo": round(prezzo, 4),
            "spread": 0.0,
            "tempo": elapsed,
        }

    except Exception as e:
        logging.error(f"❌ Errore /price_yahoo per {symbol}: {e}")
        elapsed = round(time.time() - start, 3)
        return {
            "symbol": symbol,
            "prezzo": 0.0,
            "spread": 0.0,
            "errore": str(e),
            "tempo": elapsed,
        }
