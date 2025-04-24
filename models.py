# models.py

from pydantic import BaseModel

class SignalResponse(BaseModel):
    segnale: str
    commento: str
    prezzo: float
    take_profit: float
    stop_loss: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    ema9: float
    ema21: float
    ema100: float
    timeframe: str
