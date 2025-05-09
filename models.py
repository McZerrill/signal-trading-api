# models.py

from pydantic import BaseModel
from typing import Optional

class SignalResponse(BaseModel):
    segnale: str
    commento: str
    prezzo: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    ema7: float
    ema25: float
    ema99: float
    timeframe: str
    spread: float  # ✅ Nuovo campo: spread percentuale tra bid e ask

    
