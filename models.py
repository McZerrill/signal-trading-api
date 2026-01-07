# models.py

from pydantic import BaseModel
from typing import Optional

class SignalResponse(BaseModel):
    symbol: str
    segnale: str
    commento: str
    prezzo: float
    take_profit: float
    stop_loss: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    ema7: float
    ema25: float
    ema99: float
    timeframe: str
    spread: float
    motivo: str = ""
    chiusa_da_backend: Optional[bool] = False

# ✅ Modello per /hotassets (lista di card Binance/Yahoo)
class HotAsset(BaseModel):
    symbol: str
    segnali: int
    trend: str
    prezzo: float
    rsi: Optional[float] = None
    ema7: Optional[float] = None
    ema25: Optional[float] = None
    ema99: Optional[float] = None
    candele_trend: Optional[int] = None
    note: Optional[str] = None

    


    

    


    
