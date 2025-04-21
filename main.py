from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from binance.client import Client
import time
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API Segnali di Borsa attiva"}

# Classe invariata per compatibilità con l'app
class SignalResponse(BaseModel):
    segnale: str
    commento: str
    prezzo: float
    take_profit: float
    stop_loss: float
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    atr: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    ema100: float = 0.0
    timeframe: str = ""

# Inizializza client Binance (usa chiavi da .env oppure None per uso pubblico)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
