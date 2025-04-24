# indicators.py

import pandas as pd

def calcola_rsi(serie: pd.Series, periodi: int = 14) -> pd.Series:
    """Calcola l'RSI (Relative Strength Index) su una serie di prezzi."""
    delta = serie.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periodi).mean()
    avg_loss = loss.rolling(window=periodi).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcola_macd(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Calcola MACD e la sua linea segnale."""
    ema12 = serie.ewm(span=12).mean()
    ema26 = serie.ewm(span=26).mean()
    macd = ema12 - ema26
    segnale = macd.ewm(span=9).mean()
    return macd, segnale

def calcola_atr(df: pd.DataFrame, periodi: int = 14) -> pd.Series:
    """Calcola l'Average True Range (ATR) da un DataFrame di OHLC."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodi).mean()
    return atr

def calcola_supporto(df: pd.DataFrame, lookback: int = 20) -> float:
    """Restituisce il minimo dei low nelle ultime N candele."""
    return round(df["low"].tail(lookback).min(), 2)
