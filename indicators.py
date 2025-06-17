import pandas as pd

def calcola_rsi(serie: pd.Series, periodi: int = 14) -> pd.Series:
    """
    Calcola l'RSI (Relative Strength Index) su una serie di prezzi.
    """
    delta = serie.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=periodi, min_periods=1).mean()
    avg_loss = loss.rolling(window=periodi, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcola_macd(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Calcola MACD (12-26 EMA) e la sua linea segnale (9-period EMA).
    """
    ema12 = serie.ewm(span=12, adjust=False).mean()
    ema26 = serie.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    segnale = macd.ewm(span=9, adjust=False).mean()
    return macd, segnale

def calcola_atr(df: pd.DataFrame, periodi: int = 14) -> pd.Series:
    """
    Calcola l'Average True Range (ATR) da un DataFrame con colonne 'high', 'low' e 'close'.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodi, min_periods=1).mean()
    return atr

def calcola_supporto(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Restituisce il minimo dei valori 'low' nelle ultime N candele.
    """
    return df["low"].tail(lookback).min()

def calcola_ema(df: pd.DataFrame, colonne: list[int]) -> dict[int, pd.Series]:
    """
    Calcola un dizionario di EMA per gli span indicati su 'close'.
    """
    ema_dict = {}
    for periodo in colonne:
        ema_dict[periodo] = df["close"].ewm(span=periodo, adjust=False).mean()
    return ema_dict

def calcola_percentuale_guadagno(
    guadagno_target: float = 0.5,
    investimento: float = 100.0,
    commissione: float = 0.001,
    spread: float = 0.0
) -> float:
    """
    Calcola la percentuale totale necessaria per raggiungere un guadagno netto desiderato,
    includendo le commissioni e lo spread.

    Restituisce una percentuale decimale (es: 0.0075 = +0.75%)
    """
    costi_totali = (commissione * 2) + spread
    percentuale_guadagno = (guadagno_target / investimento) + costi_totali
    return percentuale_guadagno
