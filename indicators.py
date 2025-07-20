import pandas as pd
import logging


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

    if not rsi.dropna().empty:
        logging.debug(f"[RSI] Ultimo RSI calcolato: {rsi.dropna().iloc[-1]:.2f}")

    return rsi

def calcola_macd(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Calcola MACD (12-26 EMA) e la sua linea segnale (9-period EMA).
    """
    ema12 = serie.ewm(span=12, adjust=False).mean()
    ema26 = serie.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    segnale = macd.ewm(span=9, adjust=False).mean()

    if not macd.dropna().empty and not segnale.dropna().empty:
        logging.debug(f"[MACD] Ultimi valori → MACD: {macd.dropna().iloc[-1]:.4f}, Segnale: {segnale.dropna().iloc[-1]:.4f}")

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

    if not atr.dropna().empty:
        logging.debug(f"[ATR] Ultimo ATR calcolato: {atr.dropna().iloc[-1]:.6f}")

    return atr

def calcola_supporto(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Restituisce il minimo dei valori 'low' nelle ultime N candele.
    """
    minimo = df["low"].tail(lookback).min()
    logging.debug(f"[SUPPORTO] Minimo {lookback} candele: {minimo:.6f}")
    return minimo

def calcola_ema(df: pd.DataFrame, colonne: list[int]) -> dict[int, pd.Series]:
    """
    Calcola un dizionario di EMA per gli span indicati su 'close'.
    """
    ema_dict = {}
    for periodo in colonne:
        serie = df["close"].ewm(span=periodo, adjust=False).mean()
        ema_dict[periodo] = serie
        if not serie.dropna().empty:
            logging.debug(f"[EMA] EMA {periodo}: ultimo valore = {serie.dropna().iloc[-1]:.6f}")
    return ema_dict

def calcola_percentuale_guadagno(
    guadagno_target: float = 0.5,
    investimento: float = 100.0,
    spread: float = 0.0,
    commissione: float = 0.1
) -> float:
    """
    Calcola la variazione percentuale del prezzo (in valore lordo)
    necessaria per ottenere un guadagno netto desiderato.

    Tiene conto di:
    - commissioni in percentuale (una per ingresso e una per uscita)
    - spread bid/ask in percentuale
    - guadagno netto desiderato in USDC
    - investimento iniziale in USDC

    Restituisce una percentuale decimale (es: 0.0075 = +0.75%)
    """
    # Converti commissione da % a decimale
    commissione_decimale = commissione / 100
    spread_decimale = spread / 100

    # Calcola i costi totali da coprire (spread + doppia commissione)
    costi_totali = (commissione_decimale * 2) + spread_decimale

    # Calcola la percentuale di guadagno lorda necessaria
    percentuale_guadagno_lorda = (guadagno_target / investimento) + costi_totali

    return percentuale_guadagno_lorda
