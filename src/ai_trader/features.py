"""Engenharia de features a partir de candles.

Prioriza features de price-action puro (retornos, forma do candle,
volatilidade realizada) sobre indicadores classicos, mas inclui os
indicadores classicos tambem como colunas opcionais - a estrategia (ou o
modelo de ML) decide o quanto confiar em cada um.
"""
import numpy as np
import pandas as pd


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    """df: DataFrame com colunas timestamp, open, close, high, low, volume.
    Retorna uma copia com colunas de features adicionadas.
    """
    out = df.copy()
    close = out["close"]

    # ---- price-action puro ----
    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["body"] = (out["close"] - out["open"]) / out["open"]
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]
    out["range"] = (out["high"] - out["low"]) / out["open"]
    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_30"] = out["ret_1"].rolling(30).std()
    out["momentum_5"] = close - close.shift(5)
    out["momentum_10"] = close - close.shift(10)
    # candles OTC costumam vir com volume sempre 0 -> desvio padrao 0 -> 0/0.
    # nesse caso o z-score fica sem informacao (0) em vez de propagar NaN.
    vol_std = out["volume"].rolling(20).std().replace(0, np.nan)
    out["volume_z"] = ((out["volume"] - out["volume"].rolling(20).mean()) / vol_std).fillna(0)

    # ---- indicadores classicos (uso opcional) ----
    out["sma_10"] = close.rolling(10).mean()
    out["sma_30"] = close.rolling(30).mean()
    out["sma_diff"] = (out["sma_10"] - out["sma_30"]) / out["sma_30"]
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["rsi_14"] = _rsi(close, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_position"] = (close - bb_mid) / bb_std.replace(0, np.nan)

    return out


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "body", "upper_wick", "lower_wick", "range",
    "volatility_10", "volatility_30", "momentum_5", "momentum_10", "volume_z",
    "sma_diff", "macd", "macd_signal", "rsi_14", "bb_position",
]

# so price-action cru (retorno, forma do candle, volatilidade, momentum) -
# nenhuma formula de indicador classico (sem RSI/MACD/Bollinger/SMA prontos).
# usado pela variante 'GP cru' que descobre padroes sem herdar formulas dos
# anos 70-80 (ver genetic_program.py).
RAW_FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "body", "upper_wick", "lower_wick", "range",
    "volatility_10", "volatility_30", "momentum_5", "momentum_10", "volume_z",
]


def label_up_down(df, horizon=1):
    """1 se o preco de fechamento 'horizon' candles a frente for maior, senao 0."""
    future_close = df["close"].shift(-horizon)
    return (future_close > df["close"]).astype(int)
