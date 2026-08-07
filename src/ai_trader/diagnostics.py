"""Caracterizacao estatistica da serie de precos.

Nao temos acesso ao algoritmo real que a IQ Option usa para gerar os precos
OTC (e proprietario, nao documentado publicamente) - entao 'engenharia
reversa' aqui significa medir propriedades estatisticas da serie (e nao
adivinhar o codigo-fonte deles):

  - Hurst exponent: H<0.5 = serie com reversao a media, H=0.5 = passeio
    aleatorio puro, H>0.5 = serie com tendencia/persistencia.
  - Half-life de reversao a media (regressao tipo Ornstein-Uhlenbeck).
  - Autocorrelacao dos retornos em varios lags (persistencia/anti-persistencia
    candle a candle).
  - Granularidade dos precos (tamanho minimo de movimento) - feeds sinteticos
    as vezes tem passo de preco quantizado, diferente de mercado real.

Essas medidas alimentam as estrategias de diagnostics-aware em strategies.py.
"""
import numpy as np
import pandas as pd


def hurst_exponent(prices, max_lag=100):
    prices = np.asarray(prices, dtype=float)
    max_lag = min(max_lag, len(prices) // 2)
    if max_lag < 10:
        return None
    lags = range(2, max_lag)
    tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]
    tau = [t if t > 0 else 1e-12 for t in tau]
    slope, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(slope)


def mean_reversion_half_life(prices):
    prices = np.asarray(prices, dtype=float)
    lagged = prices[:-1]
    delta = prices[1:] - lagged
    beta, _ = np.polyfit(lagged, delta, 1)
    if beta >= 0:
        return None  # nao mostrou reversao a media nessa amostra
    return float(-np.log(2) / beta)


def autocorrelation(returns, max_lag=10):
    s = pd.Series(returns).dropna()
    return {lag: float(s.autocorr(lag=lag)) for lag in range(1, max_lag + 1)}


def tick_granularity(prices):
    diffs = np.abs(np.diff(np.asarray(prices, dtype=float)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return {"min_move": None, "num_distinct_moves": 0}
    return {
        "min_move": float(diffs.min()),
        "median_move": float(np.median(diffs)),
        "num_distinct_moves": int(len(np.unique(np.round(diffs, 8)))),
    }


def summarize(candles_df):
    close = candles_df["close"]
    returns = close.pct_change().dropna()

    h = hurst_exponent(close.values)
    hl = mean_reversion_half_life(close.values)
    acf = autocorrelation(returns, max_lag=5)
    ticks = tick_granularity(close.values)

    if h is None:
        regime = "indefinido (poucos dados)"
    elif h < 0.45:
        regime = "reversao a media (H=%.3f)" % h
    elif h > 0.55:
        regime = "tendencia/persistencia (H=%.3f)" % h
    else:
        regime = "proximo de passeio aleatorio puro (H=%.3f)" % h

    lag1 = acf.get(1)
    if lag1 is not None:
        if lag1 < -0.05:
            lag1_desc = "anti-persistente (candle costuma reverter o anterior)"
        elif lag1 > 0.05:
            lag1_desc = "persistente (candle costuma seguir o anterior)"
        else:
            lag1_desc = "sem correlacao serial detectavel"
    else:
        lag1_desc = "indisponivel"

    return {
        "hurst": h,
        "regime": regime,
        "half_life_candles": hl,
        "autocorrelation": acf,
        "lag1_interpretation": lag1_desc,
        "tick_granularity": ticks,
        "n_candles": len(candles_df),
    }


def format_report(summary):
    lines = [
        "=== Diagnostico estatistico da serie ===",
        f"candles analisados: {summary['n_candles']}",
        f"Hurst exponent: {summary['hurst']} -> {summary['regime']}",
        f"half-life de reversao a media: {summary['half_life_candles']} candles"
        if summary["half_life_candles"] else "half-life: sem reversao a media detectada",
        f"autocorrelacao lag-1: {summary['autocorrelation'].get(1)} -> {summary['lag1_interpretation']}",
        f"granularidade de preco: {summary['tick_granularity']}",
    ]
    return "\n".join(lines)
