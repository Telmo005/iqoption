"""Backtest vetorizado de uma estrategia contra candles historicos, usando a
economia real de uma opcao digital de duracao fixa: acerta -> +amount*payout,
erra -> -amount. Sem lookahead: o sinal do candle i so usa dados ate o candle
i (as features usam rolling/shift, nunca dados futuros), e o resultado e
decidido pelo fechamento do candle i+horizon.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import build_features


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    num_trades: int
    win_rate: float
    breakeven_win_rate: float
    edge: float
    total_pnl: float
    expectancy: float
    max_drawdown: float
    pnl_curve: pd.Series = field(repr=False)
    trade_pnls: np.ndarray = field(repr=False, default=None)
    amount: float = 1.0

    @property
    def fitness(self):
        """Metrica unica pra ranquear estrategias: expectancy penalizada por
        drawdown e por poucos trades (evita estrategias 'sortudas' com 5 sinais)."""
        if self.num_trades < 30:
            return -1.0
        dd_penalty = 1.0 / (1.0 + self.max_drawdown)
        sample_confidence = min(1.0, self.num_trades / 200)
        return self.expectancy * dd_penalty * sample_confidence


def backtest(strategy, candles_df, payout=0.85, horizon=1, min_confidence=0.15,
             amount=1.0, features_df=None):
    if features_df is None:
        features_df = build_features(candles_df)
    signals = strategy.signals(features_df)

    entry = candles_df["close"]
    expiry = candles_df["close"].shift(-horizon)

    action = signals["action"]
    confidence = signals["confidence"].fillna(0)

    tradable = action.notna() & (confidence >= min_confidence) & expiry.notna()

    is_call = action == "call"
    win = np.where(is_call, expiry > entry, expiry < entry)
    pnl = np.where(win, amount * payout, -amount)
    pnl = pd.Series(pnl, index=candles_df.index)
    pnl = pnl.where(tradable, 0.0)

    trades = pnl[tradable]
    num_trades = int(tradable.sum())
    breakeven_win_rate = 1.0 / (1.0 + payout)

    if num_trades == 0:
        return BacktestResult(
            strategy_name=strategy.name, params=getattr(strategy, "params", {}),
            num_trades=0, win_rate=0.0, breakeven_win_rate=breakeven_win_rate,
            edge=0.0, total_pnl=0.0, expectancy=0.0, max_drawdown=0.0,
            pnl_curve=pnl.cumsum(), trade_pnls=np.array([]), amount=amount)

    win_rate = float((trades > 0).mean())
    total_pnl = float(trades.sum())
    expectancy = float(trades.mean())
    curve = pnl.cumsum()
    running_max = curve.cummax()
    drawdown = (running_max - curve)
    max_drawdown = float(drawdown.max())

    return BacktestResult(
        strategy_name=strategy.name, params=getattr(strategy, "params", {}),
        num_trades=num_trades, win_rate=win_rate,
        breakeven_win_rate=breakeven_win_rate, edge=win_rate - breakeven_win_rate,
        total_pnl=total_pnl, expectancy=expectancy, max_drawdown=max_drawdown,
        pnl_curve=curve, trade_pnls=trades.to_numpy(), amount=amount)
