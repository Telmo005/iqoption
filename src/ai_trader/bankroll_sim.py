"""Simulacao de Monte Carlo de evolucao de banca.

Edge medio positivo nao basta: uma estrategia com edge pequeno e variancia
alta pode ter probabilidade real de "quebrar" a banca no caminho, antes da
media de longo prazo se manifestar (risco de ruina). Aqui pegamos a
sequencia REAL de resultados por operacao observada no backtest (nao um
Bernoulli teorico com o win_rate medio - usamos bootstrap da amostra real,
que preserva a distribuicao empirica de streaks/variancia) e simulamos
milhares de sequencias futuras possiveis de operacoes, apostando uma fracao
fixa da banca a cada operacao (aposta proporcional). Disso saem duas
perguntas que o usuario pediu pra responder:

  - qual a chance de "quebrar" a banca (cair abaixo de X% do saldo inicial)?
  - qual a chance de progredir (atingir um multiplo do saldo inicial)?
"""
import numpy as np


def simulate(trade_pnls, backtest_amount=1.0, starting_balance=10.0,
             bet_fraction=0.1, num_trades=200, num_simulations=5000,
             target_multiplier=2.0, ruin_fraction=0.2, seed=None):
    pnls = np.asarray(trade_pnls, dtype=float)
    pnls = pnls[pnls != 0]
    if len(pnls) < 20:
        raise ValueError("poucas operacoes no backtest pra simular banca com confianca")

    unit_returns = pnls / backtest_amount  # retorno por unidade apostada: +payout ou -1

    rng = np.random.default_rng(seed)
    ruin_balance = starting_balance * ruin_fraction
    target_balance = starting_balance * target_multiplier

    final_balances = np.empty(num_simulations)
    ruined = np.zeros(num_simulations, dtype=bool)
    reached_target = np.zeros(num_simulations, dtype=bool)

    for i in range(num_simulations):
        balance = starting_balance
        sampled = rng.choice(unit_returns, size=num_trades, replace=True)
        for r in sampled:
            stake = balance * bet_fraction
            balance += stake * r
            if balance <= ruin_balance:
                ruined[i] = True
                break
            if balance >= target_balance:
                reached_target[i] = True
                break
        final_balances[i] = max(balance, 0.0)

    return {
        "ruin_probability": float(ruined.mean()),
        "target_probability": float(reached_target.mean()),
        "median_final_balance": float(np.median(final_balances)),
        "p10_final_balance": float(np.percentile(final_balances, 10)),
        "p90_final_balance": float(np.percentile(final_balances, 90)),
        "mean_final_balance": float(final_balances.mean()),
        "starting_balance": starting_balance,
        "ruin_balance": ruin_balance,
        "target_balance": target_balance,
        "bet_fraction": bet_fraction,
        "num_trades_simulated": num_trades,
        "num_simulations": num_simulations,
        "num_source_trades": len(unit_returns),
    }


def format_report(sim):
    return (
        f"banca inicial: {sim['starting_balance']:.2f} | aposta: "
        f"{sim['bet_fraction']*100:.0f}% do saldo a cada operacao\n"
        f"probabilidade de ruina (cair a {sim['ruin_balance']:.2f}): "
        f"{sim['ruin_probability']*100:.1f}%\n"
        f"probabilidade de atingir {sim['target_balance']:.2f} "
        f"({sim['target_balance']/sim['starting_balance']:.1f}x) em ate "
        f"{sim['num_trades_simulated']} operacoes: {sim['target_probability']*100:.1f}%\n"
        f"saldo final mediano simulado: {sim['median_final_balance']:.2f} "
        f"(p10={sim['p10_final_balance']:.2f}, p90={sim['p90_final_balance']:.2f})\n"
        f"baseado em {sim['num_source_trades']} operacoes reais do backtest, "
        f"{sim['num_simulations']} simulacoes"
    )
