"""Busca autonoma de estrategia.

Roda ondas continuas de programacao genetica (arvores de expressao livres,
sem formula fixa) intercaladas com a busca evolutiva das familias fixas, sem
precisar de intervencao humana entre ondas. Cada individuo avaliado em
qualquer onda, de qualquer sessao, fica gravado no registro persistente
(registry.py) - isso e o "feedback das estrategias ja geradas" pedido, e
tambem o que evita comecar do zero: a base cresce a cada execucao.

### sobre o risco de "achar" uma estrategia vencedora por acaso

Rodar centenas/milhares de candidatos e comparar todos contra o MESMO pedaco
de dados "de fora da amostra" contamina esse pedaco por selecao multipla: se
voce testa candidato suficiente contra a mesma janela historica, um deles vai
parecer bom so por sorte, mesmo sem nenhum edge real (e exatamente o problema
classico de overfitting em busca de estrategia). Por isso o split aqui e em
TRES partes, nao duas:

  - train:   ajusta arvores de GP / modelo de ML
  - validation: usado REPETIDAMENTE, onda apos onda, pra comparar candidatos
    e escolher quem parece melhor - contaminado por selecao multipla de
    proposito, e serve so pra guiar a busca, nunca pra decidir sozinho
  - holdout: NUNCA tocado durante a busca. So no final, o UNICO candidato
    escolhido (nao uma competicao de varios) e testado contra o holdout,
    uma unica vez. So se o edge sobreviver la e que o candidato e aceito.

Isso e o mais perto de uma validacao estatisticamente honesta que da pra
fazer sem meses de dados ao vivo.

Para ser aceito como vencedor, o candidato precisa passar, NO HOLDOUT:
  1. edge (taxa de acerto - breakeven) >= MIN_EDGE_TO_TRADE
  2. simulacao de Monte Carlo de banca (bankroll_sim.py) mostra probabilidade
     de ruina <= MAX_RUIN_PROBABILITY

A busca para quando o orcamento de tempo (AUTONOMOUS_SECONDS) acaba. No fim,
o melhor candidato de VALIDATION (nao holdout - holdout so pode ser gasto uma
vez) e re-testado contra o holdout intocado, e so essa reavaliacao final
decide se ele e aceito pra operar ao vivo.

Sobre escala: isto NAO testa "milhoes de estrategias" numa unica chamada -
seria preciso horas/dias de computacao continua pra isso de verdade. O que
existe e uma base que cresce a cada sessao rodada; rode de novo (ou com
AUTONOMOUS_SECONDS maior) para acumular mais tentativas.
"""
import logging
import time
import uuid
from types import SimpleNamespace

from . import bankroll_sim, config, registry
from .backtester import backtest
from .diagnostics import format_report, summarize
from .evolve import evolve_strategies, train_val_holdout_split
from .features import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS
from .genetic_program import run_gp

logger = logging.getLogger(__name__)


def _evaluate_bankroll(result):
    if result.trade_pnls is None or len(result.trade_pnls) < 20:
        return None
    try:
        return bankroll_sim.simulate(
            result.trade_pnls, backtest_amount=result.amount,
            starting_balance=config.BANKROLL_STARTING_BALANCE,
            bet_fraction=config.BANKROLL_BET_FRACTION,
            num_trades=config.BANKROLL_HORIZON_TRADES,
            num_simulations=config.BANKROLL_NUM_SIMULATIONS,
            target_multiplier=config.BANKROLL_TARGET_MULTIPLIER,
            ruin_fraction=config.BANKROLL_RUIN_FRACTION)
    except ValueError:
        return None


def _passes_criteria(result, bankroll):
    if result.num_trades < 30 or result.edge < config.MIN_EDGE_TO_TRADE:
        return False
    if bankroll is None or bankroll["ruin_probability"] > config.MAX_RUIN_PROBABILITY:
        return False
    return True


def autonomous_search(candles_df, time_budget_seconds=None, on_progress=None):
    """Busca 100% local: recebe os candles ja baixados (ver broker.get_candles_df,
    chamado uma vez por quem orquestra) e nao toca a rede em nenhum momento
    daqui pra frente. Isso e proposital: segurar uma conexao de websocket
    aberta e ociosa durante minutos de computacao pura foi exatamente o que
    derrubou a conexao em teste real (o servidor fecha socket parado). Quem
    chama esta funcao decide quando e com que frequencia buscar candles
    novos - a busca em si nunca precisa de rede.

    'on_progress(session_id, trials_tested_nesta_sessao)' e chamado apos
    cada onda, se passado - serve pra quem chama mostrar um contador ao vivo
    de quantas estrategias ja foram testadas (nao depende do relogio, so do
    numero de tentativas de fato realizadas)."""
    time_budget_seconds = time_budget_seconds or config.AUTONOMOUS_SECONDS
    session_id = uuid.uuid4().hex[:12]
    logger.info("=== busca autonoma iniciada | sessao=%s orcamento=%ds ===",
                session_id, time_budget_seconds)

    diag = summarize(candles_df)
    print(format_report(diag))

    train_df, val_df, holdout_df = train_val_holdout_split(candles_df)
    logger.info("split: %d treino / %d validacao / %d holdout (intocado ate o final)",
                len(train_df), len(val_df), len(holdout_df))

    def persist(kind, split):
        def _cb(strat, result):
            registry.record_trial(
                session_id, kind, strat.name, strat.params,
                config.ACTIVE, config.DURATION_MINUTES, result, split=split,
                candles_used=len(candles_df))
        return _cb

    def _heartbeat():
        if on_progress is None:
            return
        try:
            on_progress(session_id, registry.count_trials(session_id=session_id))
        except Exception:
            logger.exception("on_progress falhou (ignorando)")

    start = time.time()
    wave = 0
    best_candidate = None  # (score, strategy, result, bankroll) - avaliado em VALIDATION
    _heartbeat()  # cobre o intervalo antes da primeira onda terminar

    while time.time() - start < time_budget_seconds:
        wave += 1
        elapsed = time.time() - start
        logger.info("onda %d (decorrido %.0fs / %ds)...", wave, elapsed, time_budget_seconds)

        # individuos DURANTE a evolucao da GP sao julgados no treino; so o
        # vencedor de cada onda e reavaliado em validation (linha abaixo) -
        # validation ainda e reutilizado entre ondas, entao serve so pra
        # guiar a busca, nao pra decidir sozinho (ver holdout no final).
        # on_generation chama o heartbeat a cada geracao (nao so no final da
        # onda inteira) - sem isso, uma onda de 60s deixava o painel ate 35s
        # achando que o processo tinha parado, mesmo ele estando vivo.
        gp_strategy, gp_result, _ = run_gp(
            train_df, val_df,
            population_size=config.GP_POPULATION,
            generations=config.GP_WAVE_GENERATIONS,
            max_depth=config.GP_MAX_DEPTH,
            payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
            on_individual=persist("gp", "train"), on_generation=lambda g, t: _heartbeat(),
            terminal_set=FEATURE_COLUMNS, strategy_name="genetic_program")
        persist("gp", "validation")(gp_strategy, gp_result)

        # mesma busca, mas a arvore so pode usar price-action cru (sem
        # RSI/MACD/Bollinger/SMA prontos) - descobre padroes sem herdar
        # formulas de indicador; compete lado a lado com tudo o resto pelo
        # mesmo criterio de fitness, sem substituir nada que ja existe.
        gp_raw_strategy, gp_raw_result, _ = run_gp(
            train_df, val_df,
            population_size=config.GP_POPULATION,
            generations=config.GP_WAVE_GENERATIONS,
            max_depth=config.GP_MAX_DEPTH,
            payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
            on_individual=persist("gp_raw", "train"), on_generation=lambda g, t: _heartbeat(),
            terminal_set=RAW_FEATURE_COLUMNS, strategy_name="genetic_program_raw")
        persist("gp_raw", "validation")(gp_raw_strategy, gp_raw_result)

        candidates = [(gp_strategy, gp_result), (gp_raw_strategy, gp_raw_result)]

        if wave % 3 == 1:
            fam_strategy, fam_result, _ = evolve_strategies(
                _concat(train_df, val_df),
                payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
                generations=2, on_individual=persist("family", "validation"))
            candidates.append((fam_strategy, fam_result))

        for strat, result in candidates:
            if result.num_trades < 30 or result.edge <= 0:
                continue
            bankroll = _evaluate_bankroll(result)
            if bankroll is None:
                continue
            score = result.edge * (1 - bankroll["ruin_probability"])
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, strat, result, bankroll)
                logger.info(
                    "novo melhor candidato desta sessao (validation, ainda "
                    "nao confirmado): %s edge=%.3f prob_ruina=%.1f%% prob_meta=%.1f%%",
                    strat.name, result.edge, bankroll["ruin_probability"] * 100,
                    bankroll["target_probability"] * 100)

        _heartbeat()

    session_trials = registry.count_trials(session_id=session_id)
    total_trials = registry.count_trials()
    logger.info(
        "=== busca autonoma encerrada: %d tentativas nesta sessao, %d no total "
        "(todas as sessoes ja rodadas) ===", session_trials, total_trials)

    leaderboard = registry.best_overall(
        active=config.ACTIVE, min_trades=30, limit=10, split="validation")
    print("\n" + registry.format_leaderboard(leaderboard))

    if best_candidate is None:
        logger.error(
            "nenhum candidato desta sessao teve operacoes suficientes pra "
            "simular banca. veja o leaderboard historico acima - pode haver "
            "algo melhor de sessoes anteriores, mas nada novo foi encontrado agora.")
        return None, session_id

    _, strategy, val_result, val_bankroll = best_candidate

    # ---- confirmacao final: UMA UNICA avaliacao no holdout intocado ----
    # mas so se essa MESMA estrategia (nome+parametros identicos) ainda nao
    # foi confirmada ha pouco tempo - backtest e deterministico, entao reavaliar
    # o mesmo candidato onda apos onda (comum quando nada supera o campeao
    # ainda) so duplicava a linha no banco sem informacao nova.
    reuse_window_seconds = config.REUSE_CONFIRMED_MAX_AGE_HOURS * 3600
    existing = registry.find_recent_holdout_by_name(
        config.ACTIVE, config.DURATION_MINUTES, strategy.name, strategy.params,
        max_age_seconds=reuse_window_seconds) if reuse_window_seconds > 0 else None

    if existing is not None:
        logger.info(
            "%s com estes parametros ja foi confirmado no holdout ha pouco "
            "(trial #%d) - reaproveitando em vez de re-testar e duplicar no banco",
            strategy.name, existing["id"])
        holdout_result = SimpleNamespace(
            edge=existing["edge"], fitness=existing["fitness"],
            num_trades=existing["num_trades"])
        holdout_bankroll = (
            {"ruin_probability": existing["ruin_probability"],
             "target_probability": existing["target_probability"]}
            if existing["ruin_probability"] is not None else None)
        found_winner = (
            existing["edge"] >= config.MIN_EDGE_TO_TRADE
            and (existing["ruin_probability"] is None
                 or existing["ruin_probability"] <= config.MAX_RUIN_PROBABILITY))
        return (strategy, holdout_result, holdout_bankroll, found_winner), session_id

    logger.info(
        "confirmando o melhor candidato de validation (%s) contra o holdout "
        "nunca tocado (%d candles)...", strategy.name, len(holdout_df))
    holdout_result = backtest(
        strategy, holdout_df, payout=config.DEFAULT_PAYOUT, horizon=1,
        min_confidence=0.15, amount=config.AMOUNT)
    holdout_bankroll = _evaluate_bankroll(holdout_result)
    registry.record_trial(
        session_id, "holdout_confirmation", strategy.name, strategy.params,
        config.ACTIVE, config.DURATION_MINUTES, holdout_result, split="holdout",
        ruin_probability=holdout_bankroll["ruin_probability"] if holdout_bankroll else None,
        target_probability=holdout_bankroll["target_probability"] if holdout_bankroll else None,
        strategy=strategy, candles_used=len(candles_df))

    found_winner = _passes_criteria(holdout_result, holdout_bankroll)

    print(f"\n=== Escolha da IA (sessao {session_id}) ===")
    print(f"estrategia: {strategy.name} | params: {strategy.params}")
    print(f"--- desempenho em VALIDATION (usado pra escolher, reaproveitado entre ondas) ---")
    print(f"edge: {val_result.edge:.4f} | trades: {val_result.num_trades}")
    print(f"--- desempenho em HOLDOUT (nunca tocado antes, avaliado uma unica vez) ---")
    print(f"edge: {holdout_result.edge:.4f} | trades: {holdout_result.num_trades}")
    if holdout_bankroll:
        print(bankroll_sim.format_report(holdout_bankroll))
    else:
        print("holdout com poucas operacoes pra simular banca com confianca.")
    print(f"passa nos criterios de seguranca para operar ao vivo (baseado no HOLDOUT): "
          f"{'SIM' if found_winner else 'NAO'}")

    if val_result.edge > 0 and holdout_result.edge <= 0:
        print(
            "\nATENCAO: o edge positivo apareceu em validation mas NAO se "
            "confirmou no holdout intocado - isso e o sinal classico de que "
            "aquele resultado era ruido/sorte de ter testado muitos "
            "candidatos contra o mesmo pedaco de dados, nao um padrao real.")

    return (strategy, holdout_result, holdout_bankroll, found_winner), session_id


def _concat(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)
