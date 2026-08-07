"""Aprendiz: processo SEPARADO do Gerador, 100% local (nunca abre conexao
com a IQ Option - so le candles do cache local, igual ao Gerador). NAO toca
em nada da busca autonoma normal (autonomous.py/genetic_program.run_gp,
usada pelo Gerador) - aquela continua rodando exatamente igual, sempre
explorando do zero, sem nenhuma mudanca.

O Aprendiz so olha as estrategias de programacao genetica JA CONFIRMADAS no
holdout em qualquer sessao anterior (registry.get_hall_of_fame - "nossos
campeoes alimentam este") e tenta REFINA-LAS (genetic_program.refine_from_seeds:
populacao inteira derivada delas, sem sangue aleatorio novo), convergindo pra
UMA UNICA estrategia (registry.learner_best). Nunca promove sozinho pro
campeao real usado pelo Executor - fica como sugestao visivel no dashboard
ate o usuario mandar usar de verdade (botao "usar esta", que chama
registry.pin_champion, o mesmo mecanismo do ranking).

Uso:
    cd iqoptionapi/src
    python -m ai_trader.learner
"""
import logging
import time
import traceback
import uuid
from collections import deque

from . import candle_store, config, registry, status
from .autonomous import _evaluate_bankroll, _passes_criteria
from .backtester import backtest
from .evolve import train_val_holdout_split
from .features import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS
from .genetic_program import refine_from_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_trader.learner")

MIN_CANDLES_TO_START = 500
WAIT_NO_SEEDS_SECONDS = 60
REFINE_POPULATION = 30
REFINE_GENERATIONS = 15


def _load_seed_trees():
    """Pega a hall da fama (GP ja confirmadas - do Gerador ou do proprio
    Aprendiz), separadas por tipo - classica (indicadores classicos) e raw
    (so price-action) usam terminal_set diferente, nao da pra misturar as
    arvores de uma na outra. Nomes terminados em '_raw' (tanto
    'genetic_program_raw' quanto 'learner_refined_raw') vao pro grupo raw."""
    hof = registry.get_hall_of_fame(config.ACTIVE, config.DURATION_MINUTES, limit=10)
    classic, raw = [], []
    for entry in hof:
        try:
            tree = registry.load_strategy(entry["trial_id"]).tree
        except Exception:
            continue
        (raw if entry["name"].endswith("_raw") else classic).append(tree)
    return classic, raw


def run_learner():
    events = deque(maxlen=40)

    def log_event(text):
        events.appendleft(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}")
        logger.info(text)

    log_event("aprendiz iniciado (100% local, sem rede) - so entra em acao "
               "quando ha pelo menos 1 estrategia confirmada pra refinar")
    status.write_status("learner", phase="aguardando_campeoes", refinements_total=0,
                         events=list(events))

    total_refinements = 0

    while True:
        try:
            classic_trees, raw_trees = _load_seed_trees()
            if not classic_trees and not raw_trees:
                status.write_status(
                    "learner", phase="aguardando_campeoes",
                    refinements_total=total_refinements, events=list(events))
                time.sleep(WAIT_NO_SEEDS_SECONDS)
                continue

            candles_df = candle_store.load_candles(
                config.ACTIVE, config.DURATION_MINUTES, limit=config.GENERATOR_CANDLE_WINDOW)
            if len(candles_df) < MIN_CANDLES_TO_START:
                time.sleep(10)
                continue

            train_df, val_df, holdout_df = train_val_holdout_split(candles_df)
            session_id = uuid.uuid4().hex[:12]

            def heartbeat():
                status.write_status(
                    "learner", phase="refinando", refinements_total=total_refinements,
                    seeds_classic=len(classic_trees), seeds_raw=len(raw_trees),
                    events=list(events))

            log_event(
                f"refinando a partir de {len(classic_trees)} semente(s) classica(s) "
                f"e {len(raw_trees)} raw...")
            heartbeat()

            candidates = []
            if classic_trees:
                strat, val_result, _ = refine_from_seeds(
                    classic_trees, train_df, val_df,
                    population_size=REFINE_POPULATION, generations=REFINE_GENERATIONS,
                    payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
                    terminal_set=FEATURE_COLUMNS, strategy_name="learner_refined",
                    on_generation=lambda g, t: heartbeat())
                candidates.append((strat, val_result))
            if raw_trees:
                strat, val_result, _ = refine_from_seeds(
                    raw_trees, train_df, val_df,
                    population_size=REFINE_POPULATION, generations=REFINE_GENERATIONS,
                    payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
                    terminal_set=RAW_FEATURE_COLUMNS, strategy_name="learner_refined_raw",
                    on_generation=lambda g, t: heartbeat())
                candidates.append((strat, val_result))

            best = None
            for strat, result in candidates:
                if result.num_trades < 30 or result.edge <= 0:
                    continue
                if best is None or result.fitness > best[1].fitness:
                    best = (strat, result)

            total_refinements += 1

            if best is None:
                log_event("refinamento nao produziu nada melhor que ruido desta vez")
            else:
                strategy, _val_result = best
                # confirmacao final: UMA UNICA avaliacao no holdout intocado -
                # mesma disciplina do Gerador (nunca decide com base no
                # conjunto usado pra guiar a busca).
                holdout_result = backtest(
                    strategy, holdout_df, payout=config.DEFAULT_PAYOUT, horizon=1,
                    min_confidence=0.15, amount=config.AMOUNT)
                holdout_bankroll = _evaluate_bankroll(holdout_result)
                confirmed = _passes_criteria(holdout_result, holdout_bankroll)

                trial_id = registry.record_trial(
                    session_id, "learner_refinement", strategy.name, strategy.params,
                    config.ACTIVE, config.DURATION_MINUTES, holdout_result, split="holdout",
                    ruin_probability=holdout_bankroll["ruin_probability"] if holdout_bankroll else None,
                    target_probability=holdout_bankroll["target_probability"] if holdout_bankroll else None,
                    strategy=strategy, candles_used=len(candles_df))

                learner_promoted = registry.set_learner_best_if_better(
                    config.ACTIVE, config.DURATION_MINUTES, trial_id, holdout_result.fitness)
                if learner_promoted:
                    log_event(
                        f"nova escolha do Aprendiz: {strategy.name} edge={holdout_result.edge:+.4f} "
                        f"({'confirmado' if confirmed else 'nao confirmado'} no holdout)")

                # tambem tenta virar CAMPEAO DE VERDADE (o que o Executor usa) -
                # MESMA funcao e MESMO criterio que o Gerador usa pra promover
                # os dele (confirmado sempre vence nao confirmado; empatado
                # nisso, maior fitness vence). O Aprendiz compete em pe de
                # igualdade pelo posto de campeao, mas nunca troca por algo
                # pior do que o que ja esta rodando ao vivo.
                champion_promoted = registry.promote_if_better(
                    config.ACTIVE, config.DURATION_MINUTES, trial_id, strategy.name,
                    confirmed, holdout_result.edge, holdout_result.fitness)
                if champion_promoted:
                    log_event(
                        f"NOVO CAMPEAO (via Aprendiz): {strategy.name} "
                        f"edge={holdout_result.edge:+.4f} "
                        f"{'CONFIRMADO' if confirmed else 'nao confirmado'}")
                if not learner_promoted and not champion_promoted:
                    log_event(
                        f"refinamento avaliado ({strategy.name} edge="
                        f"{holdout_result.edge:+.4f}) mas nao superou nem a escolha "
                        f"atual do Aprendiz nem o campeao real")

            status.write_status(
                "learner", phase="refinando", refinements_total=total_refinements,
                seeds_classic=len(classic_trees), seeds_raw=len(raw_trees),
                events=list(events))
            time.sleep(10)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("erro numa rodada do aprendiz: %s", exc)
            registry.log_error("learner", type(exc).__name__, str(exc), tb)
            log_event(f"ERRO: {exc}")
            status.write_status(
                "learner", phase="erro", refinements_total=total_refinements,
                events=list(events))
            time.sleep(10)


if __name__ == "__main__":
    run_learner()
