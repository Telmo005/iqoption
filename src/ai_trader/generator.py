"""Gerador de estrategias: roda continuamente, 100% local. NUNCA abre
conexao com a IQ Option - so le candles do cache local (candle_store.py,
mantido pelo sync.py num processo separado). Pode ficar ligado por horas
sem nenhum risco de conflito de conexao, porque nunca toca rede.

A cada onda concluida, promove o candidato a "campeao" (registry.champions)
se ele for melhor que o atual, seguindo a prioridade: confirmado no holdout
sempre vence nao-confirmado; dentro do mesmo status, maior fitness vence.
O executor (executor.py, processo separado) so consulta esse campeao,
nunca gera estrategia nenhuma.

Uso:
    cd iqoptionapi/src
    python -m ai_trader.generator
"""
import logging
import time
import traceback
import uuid
from collections import deque

from . import candle_store, config, registry, status
from .autonomous import autonomous_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_trader.generator")

MIN_CANDLES_TO_START = 500


def _wait_for_data():
    while True:
        n = candle_store.count_candles(config.ACTIVE, config.DURATION_MINUTES)
        if n >= MIN_CANDLES_TO_START:
            return n
        logger.info(
            "cache local ainda tem so %d candles (precisa de %d) - "
            "esperando o sincronizador rodar... tentando de novo em 30s",
            n, MIN_CANDLES_TO_START)
        status.write_status(
            "generator", phase="aguardando_dados", trials_tested_total=0,
            champion_name=None, champion_confirmed=False, champion_edge=None,
            events=[f"{time.strftime('%Y-%m-%d %H:%M:%S')}  esperando o sincronizador ter "
                    f"{MIN_CANDLES_TO_START} candles (tem {n})"])
        time.sleep(30)


def run_generator():
    events = deque(maxlen=40)

    def log_event(text):
        events.appendleft(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}")
        logger.info(text)

    log_event("gerador iniciado (100% local, sem rede)")
    status.write_status(
        "generator", phase="gerando", trials_tested_total=0,
        champion_name=None, champion_confirmed=False, champion_edge=None,
        events=list(events))

    n = _wait_for_data()
    log_event(f"{n} candles disponiveis no cache local, comecando a gerar estrategias...")
    status.write_status(
        "generator", phase="gerando", trials_tested_total=0,
        champion_name=_champion_summary()[0], champion_confirmed=_champion_summary()[1],
        champion_edge=_champion_summary()[2], candles_available=n, events=list(events))

    total_trials_this_run = 0

    while True:
        try:
            candles_df = candle_store.load_candles(
                config.ACTIVE, config.DURATION_MINUTES, limit=config.GENERATOR_CANDLE_WINDOW)
            if len(candles_df) < MIN_CANDLES_TO_START:
                time.sleep(10)
                continue

            def on_progress(session_id, trials_tested):
                name, is_confirmed, edge = _champion_summary()
                status.write_status(
                    "generator", phase="gerando", trials_tested_session=trials_tested,
                    trials_tested_total=total_trials_this_run + trials_tested,
                    champion_name=name, champion_confirmed=is_confirmed, champion_edge=edge,
                    candles_available=len(candles_df),
                    events=list(events))

            outcome, session_id = autonomous_search(
                candles_df, time_budget_seconds=config.GENERATOR_WAVE_SECONDS,
                on_progress=on_progress)

            session_trials = registry.count_trials(session_id=session_id)
            total_trials_this_run += session_trials

            if outcome is not None:
                strategy, result, bankroll, confirmed = outcome
                rows = registry.best_overall(
                    active=config.ACTIVE, min_trades=1, limit=1, split="holdout")
                trial_id = rows[0]["id"] if rows else None
                if trial_id is not None:
                    promoted = registry.promote_if_better(
                        config.ACTIVE, config.DURATION_MINUTES, trial_id,
                        strategy.name, confirmed, result.edge, result.fitness)
                    if promoted:
                        log_event(
                            f"NOVO CAMPEAO: {strategy.name} edge={result.edge:+.4f} "
                            f"{'CONFIRMADO' if confirmed else 'nao confirmado'}")

            name, is_confirmed, edge = _champion_summary()
            log_event(
                f"onda concluida ({session_trials} tentativas) - campeao atual: "
                f"{name or '(nenhum ainda)'}")
            status.write_status(
                "generator", phase="gerando", trials_tested_session=session_trials,
                trials_tested_total=total_trials_this_run,
                champion_name=name, champion_confirmed=is_confirmed, champion_edge=edge,
                candles_available=len(candles_df), events=list(events))

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("erro numa onda do gerador: %s", exc)
            registry.log_error("generator", type(exc).__name__, str(exc), tb)
            log_event(f"ERRO: {exc}")
            status.write_status(
                "generator", phase="erro", trials_tested_total=total_trials_this_run,
                champion_name=None, champion_confirmed=False, champion_edge=None,
                events=list(events))
            time.sleep(10)


def _champion_summary():
    champ = registry.get_champion(config.ACTIVE, config.DURATION_MINUTES)
    if champ is None:
        return None, False, None
    return champ["name"], champ["confirmed"], champ["edge"]


if __name__ == "__main__":
    run_generator()
