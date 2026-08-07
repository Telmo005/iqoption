"""Sincronizador de dados: o UNICO processo que fala com a IQ Option so pra
manter o cache local de candles atualizado. Conecta, busca so os candles
novos desde a ultima sincronizacao (incremental, nao baixa tudo de novo),
desconecta a atencao (mantem a conexao mas fica ocioso) e espera o proximo
ciclo. Gerador e executor nunca buscam candles direto da IQ Option.

Uso:
    cd iqoptionapi/src
    python -m ai_trader.sync
"""
import logging
import time
import traceback
from collections import deque

from . import candle_store, config, registry, status
from .broker import Broker, BrokerError, connect_forever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_trader.sync")


def run_sync():
    events = deque(maxlen=40)

    def log_event(text):
        events.appendleft(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}")
        logger.info(text)

    log_event("sincronizador iniciado")
    status.write_status("sync", phase="conectando", last_sync_at=None,
                        candles_cached=0, events=list(events))

    def _on_retry(attempt, error, wait):
        logger.warning(
            "sem internet/conexao ao iniciar (tentativa %d): %s - tentando "
            "de novo em %ds (nao vou desistir)...", attempt, error, wait)
        status.write_status(
            "sync", phase="conectando", last_sync_at=None,
            candles_cached=candle_store.count_candles(config.ACTIVE, config.DURATION_MINUTES),
            events=[f"{time.strftime('%Y-%m-%d %H:%M:%S')}  sem conexao, tentando de novo "
                    f"(tentativa {attempt})..."])

    # falha depois de conectado tenta de novo logo (a internet pode ter
    # oscilado so um instante); so espera o ciclo inteiro (1h por padrao)
    # depois de um ciclo que teve SUCESSO.
    RETRY_AFTER_ERROR_SECONDS = min(300, config.SYNC_INTERVAL_SECONDS)

    broker = None
    last_sync_at = None
    while True:
        try:
            if broker is None:
                broker = connect_forever(on_retry=_on_retry)
                log_event("conectado a IQ Option")

            broker.ensure_connected()
            saved = candle_store.sync(
                broker, config.ACTIVE, config.DURATION_MINUTES,
                initial_backfill=config.GENERATOR_CANDLE_WINDOW)
            total = candle_store.count_candles(config.ACTIVE, config.DURATION_MINUTES)
            log_event(f"sincronizado: {saved} candles novos, {total} no total no cache")
            last_sync_at = time.time()
            status.write_status(
                "sync", phase="em_dia", last_sync_at=last_sync_at,
                candles_cached=total, candles_new=saved,
                next_sync_in_seconds=config.SYNC_INTERVAL_SECONDS,
                events=list(events))
            next_wait = config.SYNC_INTERVAL_SECONDS

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("erro na sincronizacao: %s", exc)
            registry.log_error("sync", type(exc).__name__, str(exc), tb)
            log_event(f"ERRO: {exc} - tentando de novo em {RETRY_AFTER_ERROR_SECONDS}s")
            status.write_status(
                "sync", phase="erro", last_sync_at=last_sync_at,
                candles_cached=candle_store.count_candles(config.ACTIVE, config.DURATION_MINUTES),
                events=list(events))
            broker = None  # forca reconexao completa no proximo ciclo
            next_wait = RETRY_AFTER_ERROR_SECONDS

        cached = candle_store.count_candles(config.ACTIVE, config.DURATION_MINUTES)
        status.write_status(
            "sync", phase="aguardando", last_sync_at=last_sync_at,
            candles_cached=cached, next_sync_in_seconds=next_wait, events=list(events))

        # dorme em pedacos (nao um sleep so) e reescreve o heartbeat a cada
        # pedaco - senao, com o intervalo padrao de 1h, o dashboard passa a
        # maior parte do tempo achando (por causa do heartbeat velho) que
        # este processo parou, mesmo rodando normal. last_sync_at continua
        # fixo no horario do ultimo sync de verdade, so o heartbeat (updated_at,
        # carimbado automatico pelo write_status) avanca.
        HEARTBEAT_CHUNK_SECONDS = 20
        remaining = next_wait
        while remaining > 0:
            chunk = min(HEARTBEAT_CHUNK_SECONDS, remaining)
            time.sleep(chunk)
            remaining -= chunk
            status.write_status(
                "sync", phase="aguardando", last_sync_at=last_sync_at,
                candles_cached=cached, next_sync_in_seconds=remaining, events=list(events))


if __name__ == "__main__":
    run_sync()
