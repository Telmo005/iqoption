"""Camada fina sobre iqoptionapi usando apenas os caminhos confirmados como
funcionais contra os servidores atuais da IQ Option (testado manualmente):

  - connect(), get_balance(), get_balances(), change_balance() -> OK
  - get_candles() -> OK
  - buy_digital_spot_v2() com pares "-OTC" -> OK (confirmado com ordem real)

Os seguintes caminhos foram testados e estao QUEBRADOS na biblioteca hoje
(travam esperando uma resposta que o servidor nao envia mais, ou retornam
"invalid instrument"/"asset not found" para pares nao-OTC). Ver issues
publicas: iqoptionapi/iqoptionapi#101, #102, #110, #112.
  - buy_digital_spot() (v1)
  - buy_digital_spot_v2() com pares nao-OTC
  - get_digital_underlying_list_data()
  - get_available_leverages(), buy_order() (CFD/Forex/Cripto com alavancagem)
  - get_position_history(), check_win_digital(), get_digital_position()

Por isso este modulo:
  - forca o uso de pares "-OTC" para digitais (ver config.ACTIVE)
  - calcula o resultado (win/loss) por DELTA DE SALDO em vez de usar os
    endpoints de historico/checagem quebrados
  - recusa executar o modo alavancagem ate o caminho ser corrigido
"""
import logging
import sys
import threading
import time

import pandas as pd

from iqoptionapi.stable_api import IQ_Option

from . import config

logger = logging.getLogger(__name__)

# a biblioteca iqoptionapi (nao-oficial) ja tem endpoints DOCUMENTADOS que
# travam esperando uma resposta que o servidor nao manda mais (ver buy_order
# no topo deste arquivo) - descoberto na pratica que get_candles tambem pode
# travar assim depois de uma queda de conexao mal-sucedida ("Connection is
# already closed" seguido de silencio total, 100% de CPU, por HORAS). Toda
# chamada de rede roda numa thread DAEMON dedicada (uma nova a cada chamada,
# nao um pool compartilhado) com prazo maximo - se estourar, a chamada e
# tratada como falha e a thread travada fica abandonada. Precisa ser daemon
# (nao um ThreadPoolExecutor comum, que usa threads NAO-daemon): sem isso,
# uma unica chamada travada seguraria o processo inteiro na hora de encerrar,
# e travamentos repetidos esgotariam um pool de tamanho fixo.
CALL_TIMEOUT_SECONDS = 30


class BrokerError(RuntimeError):
    pass


class Broker:
    def __init__(self, email=None, password=None):
        email = email or config.IQ_EMAIL
        password = password or config.IQ_PASSWORD
        if not email or not password:
            raise BrokerError(
                "defina IQ_EMAIL e IQ_PASSWORD nas variaveis de ambiente")
        self.iq = IQ_Option(email, password)
        self.balance_mode = "REAL" if config.ARM_REAL_MONEY else "PRACTICE"

    def connect(self, max_attempts=4):
        """Tenta conectar com retry (blips de rede/DNS passageiros nao devem
        derrubar o processo inteiro - ja aconteceu em teste real).

        self.iq.connect() passa pelo mesmo prazo maximo das outras chamadas
        de rede (_call_with_timeout) - sem isso, essa era a UNICA chamada de
        rede deste arquivo sem protecao: se travasse silenciosamente (mesma
        categoria de bug ja vista em get_candles, que prendeu o executor
        travado por horas), o processo ficava preso pra sempre em
        'conectando...', sem nunca cair no except nem tentar de novo."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                check, reason = self._call_with_timeout(self.iq.connect)
                if not check:
                    last_error = reason
                else:
                    self._call_with_timeout(self.iq.change_balance, self.balance_mode)
                    logger.info("conectado, operando em conta %s", self.balance_mode)
                    return self
            except Exception as exc:
                last_error = str(exc)
            if attempt < max_attempts:
                wait = min(30, 3 * attempt)
                logger.warning(
                    "falha ao conectar (tentativa %d/%d): %s - tentando de novo em %ds...",
                    attempt, max_attempts, last_error, wait)
                time.sleep(wait)
        raise BrokerError(f"falha ao conectar apos {max_attempts} tentativas: {last_error}")

    def ensure_connected(self):
        """O websocket cai sozinho depois de alguns minutos sem trafego (ex:
        durante uma busca longa que so faz computacao local, sem rede) - sem
        isso, a proxima chamada de rede trava pra sempre esperando resposta
        de uma conexao morta. Chamado antes de toda operacao de rede.

        Isto e so a checagem OTIMISTA (rapida, sem round-trip). Quando o
        servidor derruba a conexao sem um fechamento 'limpo', essa checagem
        as vezes ainda reporta 'conectado' e so a chamada de verdade falha -
        por isso os metodos publicos tambem usam _call() abaixo, que
        reconecta e tenta de novo se a chamada real falhar."""
        try:
            connected = self.iq.check_connect()
        except Exception:
            connected = False
        if not connected:
            logger.warning("conexao com a IQ Option caiu, reconectando...")
            self.connect()

    def _call_with_timeout(self, fn, *args, **kwargs):
        outcome = {}

        def _run():
            try:
                outcome["value"] = fn(*args, **kwargs)
            except BaseException as exc:  # repassa qualquer excecao pra thread principal
                outcome["error"] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=CALL_TIMEOUT_SECONDS)
        if t.is_alive():
            raise BrokerError(
                f"{getattr(fn, '__name__', fn)} nao respondeu em "
                f"{CALL_TIMEOUT_SECONDS}s (biblioteca travada) - abandonando")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def _call(self, fn, *args, **kwargs):
        """Roda fn(*args, **kwargs) com prazo maximo (ver _call_with_timeout);
        se falhar (por excecao OU por travar), forca reconexao e tenta mais
        UMA vez antes de desistir de vez."""
        try:
            return self._call_with_timeout(fn, *args, **kwargs)
        except BrokerError as exc:
            logger.warning(
                "chamada de rede falhou (%s), forcando reconexao e "
                "tentando de novo...", exc)
        except Exception as exc:
            logger.warning(
                "chamada de rede falhou (%s: %s), forcando reconexao e "
                "tentando de novo...", type(exc).__name__, exc)
        self.connect()
        return self._call_with_timeout(fn, *args, **kwargs)

    def get_balance(self):
        self.ensure_connected()
        return self._call(self.iq.get_balance)

    def reset_practice_balance(self):
        """So mexe na conta PRACTICE, sempre - nao existe 'resetar saldo' pra
        conta REAL (nao faz sentido, e seria perigoso demais deixar essa
        chamada depender do ARM_REAL_MONEY de alguma forma)."""
        self.ensure_connected()
        if self.balance_mode != "PRACTICE":
            self._call_with_timeout(self.iq.change_balance, "PRACTICE")
        self._call(self.iq.reset_practice_balance)
        return self._call(self.iq.get_balance)

    def get_candles_df(self, active, timeframe_seconds, count):
        """A API da IQ Option limita ~1000 candles por chamada, entao pagina
        pra tras (endtime decrescente) ate juntar 'count' candles."""
        self.ensure_connected()
        all_candles = {}
        endtime = time.time()
        attempts = 0
        while len(all_candles) < count and attempts < 50:
            batch = min(count - len(all_candles), 1000)
            raw = self._call(self.iq.get_candles, active, timeframe_seconds, batch, endtime)
            attempts += 1
            if not raw:
                break
            new_rows = [c for c in raw if c["from"] not in all_candles]
            if not new_rows:
                break
            for c in new_rows:
                all_candles[c["from"]] = c
            endtime = min(c["from"] for c in new_rows) - 1

        if not all_candles:
            raise BrokerError(f"nao consegui obter candles para {active}")

        df = pd.DataFrame(list(all_candles.values()))
        df = df.sort_values("from").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["from"], unit="s")
        return df[["timestamp", "from", "open", "close", "min", "max", "volume"]].rename(
            columns={"min": "low", "max": "high"})

    def _ensure_otc(self, active):
        if not active.upper().endswith("-OTC"):
            logger.warning(
                "%s nao e OTC: buy_digital_spot_v2 esta quebrado para pares "
                "nao-OTC nesta biblioteca hoje. Use um ativo '-OTC'.", active)
        return active

    def place_digital_trade(self, active, amount, action, duration_minutes):
        self.ensure_connected()
        active = self._ensure_otc(active)
        status, order_id = self._call(
            self.iq.buy_digital_spot_v2, active, amount, action, duration_minutes)
        if not status:
            raise BrokerError(f"falha ao abrir operacao: {order_id}")
        return order_id

    def place_leveraged_trade(self, *args, **kwargs):
        # buy_order() (CFD/Forex/Cripto com alavancagem) fica pendurado
        # indefinidamente contra os servidores atuais da IQ Option (testado).
        # Recusa explicita em vez de travar o processo silenciosamente.
        raise BrokerError(
            "modo alavancagem indisponivel: buy_order() nao responde no "
            "backend atual da IQ Option (confirmado em teste real e em "
            "issues publicas da lib). Ligue ARM_LEVERAGE apenas depois que "
            "esse caminho for corrigido/reescrito.")

    def wait_and_get_trade_pnl(self, duration_minutes, buffer_seconds=5, timeout=None, heartbeat=None):
        """Calcula o resultado de uma operacao pelo delta de saldo.

        Os endpoints de historico/checagem de resultado da biblioteca estao
        quebrados hoje (ver docstring do modulo), entao lemos o saldo antes
        de abrir a operacao e de novo apos a expiracao: a diferenca e o P/L
        exato daquela operacao, desde que nenhuma outra operacao seja aberta
        no meio do caminho (ver config.MAX_CONCURRENT_TRADES=1).

        'heartbeat', se passado, e chamado a cada poucos segundos durante a
        espera - sem isso, uma espera de 60s+ sem nenhuma atualizacao de
        status fazia o painel achar que o processo tinha travado/fechado
        (o limiar de 'sem resposta' e bem menor que 60s de proposito, pra
        pegar quedas de verdade rapido)."""
        wait_seconds = duration_minutes * 60 + buffer_seconds
        if timeout is not None:
            wait_seconds = min(wait_seconds, timeout)
        end = time.time() + wait_seconds
        while time.time() < end:
            if heartbeat:
                heartbeat()
            time.sleep(min(5, max(0, end - time.time())))
        self.ensure_connected()
        return self.get_balance()


def connect_forever(email=None, password=None, on_retry=None, max_wait_seconds=60):
    """Pra usar no startup dos processos (sync.py, executor.py): se a
    internet estiver fora bem no momento em que o processo inicia, Broker.connect()
    sozinho desiste depois de 4 tentativas (~20-30s) e derruba o processo
    inteiro, exigindo reinicio manual. Aqui nao ha nada 'em andamento' pra
    perder esperando - entao vale a pena insistir indefinidamente (espera
    cresce ate um teto) em vez de desistir."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return Broker(email, password).connect()
        except BrokerError as exc:
            wait = min(max_wait_seconds, 5 * attempt)
            if on_retry:
                on_retry(attempt, str(exc), wait)
            time.sleep(wait)
