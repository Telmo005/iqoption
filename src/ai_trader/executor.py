"""Executor: o UNICO processo que abre operacoes de verdade. Nunca gera
estrategia nenhuma - so consulta o "campeao atual" (registry.champions,
mantido pelo generator.py rodando em outro processo) e opera com ele,
reconferindo periodicamente se o campeao mudou.

Dinheiro REAL exige campeao CONFIRMADO no holdout - essa trava fica no
codigo, nao numa flag/env, de proposito (ver comentario mais abaixo).

Uso:
    cd iqoptionapi/src
    python -m ai_trader.executor
"""
import logging
import sys
import time
import traceback
from collections import deque

import pandas as pd

from . import bankroll_sim, config, registry, status
from .broker import Broker, BrokerError, connect_forever
from .features import build_features
from .hedge import decide_hedge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_trader.executor")

CHAMPION_POLL_SECONDS = 15


def run_executor(max_trades=None):
    mode = "REAL" if config.ARM_REAL_MONEY else "PRACTICE"

    def _on_retry(attempt, error, wait):
        logger.warning(
            "sem internet/conexao ao iniciar (tentativa %d): %s - tentando "
            "de novo em %ds (nao vou desistir)...", attempt, error, wait)
        status.write_status(
            "executor", phase="iniciando",
            events=[f"{time.strftime('%Y-%m-%d %H:%M:%S')}  sem conexao, tentando de novo "
                    f"(tentativa {attempt})..."])

    broker = connect_forever(on_retry=_on_retry)

    events = deque(maxlen=40)

    def log_event(text):
        events.appendleft(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}")
        logger.info(text)

    def push_status(phase, **extra):
        status.write_status(
            "executor", phase=phase, active=config.ACTIVE,
            duration_minutes=config.DURATION_MINUTES, account_mode=mode,
            balance=balance, daily_pnl=daily_pnl, trades_done=trades_done,
            wins=wins, losses=losses, max_trades=max_trades,
            max_daily_loss=config.MAX_DAILY_LOSS,
            strategy_name=strategy.name if strategy else None,
            strategy_confirmed=bool(champion and champion["confirmed"]),
            strategy_edge=champion["edge"] if champion else None,
            events=list(events), **extra)

    daily_pnl = 0.0
    trades_done = 0
    wins = 0
    losses = 0
    balance = broker.get_balance()
    strategy = None
    champion = None
    active_champion_id = None

    log_event(f"executor iniciado | conta={mode} ativo={config.ACTIVE}")
    push_status("iniciando")

    while max_trades is None or trades_done < max_trades:
        try:
            champion = registry.get_champion(config.ACTIVE, config.DURATION_MINUTES)
            if champion is None:
                log_event("nenhuma estrategia disponivel ainda - esperando o gerador achar uma...")
                push_status("aguardando_estrategia")
                time.sleep(CHAMPION_POLL_SECONDS)
                continue

            if champion["id"] != active_champion_id:
                strategy = champion["strategy"]
                active_champion_id = champion["id"]
                log_event(
                    f"usando estrategia: {champion['name']} "
                    f"({'confirmada' if champion['confirmed'] else 'nao confirmada'}, "
                    f"edge={champion['edge']:+.4f})")

            # ---- ESTA TRAVA NAO E CONFIGURAVEL POR FLAG/ENV -----------
            # dinheiro real so opera com campeao confirmado no holdout.
            if config.ARM_REAL_MONEY and not champion["confirmed"]:
                log_event("dinheiro real bloqueado: campeao atual nao confirmado no holdout")
                push_status("bloqueado_dinheiro_real")
                time.sleep(CHAMPION_POLL_SECONDS)
                continue
            # -------------------------------------------------------------

            if daily_pnl <= -config.MAX_DAILY_LOSS:
                logger.error("limite de perda diaria atingido. parando.")
                log_event("limite de perda diaria atingido - parando")
                push_status("parado")
                break

            push_status("verificando_sinal")
            recent = broker.get_candles_df(
                config.ACTIVE, config.DURATION_MINUTES * 60, count=200)
            features_df = build_features(recent)
            action, confidence = strategy.predict_last(features_df)

            if action is None or (isinstance(action, float) and pd.isna(action)):
                log_event("sem sinal, aguardando proximo candle")
                wait_end = time.time() + config.DURATION_MINUTES * 60
                while time.time() < wait_end:
                    push_status("aguardando_sinal")
                    time.sleep(min(5, max(0, wait_end - time.time())))
                continue

            amount = min(config.AMOUNT, config.MAX_AMOUNT_PER_TRADE)

            balance_before = broker.get_balance()
            # so mexe no saldo PRACTICE, nunca no REAL - a conta PRACTICE
            # existe so pra manter o sistema operando, entao se ficar sem
            # saldo pra cobrir a proxima aposta ela se renova sozinha.
            if mode == "PRACTICE" and balance_before < amount:
                log_event(
                    f"saldo PRACTICE insuficiente ({balance_before:.2f} < {amount:.2f}) - "
                    f"renovando automaticamente")
                try:
                    balance_before = broker.reset_practice_balance()
                    balance = balance_before
                    log_event(f"saldo PRACTICE renovado: {balance_before:.2f}")
                except BrokerError as exc:
                    logger.error("falha ao renovar saldo PRACTICE: %s", exc)
                    log_event(f"falha ao renovar saldo PRACTICE: {exc}")
                    registry.log_error("executor", "BrokerError", f"falha ao renovar saldo: {exc}", None)
                    time.sleep(10)
                    continue

            log_event(f"sinal: {action} confianca={confidence:.2f} valor={amount:.2f}")
            push_status("operando")

            try:
                broker.place_digital_trade(
                    config.ACTIVE, amount, action, config.DURATION_MINUTES)
            except BrokerError as exc:
                logger.error("falha ao abrir operacao: %s", exc)
                log_event(f"falha ao abrir operacao: {exc}")
                registry.log_error("executor", "BrokerError", str(exc), None)
                time.sleep(5)
                continue

            hedge_decision = decide_hedge(action, confidence, amount)
            if hedge_decision:
                hedge_action, hedge_amount = hedge_decision
                log_event(f"hedge: {hedge_action} valor={hedge_amount:.2f}")
                try:
                    broker.place_digital_trade(
                        config.ACTIVE, hedge_amount, hedge_action, config.DURATION_MINUTES)
                except BrokerError as exc:
                    logger.error("falha ao abrir hedge: %s", exc)
                    log_event(f"falha ao abrir hedge: {exc}")

            push_status("aguardando_resultado")
            balance_after = broker.wait_and_get_trade_pnl(
                config.DURATION_MINUTES,
                heartbeat=lambda: push_status("aguardando_resultado"))
            pnl = balance_after - balance_before
            balance = balance_after
            daily_pnl += pnl
            trades_done += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            resultado = "GANHOU" if pnl > 0 else ("PERDEU" if pnl < 0 else "EMPATE")
            log_event(f"{resultado} pnl={pnl:+.2f} saldo={balance_after:.2f}")
            registry.record_trade(
                config.ACTIVE, config.DURATION_MINUTES, mode, strategy.name,
                action, amount, confidence, pnl, balance_after,
                is_hedge=bool(hedge_decision))
            logger.info(
                "resultado: pnl=%.2f saldo=%.2f pnl_diario=%.2f trades=%d (vitorias=%d derrotas=%d)",
                pnl, balance_after, daily_pnl, trades_done, wins, losses)
            push_status("aguardando_sinal")

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("erro inesperado no executor: %s", exc)
            registry.log_error("executor", type(exc).__name__, str(exc), tb)
            log_event(f"ERRO: {exc}")
            push_status("erro")
            time.sleep(10)

    win_pct = (wins / trades_done * 100) if trades_done else 0.0
    logger.info(
        "=== executor encerrado: %d operacoes, %d vitorias, %d derrotas "
        "(%.1f%% acerto), pnl total=%.2f ===", trades_done, wins, losses, win_pct, daily_pnl)
    log_event("executor encerrado")
    push_status("encerrado")


if __name__ == "__main__":
    run_executor(max_trades=config.FORWARD_TEST_TRADES)
