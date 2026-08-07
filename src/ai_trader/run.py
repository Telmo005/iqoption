"""Orquestrador principal do AI trader.

Uso:
    cd iqoptionapi/src
    set IQ_EMAIL=seu@email.com          (PowerShell: $env:IQ_EMAIL="...")
    set IQ_PASSWORD=suasenha
    python -m ai_trader.run

Por padrao roda 100% em conta PRACTICE (demo) e so opera opcoes digitais em
pares "-OTC" (o unico caminho de execucao confirmado funcional na biblioteca
hoje - ver broker.py). Sempre faz backtest/busca evolutiva ANTES de arriscar
qualquer operacao, e so opera ao vivo se a melhor estrategia encontrada bater
o limiar minimo de "edge" (config.MIN_EDGE_TO_TRADE). Nao ha garantia de
lucro - isto e uma ferramenta de busca/execucao de estrategia, nao uma
maquina de dinheiro.
"""
import argparse
import logging
import sys
import time
import uuid
from collections import deque

import pandas as pd

from . import bankroll_sim, config, registry, status
from .broker import Broker, BrokerError
from .diagnostics import summarize, format_report
from .evolve import evolve_strategies, train_test_split
from .features import build_features
from .genetic_program import run_gp
from .hedge import decide_hedge
from .strategies import DEFAULT_STRATEGY_FACTORIES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_trader")


def print_leaderboard(history):
    """Mostra TODAS as estrategias testadas na ultima geracao, nao so a
    vencedora - assim da pra ver que a busca de fato testou varios tipos
    (tendencia, reversao a media, momentum, autocorrelacao, ML) e nao so
    escolheu a primeira que apareceu."""
    last_gen = history[-1]
    ranked = sorted(last_gen, key=lambda row: row[2], reverse=True)
    print("\n=== Leaderboard (ultima geracao) ===")
    print(f"{'estrategia':<22}{'trades':>8}{'edge':>10}{'fitness':>12}  params")
    for name, params, fitness, num_trades, edge in ranked:
        print(f"{name:<22}{num_trades:>8}{edge:>10.4f}{fitness:>12.5f}  {params}")


def find_best_strategy(broker, count=3000):
    session_id = uuid.uuid4().hex[:12]
    logger.info("sessao: %s", session_id)

    def persist(kind, split):
        def _cb(strat, result):
            registry.record_trial(
                session_id, kind, strat.name, strat.params,
                config.ACTIVE, config.DURATION_MINUTES, result, split=split)
        return _cb

    logger.info("baixando historico de candles de %s...", config.ACTIVE)
    candles_df = broker.get_candles_df(
        config.ACTIVE, config.DURATION_MINUTES * 60, count=count)
    logger.info("%d candles obtidos", len(candles_df))

    diag = summarize(candles_df)
    print(format_report(diag))

    logger.info("rodando busca evolutiva sobre %d tipos de estrategia...",
                len(DEFAULT_STRATEGY_FACTORIES))

    best_strategy, best_result, history = evolve_strategies(
        candles_df, payout=config.DEFAULT_PAYOUT,
        horizon=1, amount=config.AMOUNT, on_individual=persist("family", "test"))

    print_leaderboard(history)

    logger.info(
        "melhor das familias fixas: %s params=%s | trades=%d win_rate=%.3f "
        "breakeven=%.3f edge=%.3f expectancy=%.4f max_drawdown=%.2f fitness=%.4f",
        best_strategy.name, best_strategy.params, best_result.num_trades,
        best_result.win_rate, best_result.breakeven_win_rate, best_result.edge,
        best_result.expectancy, best_result.max_drawdown, best_result.fitness)

    if config.GP_ENABLED:
        logger.info(
            "rodando programacao genetica (populacao=%d geracoes=%d "
            "profundidade_max=%d)...", config.GP_POPULATION,
            config.GP_GENERATIONS, config.GP_MAX_DEPTH)
        train_df, test_df = train_test_split(candles_df)
        gp_strategy, gp_result, gp_history = run_gp(
            train_df, test_df,
            population_size=config.GP_POPULATION,
            generations=config.GP_GENERATIONS,
            max_depth=config.GP_MAX_DEPTH,
            payout=config.DEFAULT_PAYOUT, amount=config.AMOUNT,
            on_individual=persist("gp", "train"))
        persist("gp", "test")(gp_strategy, gp_result)

        logger.info(
            "melhor da programacao genetica: %s | trades=%d win_rate=%.3f "
            "edge=%.3f expectancy=%.4f fitness=%.4f",
            gp_strategy.params["expr"], gp_result.num_trades, gp_result.win_rate,
            gp_result.edge, gp_result.expectancy, gp_result.fitness)
        print(f"\n=== Melhor arvore da programacao genetica ===\n{gp_strategy.params['expr']}")

        if gp_result.fitness > best_result.fitness:
            logger.info(
                "a arvore evoluida por programacao genetica superou todas "
                "as familias fixas - usando ela como estrategia vencedora.")
            best_strategy, best_result = gp_strategy, gp_result
        else:
            logger.info("nenhuma arvore da programacao genetica superou as familias fixas desta vez.")

    logger.info("re-treinando %s com todo o historico disponivel...", best_strategy.name)
    best_strategy.fit(build_features(candles_df))

    return best_strategy, best_result, candles_df


def gate_or_stop(best_result):
    if best_result.num_trades < 30:
        logger.error(
            "poucos sinais no backtest (%d trades) para confiar na estrategia. "
            "abortando - nao vou operar dinheiro/tempo real com isso.",
            best_result.num_trades)
        return False
    if best_result.edge < config.MIN_EDGE_TO_TRADE:
        logger.error(
            "edge da melhor estrategia (%.3f) abaixo do minimo exigido (%.3f). "
            "isso significa que, no historico testado, a estrategia nao supera "
            "com folga a vantagem estrutural da corretora (payout). abortando.",
            best_result.edge, config.MIN_EDGE_TO_TRADE)
        return False

    try:
        bankroll = bankroll_sim.simulate(
            best_result.trade_pnls, backtest_amount=best_result.amount,
            starting_balance=config.BANKROLL_STARTING_BALANCE,
            bet_fraction=config.BANKROLL_BET_FRACTION,
            num_trades=config.BANKROLL_HORIZON_TRADES,
            num_simulations=config.BANKROLL_NUM_SIMULATIONS,
            target_multiplier=config.BANKROLL_TARGET_MULTIPLIER,
            ruin_fraction=config.BANKROLL_RUIN_FRACTION)
    except ValueError as exc:
        logger.error("nao foi possivel simular banca: %s. abortando.", exc)
        return False

    print("\n" + bankroll_sim.format_report(bankroll))

    if bankroll["ruin_probability"] > config.MAX_RUIN_PROBABILITY:
        logger.error(
            "probabilidade de ruina da banca (%.1f%%) acima do maximo aceito "
            "(%.1f%%). o edge medio ate e positivo, mas a variancia e alta "
            "demais pra arriscar a banca com confianca. abortando.",
            bankroll["ruin_probability"] * 100, config.MAX_RUIN_PROBABILITY * 100)
        return False

    return True


def run_live_loop(broker, strategy, best_result=None, max_trades=None,
                  search_wave_seconds=0, search_cache_refresh_seconds=300):
    """Teste pra frente (forward test): opera de verdade (PRACTICE por
    padrao) em candles que ainda nem existiam quando a estrategia foi
    escolhida no backtest - a unica validacao que nao pode ser contaminada
    por overfitting de busca, porque o resultado nao existe ainda no momento
    da escolha.

    Se 'search_wave_seconds' > 0, a cada ciclo (antes de checar o sinal) roda
    uma onda curta de busca autonoma 100% LOCAL (sem rede) sobre um cache de
    candles renovado a cada 'search_cache_refresh_seconds'. A rede so e usada
    para (1) renovar esse cache de vez em quando e (2) abrir operacao/checar
    saldo - nunca fica aberta e ociosa durante a computacao da busca, que foi
    o que derrubou a conexao (o servidor fecha websocket parado) em teste
    real. Se a onda achar um candidato melhor E confirmado no holdout, troca
    a estrategia ativa."""
    if config.ARM_LEVERAGE:
        logger.warning(
            "ARM_LEVERAGE esta ligado, mas o modo alavancagem esta "
            "indisponivel hoje (buy_order nao responde no backend atual da "
            "IQ Option). Operando apenas opcoes digitais OTC.")

    logger.info(
        "=== iniciando teste pra frente (forward test) | ate %s "
        "operacoes ou perda diaria de %.2f, o que vier primeiro ===",
        max_trades if max_trades is not None else "infinitas", config.MAX_DAILY_LOSS)

    events = deque(maxlen=40)

    def log_event(text):
        events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def push_status(phase):
        status.write_status(
            phase=phase,
            active=config.ACTIVE,
            duration_minutes=config.DURATION_MINUTES,
            account_mode="REAL" if config.ARM_REAL_MONEY else "PRACTICE",
            strategy_name=strategy.name,
            strategy_params=str(strategy.params),
            strategy_confirmed=bool(best_result and best_result.edge >= config.MIN_EDGE_TO_TRADE),
            strategy_edge=best_result.edge if best_result else None,
            balance=balance,
            daily_pnl=daily_pnl,
            trades_done=trades_done,
            wins=wins,
            losses=losses,
            max_trades=max_trades,
            max_daily_loss=config.MAX_DAILY_LOSS,
            events=list(events),
        )

    daily_pnl = 0.0
    trades_done = 0
    wins = 0
    losses = 0
    balance = broker.get_balance()
    search_cache = None
    search_cache_at = 0.0
    log_event("teste pra frente iniciado")
    push_status("iniciando")
    while max_trades is None or trades_done < max_trades:
        if search_wave_seconds > 0:
            push_status("buscando")
            from .autonomous import autonomous_search
            now = time.time()
            if search_cache is None or (now - search_cache_at) > search_cache_refresh_seconds:
                broker.ensure_connected()
                search_cache = broker.get_candles_df(
                    config.ACTIVE, config.DURATION_MINUTES * 60, count=3000)
                search_cache_at = now
                logger.info("cache de candles pra busca renovado (%d candles)", len(search_cache))

            try:
                outcome, _sid = autonomous_search(search_cache, time_budget_seconds=search_wave_seconds)
            except Exception:
                logger.exception("onda de busca durante o forward test falhou, seguindo com a estrategia atual")
                outcome = None
            if outcome is not None:
                cand_strategy, cand_result, cand_bankroll, confirmed = outcome
                if confirmed and (best_result is None or cand_result.fitness > best_result.fitness):
                    logger.info(
                        "=== busca achou candidato MELHOR e CONFIRMADO durante "
                        "o forward test: %s edge=%.4f - trocando estrategia ativa ===",
                        cand_strategy.name, cand_result.edge)
                    log_event(f"nova estrategia confirmada: {cand_strategy.name} edge={cand_result.edge:.4f}")
                    strategy = cand_strategy
                    best_result = cand_result

        if daily_pnl <= -config.MAX_DAILY_LOSS:
            logger.error(
                "limite de perda diaria atingido (%.2f <= -%.2f). parando.",
                daily_pnl, config.MAX_DAILY_LOSS)
            log_event("limite de perda diaria atingido - parando")
            push_status("parado")
            break

        push_status("verificando_sinal")
        recent = broker.get_candles_df(
            config.ACTIVE, config.DURATION_MINUTES * 60, count=200)
        features_df = build_features(recent)
        action, confidence = strategy.predict_last(features_df)

        if action is None or (isinstance(action, float) and pd.isna(action)):
            logger.info("sem sinal, aguardando proximo candle...")
            log_event("sem sinal, aguardando proximo candle")
            push_status("aguardando_sinal")
            time.sleep(config.DURATION_MINUTES * 60)
            continue

        amount = min(config.AMOUNT, config.MAX_AMOUNT_PER_TRADE)
        logger.info("sinal: %s confianca=%.2f valor=%.2f", action, confidence, amount)
        log_event(f"sinal: {action} confianca={confidence:.2f} valor={amount:.2f}")
        push_status("operando")

        balance_before = broker.get_balance()

        try:
            broker.place_digital_trade(
                config.ACTIVE, amount, action, config.DURATION_MINUTES)
        except BrokerError as exc:
            logger.error("falha ao abrir operacao principal: %s", exc)
            log_event(f"falha ao abrir operacao: {exc}")
            time.sleep(5)
            continue

        hedge_decision = decide_hedge(action, confidence, amount)
        if hedge_decision:
            hedge_action, hedge_amount = hedge_decision
            logger.info("confianca baixa -> hedge: %s valor=%.2f", hedge_action, hedge_amount)
            log_event(f"hedge: {hedge_action} valor={hedge_amount:.2f}")
            try:
                broker.place_digital_trade(
                    config.ACTIVE, hedge_amount, hedge_action, config.DURATION_MINUTES)
            except BrokerError as exc:
                logger.error("falha ao abrir hedge (seguindo so com a principal): %s", exc)
                log_event(f"falha ao abrir hedge: {exc}")

        push_status("aguardando_resultado")
        balance_after = broker.wait_and_get_trade_pnl(config.DURATION_MINUTES)
        pnl = balance_after - balance_before
        balance = balance_after
        daily_pnl += pnl
        trades_done += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        logger.info(
            "resultado da rodada: pnl=%.2f saldo=%.2f pnl_diario=%.2f trades=%d "
            "(vitorias=%d derrotas=%d)",
            pnl, balance_after, daily_pnl, trades_done, wins, losses)
        resultado = "GANHOU" if pnl > 0 else ("PERDEU" if pnl < 0 else "EMPATE")
        log_event(f"{resultado} pnl={pnl:+.2f} saldo={balance_after:.2f}")
        push_status("aguardando_sinal")

    win_pct = (wins / trades_done * 100) if trades_done else 0.0
    logger.info(
        "=== teste pra frente encerrado: %d operacoes reais, %d vitorias, "
        "%d derrotas (%.1f%% de acerto), pnl total=%.2f ===",
        trades_done, wins, losses, win_pct, daily_pnl)
    log_event("teste pra frente encerrado")
    push_status("encerrado")


def main():
    parser = argparse.ArgumentParser(description="AI trader para IQ Option")
    parser.add_argument(
        "--report", action="store_true",
        help="so faz diagnostico + backtest/validacao e mostra o relatorio; "
             "nunca abre operacao, mesmo que a estrategia passe no portao.")
    parser.add_argument(
        "--autonomous", action="store_true",
        help="roda a busca autonoma (ondas continuas de programacao genetica "
             "+ familias fixas, com simulacao de banca, ate achar algo bom ou "
             "esgotar IQ_AUTONOMOUS_SECONDS) em vez da busca unica padrao.")
    args = parser.parse_args()

    mode = "REAL" if config.ARM_REAL_MONEY else "PRACTICE"
    logger.info("iniciando AI trader | conta=%s ativo=%s duracao=%dmin",
                mode, config.ACTIVE, config.DURATION_MINUTES)

    broker = Broker().connect()

    if args.autonomous:
        from .autonomous import autonomous_search

        events = deque(maxlen=40)

        def log_event(text):
            events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

        def push_search_status(strategy_name, trials_tested):
            status.write_status(
                phase="buscando", active=config.ACTIVE, duration_minutes=config.DURATION_MINUTES,
                account_mode=mode, strategy_name=strategy_name,
                strategy_params="", strategy_confirmed=False, strategy_edge=None,
                balance=broker.get_balance(), daily_pnl=0.0, trades_done=0, wins=0, losses=0,
                max_trades=config.FORWARD_TEST_TRADES, max_daily_loss=config.MAX_DAILY_LOSS,
                trials_tested=trials_tested, events=list(events))

        # reaproveita uma estrategia ja confirmada no holdout em sessao
        # anterior, se existir e for recente o bastante - evita gastar
        # minutos procurando de novo algo que ja foi achado e validado.
        reused = None
        if config.REUSE_CONFIRMED_MAX_AGE_HOURS > 0:
            reused = registry.find_confirmed_strategy(
                config.ACTIVE, config.DURATION_MINUTES,
                min_edge=config.MIN_EDGE_TO_TRADE,
                max_ruin_probability=config.MAX_RUIN_PROBABILITY,
                max_age_seconds=config.REUSE_CONFIRMED_MAX_AGE_HOURS * 3600)

        if reused is not None:
            age_min = (time.time() - reused["created_at"]) / 60
            logger.info(
                "reaproveitando estrategia ja confirmada na base (id=%s, %s, "
                "encontrada ha %.0f min): edge=%.4f fitness=%.5f",
                reused["id"], reused["name"], age_min, reused["edge"], reused["fitness"])
            log_event(f"reaproveitando estrategia confirmada da base: {reused['name']} "
                      f"(achada ha {age_min:.0f} min)")
            push_search_status(reused["name"], registry.count_trials())

            class _Result:
                pass
            best_result = _Result()
            best_result.edge = reused["edge"]
            best_result.fitness = reused["fitness"]
            best_result.num_trades = reused["num_trades"]
            best_strategy = reused["strategy"]
            bankroll = {"ruin_probability": reused["ruin_probability"]} if reused["ruin_probability"] is not None else None
            passed = True
        else:
            push_search_status("(busca inicial em andamento)", 0)
            log_event(f"baixando historico e rodando busca inicial (ate {config.AUTONOMOUS_SECONDS}s)...")
            push_search_status("(busca inicial em andamento)", 0)

            logger.info("baixando historico de candles de %s...", config.ACTIVE)
            initial_candles = broker.get_candles_df(
                config.ACTIVE, config.DURATION_MINUTES * 60, count=3000)
            logger.info("%d candles obtidos, iniciando busca 100%% local...", len(initial_candles))

            def on_progress(session_id, trials_tested):
                push_search_status("(busca inicial em andamento)", trials_tested)

            outcome, session_id = autonomous_search(initial_candles, on_progress=on_progress)
            if outcome is None:
                logger.error("nenhum candidato encontrado ainda nesta busca. abortando.")
                sys.exit(1)

            best_strategy, best_result, bankroll, passed = outcome
        ruin_pct = f"{bankroll['ruin_probability']*100:.1f}%" if bankroll else "desconhecida"

        # ---- ESTA TRAVA NAO E CONFIGURAVEL POR FLAG/ENV -----------------
        # dinheiro real so opera com estrategia confirmada no holdout. Se
        # voce quer mudar isso, mude este codigo, nao uma variavel de
        # ambiente - a decisao precisa ser deliberada, nao um valor default.
        if config.ARM_REAL_MONEY and not passed:
            logger.error(
                "ARM_REAL_MONEY ligado, mas o candidato NAO foi confirmado "
                "no holdout (edge=%.4f, prob_ruina=%s). Operar dinheiro real "
                "com isso tem alta chance de perda, baseado no proprio teste "
                "que acabamos de rodar. Abortando - sem esta trava.",
                best_result.edge, ruin_pct)
            sys.exit(1)
        # -------------------------------------------------------------------

        if not passed:
            logger.warning(
                "operando em PRACTICE sem confirmacao estatistica de edge no "
                "holdout (edge=%.4f, prob_ruina=%s) - por pedido explicito. "
                "e dinheiro de demonstracao, mas o resultado pode ser ruim "
                "como o holdout indicou.", best_result.edge, ruin_pct)

        if args.report:
            return

        run_live_loop(broker, best_strategy, best_result=best_result,
                      max_trades=config.FORWARD_TEST_TRADES,
                      search_wave_seconds=config.LIVE_SEARCH_WAVE_SECONDS)
        return

    best_strategy, best_result, candles_df = find_best_strategy(broker)

    passed = gate_or_stop(best_result)

    if args.report:
        logger.info("modo --report: encerrando sem operar (portao: %s)",
                     "passou" if passed else "nao passou")
        return

    if not passed:
        sys.exit(1)

    run_live_loop(broker, best_strategy)


if __name__ == "__main__":
    main()
