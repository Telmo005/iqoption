"""Configuracao central e interruptores de seguranca do AI trader."""
import os
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"


def _load_dotenv(path):
    """Le um arquivo .env simples (KEY=VALUE por linha) e preenche
    os.environ so onde a variavel ainda nao existe - assim uma variavel de
    ambiente real (setada na sessao do terminal) sempre tem prioridade sobre
    o arquivo."""
    if not path.exists():
        return
    # utf-8-sig (nao utf-8 puro) porque o PowerShell (Out-File -Encoding
    # utf8) grava um BOM no inicio do arquivo por padrao - sem isso, o BOM
    # gruda na primeira chave da primeira linha (vira "﻿IQ_EMAIL" em vez
    # de "IQ_EMAIL") e a variavel some silenciosamente. utf-8-sig ignora o
    # BOM quando ele existe e nao muda nada quando nao existe.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ENV_FILE)


def _env_bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")

# senha do PAINEL WEB (nao confundir com IQ_PASSWORD, que e a senha da IQ
# Option). Vazio = sem senha nenhuma (comportamento original, pensado so
# pra rede local/confiavel). Se o painel for exposto na internet (ex: numa
# VPS), defina isso ANTES - sem senha, qualquer um que ache o IP consegue
# resetar saldo, trocar campeao, etc.
DASHBOARD_PASSWORD = os.getenv("IQ_DASHBOARD_PASSWORD", "")

# ---- interruptores de seguranca (todos desligados por padrao) ----
# Ligue ARM_REAL_MONEY=1 no ambiente para operar na conta REAL em vez de PRACTICE.
ARM_REAL_MONEY = _env_bool("ARM_REAL_MONEY", False)

# Ligue ARM_LEVERAGE=1 para tentar operar CFD/Forex/Cripto com alavancagem (buy_order).
# ATENCAO: hoje (verificado em testes reais) o endpoint de alavancagem da biblioteca
# nao responde (trava) contra os servidores atuais da IQ Option. Mesmo armado, o
# broker recusa executar ate esse caminho ser corrigido - ver broker.py.
ARM_LEVERAGE = _env_bool("ARM_LEVERAGE", False)

# ---- parametros de operacao ----
ACTIVE = os.getenv("IQ_ACTIVE", "EURUSD-OTC")  # pares nao-OTC estao quebrados na lib hoje
DURATION_MINUTES = int(os.getenv("IQ_DURATION", "1"))
AMOUNT = float(os.getenv("IQ_AMOUNT", "1"))
MIN_TRADE_AMOUNT = float(os.getenv("IQ_MIN_TRADE_AMOUNT", "1"))  # minimo aceito pela IQ Option
MAX_AMOUNT_PER_TRADE = float(os.getenv("IQ_MAX_AMOUNT", "5"))
MAX_CONCURRENT_TRADES = int(os.getenv("IQ_MAX_CONCURRENT", "1"))
MAX_DAILY_LOSS = float(os.getenv("IQ_MAX_DAILY_LOSS", "20"))

# quantas operacoes reais (PRACTICE por padrao) rodar no teste pra frente,
# apos a busca autonoma achar um candidato confirmado no holdout
FORWARD_TEST_TRADES = int(os.getenv("IQ_FORWARD_TEST_TRADES", "50"))

# se ja existir, na base persistida, uma estrategia CONFIRMADA no holdout
# pra este ativo/duracao, com no maximo essa idade, usa ela direto em vez de
# gastar tempo procurando de novo do zero. 0 desliga o reaproveitamento.
REUSE_CONFIRMED_MAX_AGE_HOURS = float(os.getenv("IQ_REUSE_MAX_AGE_HOURS", "6"))

# ---- gerador continuo (generator.py, processo separado, 100% local) ----
GENERATOR_WAVE_SECONDS = int(os.getenv("IQ_GENERATOR_WAVE_SECONDS", "60"))
GENERATOR_CANDLE_WINDOW = int(os.getenv("IQ_GENERATOR_CANDLE_WINDOW", "25000"))

# de quanto em quanto tempo o sincronizador (sync.py) busca candles novos
SYNC_INTERVAL_SECONDS = int(os.getenv("IQ_SYNC_INTERVAL_SECONDS", "3600"))

# duracao de cada onda de busca rodada ENTRE operacoes do forward test, na
# MESMA conexao (abrir uma segunda conexao concorrente pra mesma conta
# causou bloqueio de login da IQ Option em teste real - ver run_live_loop).
# 0 desliga a busca continua durante o forward test.
LIVE_SEARCH_WAVE_SECONDS = int(os.getenv("IQ_LIVE_SEARCH_WAVE_SECONDS", "30"))

# payout tipico de opcao digital (ajuste conforme Iq.get_digital_payout(ACTIVE))
DEFAULT_PAYOUT = float(os.getenv("IQ_DEFAULT_PAYOUT", "0.85"))

# limiar minimo de "edge" (probabilidade estimada - probabilidade de equilibrio)
# que uma estrategia precisa ter no backtest para ser promovida a operar ao vivo.
MIN_EDGE_TO_TRADE = float(os.getenv("IQ_MIN_EDGE", "0.03"))

HEDGE_ENABLED = _env_bool("HEDGE_ENABLED", True)
HEDGE_CONFIDENCE_THRESHOLD = float(os.getenv("IQ_HEDGE_THRESHOLD", "0.55"))
HEDGE_SIZE_RATIO = float(os.getenv("IQ_HEDGE_SIZE_RATIO", "0.5"))

# ---- busca evolutiva (familias fixas em strategies.py) ----
EVOLVE_GENERATIONS = int(os.getenv("IQ_EVOLVE_GENERATIONS", "4"))
EVOLVE_POPULATION = os.getenv("IQ_EVOLVE_POPULATION")  # None = usa todas as familias padrao
EVOLVE_POPULATION = int(EVOLVE_POPULATION) if EVOLVE_POPULATION else None

# ---- programacao genetica (arvores de expressao livres) ----
GP_ENABLED = _env_bool("GP_ENABLED", True)
GP_POPULATION = int(os.getenv("IQ_GP_POPULATION", "40"))
GP_GENERATIONS = int(os.getenv("IQ_GP_GENERATIONS", "10"))
GP_MAX_DEPTH = int(os.getenv("IQ_GP_MAX_DEPTH", "5"))

# ---- busca autonoma (ondas continuas ate achar algo bom ou acabar o tempo) ----
AUTONOMOUS_SECONDS = int(os.getenv("IQ_AUTONOMOUS_SECONDS", "180"))
GP_WAVE_GENERATIONS = int(os.getenv("IQ_GP_WAVE_GENERATIONS", "5"))
MAX_RUIN_PROBABILITY = float(os.getenv("IQ_MAX_RUIN_PROBABILITY", "0.15"))

# ---- simulacao de banca (Monte Carlo + "saldo final simulado" por estrategia) ----
# saldo de referencia usado SO pra simular/comparar estrategias (backtest
# replay e Monte Carlo) - nao tem nenhuma ligacao com o saldo real da conta
# PRACTICE/REAL na IQ Option, que e independente disso.
BANKROLL_STARTING_BALANCE = float(os.getenv("IQ_BANKROLL_START", "10000"))
BANKROLL_BET_FRACTION = float(os.getenv("IQ_BANKROLL_BET_FRACTION", "0.1"))
BANKROLL_HORIZON_TRADES = int(os.getenv("IQ_BANKROLL_HORIZON", "200"))
BANKROLL_TARGET_MULTIPLIER = float(os.getenv("IQ_BANKROLL_TARGET_MULTIPLIER", "2.0"))
BANKROLL_RUIN_FRACTION = float(os.getenv("IQ_BANKROLL_RUIN_FRACTION", "0.2"))
BANKROLL_NUM_SIMULATIONS = int(os.getenv("IQ_BANKROLL_SIMULATIONS", "5000"))
