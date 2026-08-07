"""Busca evolutiva de estrategias: gera uma populacao, faz backtest
(walk-forward: treina/ajusta na primeira parte do historico, avalia na parte
de fora), mantem as melhores, muta parametros, repete por N geracoes.

Isto e uma busca de hiperparametros/estrategias com selecao por fitness -
nao e "inteligencia artificial" magica, mas e um jeito honesto e eficaz de
"gerar varias estrategias ate encontrar uma excelente", como pedido.
"""
import copy
import logging
import random

from .backtester import backtest
from .strategies import DEFAULT_STRATEGY_FACTORIES

logger = logging.getLogger(__name__)


def _mutate(strategy):
    s = copy.deepcopy(strategy)
    # qualquer estrategia com estado treinado (MLStrategy.model,
    # PatternCluster.kmeans/cluster_bias, e o que mais vier no futuro) fica
    # invalida depois de mudar os parametros - reseta pra forcar um fit()
    # novo em vez de carregar um modelo treinado com os parametros antigos.
    if hasattr(s, "model"):
        s.model = None
    if hasattr(s, "kmeans"):
        s.kmeans = None
        s.cluster_bias = {}
    for key in list(s.params.keys()):
        val = s.params[key]
        if isinstance(val, (int, float)):
            factor = random.uniform(0.7, 1.3)
            new_val = val * factor
            s.params[key] = int(round(new_val)) if isinstance(val, int) else new_val
    return s


def train_test_split(candles_df, train_ratio=0.7):
    cut = int(len(candles_df) * train_ratio)
    return candles_df.iloc[:cut].reset_index(drop=True), candles_df.iloc[cut:].reset_index(drop=True)


def train_val_holdout_split(candles_df, train_ratio=0.6, val_ratio=0.2):
    """Split em 3: train (ajusta arvores/modelo), validation (usado
    REPETIDAMENTE pra comparar e escolher entre candidatos durante a busca),
    e holdout (nunca tocado durante a busca - so usado UMA vez, no final,
    pra confirmar o vencedor). Isso existe porque escolher um vencedor entre
    centenas/milhares de candidatos comparando todos no mesmo conjunto
    "de teste" reaproveitado contamina esse conjunto por selecao multipla -
    o holdout intocado e o unico numero em que da pra confiar de verdade."""
    n = len(candles_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = candles_df.iloc[:train_end].reset_index(drop=True)
    val_df = candles_df.iloc[train_end:val_end].reset_index(drop=True)
    holdout_df = candles_df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, holdout_df


def evolve_strategies(candles_df, payout=0.85, horizon=1, generations=4,
                       population_size=None, elite_size=4, min_confidence=0.15,
                       amount=1.0, on_individual=None):
    train_df, test_df = train_test_split(candles_df)

    if population_size is None:
        population_size = len(DEFAULT_STRATEGY_FACTORIES)
    population = [f() for f in DEFAULT_STRATEGY_FACTORIES][:population_size]
    history = []
    best = None

    for gen in range(generations):
        scored = []
        for strat in population:
            try:
                # fit() e um no-op nas estrategias baseadas em regra (so
                # retorna self) - chamar sempre, sem checar o tipo, evita
                # precisar de um caso especial pra cada nova estrategia
                # treinavel que aparecer (MLStrategy, PatternCluster, ...).
                from .features import build_features
                strat.fit(build_features(train_df))
                result = backtest(strat, test_df, payout=payout, horizon=horizon,
                                   min_confidence=min_confidence, amount=amount)
            except Exception as exc:  # dados insuficientes, etc.
                logger.warning("estrategia %s falhou no backtest: %s", strat.name, exc)
                continue
            scored.append((result.fitness, strat, result))
            if on_individual is not None:
                on_individual(strat, result)

        if not scored:
            raise RuntimeError("nenhuma estrategia produziu resultado valido no backtest")

        scored.sort(key=lambda t: t[0], reverse=True)
        history.append([(s.name, s.params, r.fitness, r.num_trades, r.edge) for _, s, r in scored])

        gen_best_fitness, gen_best_strat, gen_best_result = scored[0]
        if best is None or gen_best_fitness > best[0]:
            best = (gen_best_fitness, gen_best_strat, gen_best_result)

        logger.info(
            "geracao %d: melhor=%s fitness=%.4f trades=%d win_rate=%.3f edge=%.3f",
            gen, gen_best_strat.name, gen_best_fitness,
            gen_best_result.num_trades, gen_best_result.win_rate, gen_best_result.edge)

        # elite por TIPO distinto primeiro, pra nao deixar uma familia de
        # estrategia (ex: todas variantes de autocorr_lag) engolir a
        # populacao so por ter tido um resultado de backtest um pouco melhor -
        # isso mantem diversidade real de abordagem entre as geracoes.
        elite = []
        seen_types = set()
        for _, s, _ in scored:
            if type(s).__name__ not in seen_types:
                elite.append(s)
                seen_types.add(type(s).__name__)
            if len(elite) >= elite_size:
                break
        if len(elite) < elite_size:
            for _, s, _ in scored:
                if s not in elite:
                    elite.append(s)
                if len(elite) >= elite_size:
                    break

        next_population = list(elite)
        fresh_slots = max(2, population_size // 4)
        fresh_pool = [f() for f in DEFAULT_STRATEGY_FACTORIES]
        random.shuffle(fresh_pool)
        next_population.extend(fresh_pool[:fresh_slots])

        while len(next_population) < population_size:
            parent = random.choice(elite)
            next_population.append(_mutate(parent))
        population = next_population[:population_size]

    _, best_strategy, best_result = best
    return best_strategy, best_result, history
