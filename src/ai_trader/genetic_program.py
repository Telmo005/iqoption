"""Programacao genetica de verdade: evolui a ESTRUTURA de uma regra de sinal
(uma arvore de expressao combinando features com operadores), nao so os
parametros de formulas ja escritas por mim (isso e o que evolve.py faz para
as 8 familias de strategies.py). Aqui o algoritmo pode descobrir combinacoes
de features/operadores que nunca foram programadas explicitamente.

Cada individuo e uma arvore: folhas sao uma feature (ex: 'rsi_14') ou uma
constante; nos internos sao operadores (soma, diferenca, produto, divisao
protegida, maior-que, min, max, negacao, valor absoluto, tanh). A arvore
inteira e avaliada vetorizada sobre o DataFrame de features e produz um
"score" continuo por candle; score > limiar = call, score < -limiar = put.

Aptidao (fitness) de cada arvore = a mesma formula de backtester.py
(expectancy penalizada por drawdown e tamanho de amostra), garantindo que a
comparacao com as estrategias de evolve.py seja igual (mesma economia de
payout, mesmo criterio de selecao).
"""
import random

import numpy as np
import pandas as pd

from .backtester import backtest
from .features import FEATURE_COLUMNS
from .strategies import Strategy, make_signals_df

# ---------------------------------------------------------------- primitivas

def _protected_div(a, b):
    b = np.where(np.abs(b) < 1e-6, 1e-6, b)
    return a / b


BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _protected_div,
    "gt": lambda a, b: np.where(a > b, 1.0, -1.0),
    "max": np.maximum,
    "min": np.minimum,
}
UNARY_OPS = {
    "neg": lambda a: -a,
    "abs": np.abs,
    "tanh": np.tanh,
}
CONSTANTS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


class Node:
    __slots__ = ("kind", "value", "children")

    def __init__(self, kind, value=None, children=None):
        self.kind = kind  # "terminal" | "const" | "unary" | "binary"
        self.value = value
        self.children = children or []

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def copy(self):
        return Node(self.kind, self.value, [c.copy() for c in self.children])

    def __repr__(self):
        if self.kind == "terminal":
            return self.value
        if self.kind == "const":
            return f"{self.value:.3g}"
        if self.kind == "unary":
            return f"{self.value}({self.children[0]!r})"
        return f"{self.value}({self.children[0]!r}, {self.children[1]!r})"


def _random_terminal(terminal_set):
    if random.random() < 0.7:
        return Node("terminal", random.choice(terminal_set))
    return Node("const", random.choice(CONSTANTS))


def random_tree(max_depth, depth=0, terminal_set=FEATURE_COLUMNS):
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return _random_terminal(terminal_set)
    if random.random() < 0.85:
        op = random.choice(list(BINARY_OPS.keys()))
        return Node("binary", op,
                     [random_tree(max_depth, depth + 1, terminal_set),
                      random_tree(max_depth, depth + 1, terminal_set)])
    op = random.choice(list(UNARY_OPS.keys()))
    return Node("unary", op, [random_tree(max_depth, depth + 1, terminal_set)])


def evaluate(node, df):
    if node.kind == "terminal":
        return df[node.value].to_numpy(dtype=float)
    if node.kind == "const":
        return np.full(len(df), node.value)
    if node.kind == "unary":
        return UNARY_OPS[node.value](evaluate(node.children[0], df))
    a = evaluate(node.children[0], df)
    b = evaluate(node.children[1], df)
    return BINARY_OPS[node.value](a, b)


def _nodes_list(node):
    nodes = [node]
    for c in node.children:
        nodes.extend(_nodes_list(c))
    return nodes


def crossover(tree_a, tree_b, max_depth=6):
    a = tree_a.copy()
    b = tree_b.copy()
    na = random.choice(_nodes_list(a))
    nb = random.choice(_nodes_list(b))
    na.kind, nb.kind = nb.kind, na.kind
    na.value, nb.value = nb.value, na.value
    na.children, nb.children = nb.children, na.children
    if a.depth() > max_depth:
        return tree_a.copy()
    return a


def mutate(tree, max_depth=6, p=0.15, terminal_set=FEATURE_COLUMNS):
    t = tree.copy()
    for n in _nodes_list(t):
        if random.random() < p:
            r = random_tree(2, terminal_set=terminal_set)
            n.kind, n.value, n.children = r.kind, r.value, r.children
    if t.depth() > max_depth:
        return tree.copy()
    return t


def _tournament_select(scored, k):
    contestants = random.sample(scored, min(k, len(scored)))
    contestants.sort(key=lambda t: t[0], reverse=True)
    return contestants[0][1]


# ------------------------------------------------------------- Strategy wrap

class GPStrategy(Strategy):
    """Envolve uma arvore evoluida na mesma interface Strategy usada pelas
    familias fixas, entao ela pode ser comparada e usada ao vivo do mesmo
    jeito (predict_last, integracao com hedge, etc)."""
    name = "genetic_program"

    def __init__(self, tree, threshold=0.3, name=None):
        self.tree = tree
        if name is not None:
            self.name = name
        self.params = {"threshold": threshold, "expr": repr(tree), "depth": tree.depth(), "size": tree.size()}

    def signals(self, df):
        raw = evaluate(self.tree, df)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.tanh(raw / 3.0)
        thr = self.params["threshold"]
        action = np.where(raw > thr, "call", np.where(raw < -thr, "put", None))
        span = max(1e-6, 1 - thr)
        confidence = np.clip((np.abs(raw) - thr) / span, 0, 1)
        return make_signals_df(action, confidence, df.index)


# ------------------------------------------------------------------ search

def run_gp(train_df, test_df, population_size=40, generations=10, max_depth=5,
           tournament_size=4, payout=0.85, amount=1.0, min_confidence=0.15,
           elite_size=5, threshold=0.3, on_individual=None,
           terminal_set=FEATURE_COLUMNS, strategy_name=None, on_generation=None):
    """'terminal_set' controla que colunas de feature a arvore pode usar como
    folha - o padrao (FEATURE_COLUMNS) inclui indicadores classicos prontos
    (RSI/MACD/Bollinger/SMA); passar RAW_FEATURE_COLUMNS (so price-action)
    forca a arvore a descobrir padroes sem herdar essas formulas.

    'on_generation(gen_index, generations)', se passado, e chamado ao final
    de CADA geracao (nao so no final da busca inteira) - sem isso, um
    processo que so atualiza status uma vez por onda inteira (pode levar
    dezenas de segundos) parecia "parado" pro painel no meio do caminho,
    mesmo trabalhando normalmente."""
    from .features import build_features
    train_feat = build_features(train_df)
    test_feat = build_features(test_df)

    population = [random_tree(max_depth, terminal_set=terminal_set) for _ in range(population_size)]
    best = None
    history = []

    for gen in range(generations):
        scored = []
        for tree in population:
            strat = GPStrategy(tree, threshold=threshold, name=strategy_name)
            try:
                result = backtest(strat, train_df, payout=payout, horizon=1,
                                   min_confidence=min_confidence, amount=amount,
                                   features_df=train_feat)
            except Exception:
                continue
            scored.append((result.fitness, tree, result))
            if on_individual is not None:
                on_individual(strat, result)

        if not scored:
            population = [random_tree(max_depth, terminal_set=terminal_set) for _ in range(population_size)]
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        history.append(scored[0][0])

        if best is None or scored[0][0] > best[0]:
            best = scored[0]

        elite = [t for _, t, _ in scored[:elite_size]]
        next_population = [t.copy() for t in elite]

        while len(next_population) < population_size:
            if random.random() < 0.75:
                p1 = _tournament_select(scored, tournament_size)
                p2 = _tournament_select(scored, tournament_size)
                child = crossover(p1, p2, max_depth=max_depth)
            else:
                p1 = _tournament_select(scored, tournament_size)
                child = mutate(p1, max_depth=max_depth, terminal_set=terminal_set)
            next_population.append(child)

        population = next_population[:population_size]

        if on_generation is not None:
            on_generation(gen, generations)

    _, best_tree, _ = best
    best_strategy = GPStrategy(best_tree, threshold=threshold, name=strategy_name)
    # avalia no conjunto de teste (fora da amostra usada pra evoluir) -
    # e o numero que de fato importa, nao o fitness de treino
    test_result = backtest(best_strategy, test_df, payout=payout, horizon=1,
                            min_confidence=min_confidence, amount=amount,
                            features_df=test_feat)
    return best_strategy, test_result, history


# ------------------------------------------------------- refinamento (Aprendiz)

def refine_from_seeds(seed_trees, train_df, test_df, population_size=30, generations=15,
                       max_depth=6, tournament_size=4, payout=0.85, amount=1.0,
                       min_confidence=0.15, elite_size=4, threshold=0.3,
                       on_individual=None, terminal_set=FEATURE_COLUMNS,
                       strategy_name=None, on_generation=None):
    """Busca evolutiva FOCADA - usada pelo Aprendiz (learner.py), um processo
    SEPARADO do Gerador (autonomous.py/run_gp, que fica 100% intocado por
    esta funcao). A diferenca central pra run_gp: a populacao aqui nasce e
    permanece inteira derivada de 'seed_trees' (estrategias JA CONFIRMADAS
    no holdout em sessoes anteriores - ver registry.get_hall_of_fame) e
    NUNCA recebe sangue aleatorio novo. O objetivo nao e explorar o espaco
    todo de novo (isso continua sendo so trabalho do Gerador) - e refinar o
    que ja provou ter edge real, convergindo pra uma unica estrategia
    polida."""
    if not seed_trees:
        raise ValueError("refine_from_seeds precisa de pelo menos uma arvore semente")

    from .features import build_features
    train_feat = build_features(train_df)
    test_feat = build_features(test_df)

    def _seed_population():
        pop = []
        i = 0
        while len(pop) < population_size:
            base = seed_trees[i % len(seed_trees)].copy()
            round_idx = i // len(seed_trees)
            # a primeira rodada de cada semente entra intocada (preserva o
            # que ja funciona); rodadas seguintes mutam com intensidade
            # crescente - da variedade sem nunca sair da "familia" delas.
            pop.append(base if round_idx == 0 else
                        mutate(base, max_depth=max_depth, p=min(0.4, 0.1 * round_idx),
                               terminal_set=terminal_set))
            i += 1
        return pop[:population_size]

    population = _seed_population()
    best = None
    history = []

    for gen in range(generations):
        scored = []
        for tree in population:
            strat = GPStrategy(tree, threshold=threshold, name=strategy_name)
            try:
                result = backtest(strat, train_df, payout=payout, horizon=1,
                                   min_confidence=min_confidence, amount=amount,
                                   features_df=train_feat)
            except Exception:
                continue
            scored.append((result.fitness, tree, result))
            if on_individual is not None:
                on_individual(strat, result)

        if not scored:
            population = _seed_population()
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        history.append(scored[0][0])

        if best is None or scored[0][0] > best[0]:
            best = scored[0]

        elite = [t for _, t, _ in scored[:elite_size]]
        next_population = [t.copy() for t in elite]

        while len(next_population) < population_size:
            if random.random() < 0.75:
                p1 = _tournament_select(scored, tournament_size)
                p2 = _tournament_select(scored, tournament_size)
                child = crossover(p1, p2, max_depth=max_depth)
            else:
                p1 = _tournament_select(scored, tournament_size)
                child = mutate(p1, max_depth=max_depth, terminal_set=terminal_set)
            next_population.append(child)

        population = next_population[:population_size]

        if on_generation is not None:
            on_generation(gen, generations)

    _, best_tree, _ = best
    best_strategy = GPStrategy(best_tree, threshold=threshold, name=strategy_name)
    test_result = backtest(best_strategy, test_df, payout=payout, horizon=1,
                            min_confidence=min_confidence, amount=amount,
                            features_df=test_feat)
    return best_strategy, test_result, history
