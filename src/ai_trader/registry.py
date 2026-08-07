"""Registro persistente de TODA estrategia ja avaliada, em SQLite (um unico
arquivo, sem servico externo). E o 'feedback das estrategias ja geradas e o
nivel de sucesso' pedido: cada tentativa (familia fixa ou arvore de
programacao genetica) fica gravada, entao a busca acumula conhecimento entre
sessoes em vez de esquecer tudo a cada execucao, e da pra consultar depois
"qual foi a melhor estrategia ja encontrada, em qualquer sessao".

Cada linha marca em qual pedaco dos dados foi avaliada ('train' ou 'test').
Isso importa: durante a evolucao da programacao genetica, cada individuo e
julgado no pedaco de TREINO (pra dar sinal de selecao) - esse numero e
otimista e nao deve ser usado pra decidir nada, ou o "melhor de todos" vira
so a arvore que mais decorou ruido do treino. Only linhas 'test' (fora da
amostra usada pra evoluir) entram no ranking e na escolha final.
"""
import json
import pickle
import shutil
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "strategies.db"
BACKUP_DIR = Path(__file__).parent / "backups"


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    # WAL deixa leitores (webapp consultando status/campeao) e o escritor
    # (gerador gravando centenas de tentativas por geracao) nao se
    # bloquearem um ao outro - sem isso, cada INSERT do gerador podia ficar
    # esperando o modo de journal padrao (rollback) liberar, e isso sozinho
    # explicava boa parte da lentidao observada com o painel web aberto.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            params TEXT NOT NULL,
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            split TEXT NOT NULL,
            num_trades INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            win_pct REAL NOT NULL,
            win_rate REAL NOT NULL,
            breakeven_win_rate REAL NOT NULL,
            edge REAL NOT NULL,
            expectancy REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            fitness REAL NOT NULL,
            ruin_probability REAL,
            target_probability REAL
        )
    """)
    _migrate_add_win_loss_columns(conn)
    _migrate_add_strategy_blob_column(conn)
    _migrate_add_candles_used_column(conn)
    _migrate_add_trade_returns_column(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_fitness ON trials(fitness)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_session ON trials(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_split ON trials(split)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS champions (
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            confirmed INTEGER NOT NULL,
            edge REAL NOT NULL,
            fitness REAL NOT NULL,
            promoted_at REAL NOT NULL,
            PRIMARY KEY (active, duration_minutes)
        )
    """)
    _migrate_add_pinned_column(conn)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            component TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT NOT NULL,
            traceback TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_created ON errors(created_at)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS champion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            confirmed INTEGER NOT NULL,
            edge REAL NOT NULL,
            fitness REAL NOT NULL,
            promoted_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_champ_hist_time ON champion_history(promoted_at)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            account_mode TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            action TEXT NOT NULL,
            amount REAL NOT NULL,
            confidence REAL,
            pnl REAL NOT NULL,
            balance_after REAL NOT NULL,
            is_hedge INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(created_at)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS learner_best (
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (active, duration_minutes)
        )
    """)
    return conn


def _migrate_add_win_loss_columns(conn):
    """Bancos criados antes das colunas wins/losses/win_pct existirem: adiciona
    as colunas e preenche as linhas ja existentes a partir de num_trades/win_rate."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    if "wins" not in existing:
        conn.execute("ALTER TABLE trials ADD COLUMN wins INTEGER")
        conn.execute("ALTER TABLE trials ADD COLUMN losses INTEGER")
        conn.execute("ALTER TABLE trials ADD COLUMN win_pct REAL")
        conn.execute(
            "UPDATE trials SET "
            "wins = CAST(ROUND(num_trades * win_rate) AS INTEGER), "
            "losses = num_trades - CAST(ROUND(num_trades * win_rate) AS INTEGER), "
            "win_pct = win_rate * 100.0 "
            "WHERE wins IS NULL")
        conn.commit()


def record_trial(session_id, kind, name, params, active, duration_minutes, result,
                  split, ruin_probability=None, target_probability=None, strategy=None,
                  candles_used=None):
    """'strategy', se passado, e serializado (pickle) e guardado junto - so
    faz sentido pra linhas 'holdout' (a unica avaliacao confiavel de verdade),
    pra permitir reusar a MESMA estrategia depois via load_strategy() sem
    precisar re-buscar do zero.

    'candles_used' e quantos candles historicos embasaram essa avaliacao -
    pedido explicitamente pra aparecer junto de testes/vitorias/derrotas na
    tela do campeao e no ranking.

    Junto com 'strategy', tambem guarda a sequencia de retornos POR UNIDADE
    apostada (result.trade_pnls / result.amount - ou seja +payout numa
    vitoria, -1 numa derrota, na ordem real em que aconteceram no backtest).
    Guardar em unidade (nao em valor absoluto) deixa reescalar pra qualquer
    saldo/aposta configurados depois, sem precisar rodar o backtest de novo -
    e o que alimenta o "saldo final simulado" por estrategia na tela."""
    if split not in ("train", "validation", "test", "holdout"):
        raise ValueError("split precisa ser 'train', 'validation', 'test' ou 'holdout'")
    wins = round(result.num_trades * result.win_rate)
    losses = result.num_trades - wins
    win_pct = result.win_rate * 100.0
    blob = pickle.dumps(strategy) if strategy is not None else None
    trade_pnls = getattr(result, "trade_pnls", None) if strategy is not None else None
    amount = getattr(result, "amount", None) or 1.0
    returns_json = (
        json.dumps([round(float(p) / amount, 6) for p in trade_pnls])
        if trade_pnls is not None and len(trade_pnls) > 0 else None)
    conn = _connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO trials (session_id, created_at, kind, name, params, active, "
            "duration_minutes, split, num_trades, wins, losses, win_pct, win_rate, "
            "breakeven_win_rate, edge, expectancy, max_drawdown, fitness, "
            "ruin_probability, target_probability, strategy_blob, candles_used, "
            "trade_returns_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, time.time(), kind, name, json.dumps(params, default=str),
             active, duration_minutes, split, result.num_trades, wins, losses, win_pct,
             result.win_rate, result.breakeven_win_rate, result.edge, result.expectancy,
             result.max_drawdown, result.fitness, ruin_probability, target_probability, blob,
             candles_used, returns_json))
        trial_id = cur.lastrowid
    conn.close()
    return trial_id


def _migrate_add_strategy_blob_column(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    if "strategy_blob" not in existing:
        conn.execute("ALTER TABLE trials ADD COLUMN strategy_blob BLOB")
        conn.commit()


def _migrate_add_candles_used_column(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    if "candles_used" not in existing:
        conn.execute("ALTER TABLE trials ADD COLUMN candles_used INTEGER")
        conn.commit()


def _migrate_add_trade_returns_column(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    if "trade_returns_json" not in existing:
        conn.execute("ALTER TABLE trials ADD COLUMN trade_returns_json TEXT")
        conn.commit()


def _migrate_add_pinned_column(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(champions)")}
    if "pinned" not in existing:
        conn.execute("ALTER TABLE champions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def replay_final_balance(trial_id, starting_balance, stake):
    """Reconstitui o saldo final de uma estrategia partindo de
    `starting_balance`, apostando `stake` fixo em cada operacao, na ORDEM
    real observada no backtest daquele trial - nao e uma estimativa
    probabilistica, e o resultado literal que a estrategia teria dado se
    tivesse operado essas mesmas 25 mil candles com esse saldo/aposta.
    Retorna None se esse trial nao tem a sequencia de retornos guardada
    (so trials 'holdout', que sao os unicos com strategy_blob, guardam)."""
    conn = _connect()
    row = conn.execute(
        "SELECT trade_returns_json FROM trials WHERE id=?", (trial_id,)).fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    returns = json.loads(row[0])
    total_pnl = sum(r * stake for r in returns)
    return starting_balance + total_pnl


def pin_champion(active, duration_minutes, trial_id):
    """Forca um trial especifico a virar campeao AGORA (usado quando o
    usuario escolhe manualmente 'ir a live com esta' numa estrategia da
    tabela), e marca como pinado - enquanto pinado, promote_if_better() nao
    troca automaticamente por uma estrategia melhor achada pelo gerador. So
    aceita trials 'holdout' (a unica avaliacao confiavel pra operar de
    verdade). 'confirmed' e recalculado aqui com o MESMO criterio de
    autonomous.py._passes_criteria - trials nao guarda essa coluna direto,
    e o gate de dinheiro real do executor.py depende dela."""
    from . import config
    conn = _connect()
    row = conn.execute(
        "SELECT name, edge, fitness, num_trades, ruin_probability FROM trials "
        "WHERE id=? AND split='holdout'", (trial_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError("trial nao encontrado ou nao e uma avaliacao holdout")
    name, edge, fitness, num_trades, ruin_probability = row
    confirmed = (
        num_trades >= 30 and edge >= config.MIN_EDGE_TO_TRADE and
        ruin_probability is not None and ruin_probability <= config.MAX_RUIN_PROBABILITY)
    promoted_at = time.time()
    with conn:
        conn.execute(
            "INSERT INTO champions (active, duration_minutes, trial_id, name, "
            "confirmed, edge, fitness, promoted_at, pinned) VALUES (?,?,?,?,?,?,?,?,1) "
            "ON CONFLICT(active, duration_minutes) DO UPDATE SET "
            "trial_id=excluded.trial_id, name=excluded.name, confirmed=excluded.confirmed, "
            "edge=excluded.edge, fitness=excluded.fitness, promoted_at=excluded.promoted_at, "
            "pinned=1",
            (active, duration_minutes, trial_id, name, int(confirmed), edge, fitness, promoted_at))
        conn.execute(
            "INSERT INTO champion_history (active, duration_minutes, trial_id, name, "
            "confirmed, edge, fitness, promoted_at) VALUES (?,?,?,?,?,?,?,?)",
            (active, duration_minutes, trial_id, name, int(confirmed), edge, fitness, promoted_at))
    conn.close()


def unpin_champion(active, duration_minutes):
    """Volta o campeao desse ativo pro modo automatico - o gerador passa a
    poder promover uma estrategia melhor de novo assim que achar uma."""
    conn = _connect()
    with conn:
        conn.execute(
            "UPDATE champions SET pinned=0 WHERE active=? AND duration_minutes=?",
            (active, duration_minutes))
    conn.close()


# --------------------------------------------------- Aprendiz (learner.py)

def set_learner_best_if_better(active, duration_minutes, trial_id, fitness):
    """Guarda a escolha ATUAL do Aprendiz (uma unica estrategia refinada a
    partir do hall da fama) - so substitui a anterior se o novo fitness for
    estritamente maior (ratchet, igual promote_if_better). NUNCA mexe em
    champions/executor - e so uma sugestao visivel no dashboard ate o
    usuario decidir usa-la de verdade (via pin_champion)."""
    conn = _connect()
    current = conn.execute(
        "SELECT trial_id FROM learner_best WHERE active=? AND duration_minutes=?",
        (active, duration_minutes)).fetchone()
    promote = True
    if current is not None:
        row = conn.execute("SELECT fitness FROM trials WHERE id=?", (current[0],)).fetchone()
        if row is not None and row[0] >= fitness:
            promote = False
    if promote:
        with conn:
            conn.execute(
                "INSERT INTO learner_best (active, duration_minutes, trial_id, updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(active, duration_minutes) DO UPDATE SET "
                "trial_id=excluded.trial_id, updated_at=excluded.updated_at",
                (active, duration_minutes, trial_id, time.time()))
    conn.close()
    return promote


def get_learner_best(active, duration_minutes):
    conn = _connect()
    row = conn.execute(
        "SELECT l.trial_id, l.updated_at, t.name, t.edge, t.fitness, t.num_trades, "
        "t.wins, t.losses, t.win_pct, t.candles_used "
        "FROM learner_best l JOIN trials t ON t.id = l.trial_id "
        "WHERE l.active=? AND l.duration_minutes=?", (active, duration_minutes)).fetchone()
    conn.close()
    if row is None:
        return None
    (trial_id, updated_at, name, edge, fitness, num_trades, wins, losses,
     win_pct, candles_used) = row
    return {
        "trial_id": trial_id, "updated_at": updated_at, "name": name, "edge": edge,
        "fitness": fitness, "num_trades": num_trades, "wins": wins, "losses": losses,
        "win_pct": win_pct, "candles_used": candles_used,
    }


def get_hall_of_fame(active, duration_minutes, limit=6):
    """Arvores de programacao genetica que ja foram CONFIRMADAS no holdout
    alguma vez (olha o HISTORICO de campeoes inteiro, nao so o atual) -
    usadas como semente pro Aprendiz refinar (ver learner.py). Inclui tanto
    as do Gerador ('genetic_program'/'genetic_program_raw') quanto as que o
    proprio Aprendiz ja refinou e confirmou antes ('learner_refined'/
    'learner_refined_raw') - sem isso, o Aprendiz nunca conseguia construir
    em cima do proprio progresso, sempre voltando a partir do ultimo achado
    do Gerador em vez de continuar de onde parou. Familias fixas (evolve.py)
    nao entram aqui - nao tem arvore de verdade pra reaproveitar. So retorna
    metadados (leves) - quem quiser a arvore em si chama load_strategy(trial_id)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT DISTINCT t.id, t.name, t.edge, ch.promoted_at, t.num_trades, "
        "t.wins, t.losses, t.win_pct, t.candles_used "
        "FROM champion_history ch JOIN trials t ON t.id = ch.trial_id "
        "WHERE ch.active=? AND ch.duration_minutes=? AND ch.confirmed=1 "
        "AND (t.name LIKE 'genetic_program%' OR t.name LIKE 'learner_refined%') "
        "AND t.strategy_blob IS NOT NULL "
        "ORDER BY ch.promoted_at DESC LIMIT ?",
        (active, duration_minutes, limit)).fetchall()
    conn.close()
    result = []
    for r in rows:
        (trial_id, name, edge, promoted_at, num_trades, wins, losses,
         win_pct, candles_used) = r
        result.append({
            "trial_id": trial_id, "name": name, "edge": edge, "promoted_at": promoted_at,
            "num_trades": num_trades, "wins": wins, "losses": losses,
            "win_pct": win_pct, "candles_used": candles_used,
        })
    return result


def load_strategy(trial_id):
    conn = _connect()
    row = conn.execute("SELECT strategy_blob FROM trials WHERE id=?", (trial_id,)).fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    return pickle.loads(row[0])


def find_confirmed_strategy(active, duration_minutes, min_edge, max_ruin_probability,
                            max_age_seconds=None):
    """Procura, entre as confirmacoes 'holdout' ja feitas, a melhor que ainda
    passa nos criterios de seguranca atuais (edge minimo, risco de ruina
    maximo) - pra reaproveitar uma estrategia ja validada em vez de gastar
    tempo procurando de novo do zero toda vez que o robo inicia."""
    conn = _connect()
    query = (
        "SELECT id, name, params, edge, fitness, num_trades, ruin_probability, created_at "
        "FROM trials WHERE split='holdout' AND active=? AND duration_minutes=? "
        "AND edge >= ? AND strategy_blob IS NOT NULL"
    )
    params = [active, duration_minutes, min_edge]
    if max_ruin_probability is not None:
        query += " AND (ruin_probability IS NULL OR ruin_probability <= ?)"
        params.append(max_ruin_probability)
    if max_age_seconds is not None:
        query += " AND created_at >= ?"
        params.append(time.time() - max_age_seconds)
    query += " ORDER BY fitness DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    conn.close()
    if row is None:
        return None
    trial_id, name, params_json, edge, fitness, num_trades, ruin_probability, created_at = row
    strategy = load_strategy(trial_id)
    if strategy is None:
        return None
    return {
        "id": trial_id, "name": name, "edge": edge, "fitness": fitness,
        "num_trades": num_trades, "ruin_probability": ruin_probability,
        "created_at": created_at, "strategy": strategy,
    }


def find_recent_holdout_by_name(active, duration_minutes, name, params, max_age_seconds):
    """Procura uma confirmacao holdout ja gravada pra essa MESMA estrategia
    (nome + parametros identicos) dentro da janela de tempo recente. Um
    backtest e determinístico: mesma estrategia + mesmos parametros +
    mesmos candles = sempre o mesmo resultado - entao se a busca autonoma
    seleciona repetidas vezes o mesmo candidato como "melhor de validation"
    onda apos onda (comum quando nada supera o campeao atual ainda), reavaliar
    e regravar no holdout a cada vez so duplica linha identica no banco sem
    nenhuma informacao nova. Compara os parametros pelo dict (nao pela string
    JSON) pra nao depender da ordem das chaves."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, params, num_trades, wins, losses, win_pct, win_rate, edge, "
        "expectancy, max_drawdown, fitness, ruin_probability, target_probability "
        "FROM trials WHERE active=? AND duration_minutes=? AND split='holdout' "
        "AND name=? AND created_at >= ? ORDER BY created_at DESC LIMIT 20",
        (active, duration_minutes, name, time.time() - max_age_seconds)).fetchall()
    conn.close()
    cols = ["id", "params", "num_trades", "wins", "losses", "win_pct", "win_rate", "edge",
            "expectancy", "max_drawdown", "fitness", "ruin_probability", "target_probability"]
    for row in rows:
        d = dict(zip(cols, row))
        if json.loads(d["params"]) == params:
            return d
    return None


def count_trials(session_id=None, split=None):
    conn = _connect()
    query = "SELECT COUNT(*) FROM trials WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id=?"
        params.append(session_id)
    if split:
        query += " AND split=?"
        params.append(split)
    n = conn.execute(query, params).fetchone()[0]
    conn.close()
    return n


def best_overall(active=None, min_trades=30, limit=10, split="test", exclude_learner=True):
    """Por padrao so considera avaliacoes 'test' (fora da amostra usada pra
    evoluir/treinar) - e o unico numero em que da pra confiar pra ranquear.

    'exclude_learner' (default True) tira as estrategias refinadas pelo
    Aprendiz (learner_refined/learner_refined_raw) do ranking - pedido
    explicito: como o Aprendiz parte do que o Gerador ja confirmou, ele
    tende a sempre ficar por cima, escondendo se o Gerador esta encontrando
    coisa boa por conta propria. Essa funcao representa "a competicao do
    Gerador"; a escolha do Aprendiz tem seu proprio lugar na tela
    (get_learner_best / painel "Escolha do Aprendiz").

    Deduplica por (nome, parametros): a mesma estrategia com os mesmos
    parametros pode ter sido confirmada varias vezes em ondas diferentes
    (o resultado e identico, backtest e deterministico) - so a linha de
    maior fitness de cada combinacao unica entra no ranking, senao a mesma
    estrategia aparece repetida dezenas de vezes."""
    conn = _connect()
    # busca mais linhas que o limite final, pra sobrar o suficiente depois
    # de deduplicar (na pratica poucas estrategias unicas dominam o topo)
    query = ("SELECT id, session_id, created_at, kind, name, params, edge, fitness, "
              "num_trades, wins, losses, win_pct, win_rate, ruin_probability, "
              "target_probability, candles_used "
              "FROM trials WHERE num_trades >= ? AND split = ?")
    params = [min_trades, split]
    if active:
        query += " AND active = ?"
        params.append(active)
    if exclude_learner:
        query += " AND name NOT LIKE 'learner_refined%'"
    query += " ORDER BY fitness DESC LIMIT ?"
    params.append(max(limit * 20, 200))
    rows = conn.execute(query, params).fetchall()
    conn.close()
    cols = ["id", "session_id", "created_at", "kind", "name", "params", "edge",
            "fitness", "num_trades", "wins", "losses", "win_pct", "win_rate",
            "ruin_probability", "target_probability", "candles_used"]

    seen = set()
    deduped = []
    for row in rows:
        d = dict(zip(cols, row))
        # compara pelos parametros parseados (nao a string JSON crua) pra
        # nao depender da ordem das chaves na hora de gravar
        key = (d["name"], json.dumps(json.loads(d["params"]), sort_keys=True))
        # ja vem ordenado por fitness DESC -> o primeiro visto e o melhor
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
        if len(deduped) >= limit:
            break
    return deduped


def promote_if_better(active, duration_minutes, trial_id, name, confirmed, edge, fitness):
    """Guarda o 'campeao atual' pra esse ativo/duracao - o executor consulta
    isso pra saber qual estrategia usar, sem precisar ele mesmo escolher
    entre um monte de linhas de trials. Regra de prioridade: uma estrategia
    CONFIRMADA no holdout sempre vence uma nao-confirmada, nao importa o
    fitness; entre duas do mesmo status, o maior fitness vence.

    Se o campeao atual foi PINADO manualmente (usuario escolheu 'ir a live
    com esta' na tela), a promocao automatica fica suspensa ate o usuario
    voltar pro modo automatico - senao o gerador substituiria a escolha
    manual assim que achasse algo com fitness maior."""
    conn = _connect()
    current = conn.execute(
        "SELECT confirmed, fitness, pinned FROM champions WHERE active=? AND duration_minutes=?",
        (active, duration_minutes)).fetchone()

    promote = False
    if current is None:
        promote = True
    else:
        cur_confirmed, cur_fitness, cur_pinned = current
        if cur_pinned:
            promote = False
        elif confirmed and not cur_confirmed:
            promote = True
        elif confirmed == bool(cur_confirmed) and fitness > cur_fitness:
            promote = True

    if promote:
        promoted_at = time.time()
        with conn:
            conn.execute(
                "INSERT INTO champions (active, duration_minutes, trial_id, name, "
                "confirmed, edge, fitness, promoted_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(active, duration_minutes) DO UPDATE SET "
                "trial_id=excluded.trial_id, name=excluded.name, confirmed=excluded.confirmed, "
                "edge=excluded.edge, fitness=excluded.fitness, promoted_at=excluded.promoted_at",
                (active, duration_minutes, trial_id, name, int(confirmed), edge, fitness, promoted_at))
            conn.execute(
                "INSERT INTO champion_history (active, duration_minutes, trial_id, name, "
                "confirmed, edge, fitness, promoted_at) VALUES (?,?,?,?,?,?,?,?)",
                (active, duration_minutes, trial_id, name, int(confirmed), edge, fitness, promoted_at))
    conn.close()
    return promote


def get_champion_history(active=None, limit=100):
    conn = _connect()
    query = ("SELECT promoted_at, name, confirmed, edge, fitness FROM champion_history "
              "WHERE 1=1")
    params = []
    if active:
        query += " AND active=?"
        params.append(active)
    query += " ORDER BY promoted_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    cols = ["promoted_at", "name", "confirmed", "edge", "fitness"]
    return [dict(zip(cols, row)) for row in rows]


def record_trade(active, duration_minutes, account_mode, strategy_name, action, amount,
                 confidence, pnl, balance_after, is_hedge=False):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO trades (created_at, active, duration_minutes, account_mode, "
            "strategy_name, action, amount, confidence, pnl, balance_after, is_hedge) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), active, duration_minutes, account_mode, strategy_name, action,
             amount, confidence, pnl, balance_after, int(is_hedge)))
    conn.close()


def get_trade_history(active=None, limit=200):
    conn = _connect()
    query = ("SELECT created_at, strategy_name, action, amount, pnl, balance_after, is_hedge "
              "FROM trades WHERE 1=1")
    params = []
    if active:
        query += " AND active=?"
        params.append(active)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    cols = ["created_at", "strategy_name", "action", "amount", "pnl", "balance_after", "is_hedge"]
    return [dict(zip(cols, row)) for row in rows]


def get_trade_stats(active=None, since_seconds=None):
    conn = _connect()
    query = "SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM trades WHERE is_hedge=0"
    params = []
    if active:
        query += " AND active=?"
        params.append(active)
    if since_seconds:
        query += " AND created_at >= ?"
        params.append(time.time() - since_seconds)
    row = conn.execute(query, params).fetchone()
    conn.close()
    total, pnl_sum, wins = row
    total = total or 0
    return {
        "total_trades": total, "total_pnl": pnl_sum or 0.0, "wins": wins or 0,
        "losses": total - (wins or 0), "win_rate": (wins / total) if total else 0.0,
    }


def get_champion(active, duration_minutes):
    conn = _connect()
    row = conn.execute(
        "SELECT c.trial_id, c.name, c.confirmed, c.edge, c.fitness, c.promoted_at, c.pinned, "
        "t.num_trades, t.wins, t.losses, t.candles_used "
        "FROM champions c LEFT JOIN trials t ON t.id = c.trial_id "
        "WHERE c.active=? AND c.duration_minutes=?", (active, duration_minutes)).fetchone()
    conn.close()
    if row is None:
        return None
    (trial_id, name, confirmed, edge, fitness, promoted_at, pinned,
     num_trades, wins, losses, candles_used) = row
    strategy = load_strategy(trial_id)
    if strategy is None:
        return None
    return {
        "id": trial_id, "name": name, "confirmed": bool(confirmed),
        "edge": edge, "fitness": fitness, "promoted_at": promoted_at,
        "pinned": bool(pinned),
        "num_trades": num_trades, "wins": wins, "losses": losses,
        "candles_used": candles_used, "strategy": strategy,
    }


def log_error(component, error_type, message, traceback_text=None):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO errors (created_at, component, error_type, message, traceback) "
            "VALUES (?,?,?,?,?)", (time.time(), component, error_type, message, traceback_text))
    conn.close()


def get_recent_errors(limit=50, component=None):
    conn = _connect()
    query = "SELECT created_at, component, error_type, message, traceback FROM errors"
    params = []
    if component:
        query += " WHERE component=?"
        params.append(component)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    cols = ["created_at", "component", "error_type", "message", "traceback"]
    return [dict(zip(cols, row)) for row in rows]


def get_trials_per_hour(active=None, hours=24):
    """Conta tentativas por hora, pras ultimas N horas - pra grafico de
    produtividade do gerador ao longo do tempo."""
    conn = _connect()
    since = time.time() - hours * 3600
    query = (
        "SELECT CAST(created_at / 3600 AS INTEGER) AS hour_bucket, COUNT(*) "
        "FROM trials WHERE created_at >= ?"
    )
    params = [since]
    if active:
        query += " AND active=?"
        params.append(active)
    query += " GROUP BY hour_bucket ORDER BY hour_bucket"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [{"hour_ts": bucket * 3600, "count": count} for bucket, count in rows]


def format_leaderboard(rows):
    lines = ["=== Melhores estrategias de todos os tempos (fora da amostra de treino) ==="]
    lines.append(
        f"{'id':<6}{'kind':<10}{'name':<22}{'trades':>8}{'vitorias':>9}{'derrotas':>9}"
        f"{'acerto%':>9}{'edge':>9}{'fitness':>10}{'ruina%':>9}")
    for r in rows:
        ruin = f"{r['ruin_probability']*100:.1f}" if r["ruin_probability"] is not None else "-"
        lines.append(
            f"{r['id']:<6}{r['kind']:<10}{r['name']:<22}{r['num_trades']:>8}"
            f"{r['wins']:>9}{r['losses']:>9}{r['win_pct']:>9.1f}"
            f"{r['edge']:>9.4f}{r['fitness']:>10.5f}{ruin:>9}")
    return "\n".join(lines)


# ---------------------------------------------------------- backup / reset

def create_backup(label="manual"):
    """Copia o arquivo inteiro do banco (trials/campeoes/trades/erros) - NAO
    inclui candles.db, que e um arquivo separado e nunca e tocado por isto."""
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"strategies_{ts}_{label}.db"
    shutil.copy2(DB_PATH, dest)
    return dest.name


def list_backups():
    if not BACKUP_DIR.exists():
        return []
    files = sorted(BACKUP_DIR.glob("strategies_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1),
             "created_at": f.stat().st_mtime} for f in files]


def restore_backup(name):
    """Restaura um backup escolhido. Faz um backup do estado ATUAL antes de
    sobrescrever, entao restaurar tambem e reversivel."""
    safe_name = Path(name).name  # nunca deixa sair da pasta de backups
    src = BACKUP_DIR / safe_name
    if not src.is_file():
        raise FileNotFoundError(safe_name)
    create_backup(label="antes_de_restaurar")
    shutil.copy2(src, DB_PATH)


def delete_backup(name):
    """Apaga um backup especifico pelo nome do arquivo."""
    safe_name = Path(name).name  # nunca deixa sair da pasta de backups
    target = BACKUP_DIR / safe_name
    if not target.is_file():
        raise FileNotFoundError(safe_name)
    target.unlink()


def clear_old_backups(keep=2):
    """Apaga os backups mais antigos, mantendo so os `keep` mais recentes
    (sempre fica pelo menos algum pra restaurar em caso de necessidade)."""
    backups = list_backups()  # ja vem ordenado do mais novo pro mais velho
    to_delete = backups[keep:]
    for b in to_delete:
        (BACKUP_DIR / b["name"]).unlink(missing_ok=True)
    return [b["name"] for b in to_delete]


def clear_errors():
    """Apaga todo o log de erros registrado."""
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM errors")
    conn.execute("VACUUM")
    conn.close()


def reset_strategy_data():
    """So apaga tentativas/campeao/historico de campeao/escolha do aprendiz -
    as estrategias sao dificeis de achar de novo, entao isto fica separado
    do reset de saldo de proposito (pedido explicito: 'as vezes eu so quero
    resetar o saldo e nao as estrategias'). learner_best entra aqui porque
    aponta pra um trial que esta sendo apagado - deixa-lo pra tras viraria
    referencia orfa. NAO mexe em trades/errors/candles."""
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM trials")
        conn.execute("DELETE FROM champions")
        conn.execute("DELETE FROM champion_history")
        conn.execute("DELETE FROM learner_best")
    conn.execute("VACUUM")
    conn.close()


def reset_balance_data():
    """So apaga o historico de operacoes (trades) - o saldo em si e resetado
    na IQ Option por Broker.reset_practice_balance(), chamado por quem
    orquestra isto (webapp.py), nao aqui. NAO mexe em trials/campeao/errors."""
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM trades")
    conn.execute("VACUUM")
    conn.close()
