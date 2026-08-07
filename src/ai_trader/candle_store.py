"""Cache local (SQLite) de candles. Separa 'buscar dados novos' (precisa de
rede, roda de hora em hora) de 'usar dados pra pesquisar' (100% local, pode
rodar continuamente sem nunca tocar a IQ Option). E a peca que faltava pra
o gerador de estrategias rodar de verdade em paralelo, sem risco de
conflito/queda de conexao - ele nunca mais abre uma conexao propria.
"""
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "candles.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            active TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            from_ts INTEGER NOT NULL,
            open REAL NOT NULL,
            close REAL NOT NULL,
            low REAL NOT NULL,
            high REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (active, duration_minutes, from_ts)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candles_lookup "
        "ON candles(active, duration_minutes, from_ts)")
    return conn


def latest_timestamp(active, duration_minutes):
    conn = _connect()
    row = conn.execute(
        "SELECT MAX(from_ts) FROM candles WHERE active=? AND duration_minutes=?",
        (active, duration_minutes)).fetchone()
    conn.close()
    return row[0]


def count_candles(active, duration_minutes):
    conn = _connect()
    n = conn.execute(
        "SELECT COUNT(*) FROM candles WHERE active=? AND duration_minutes=?",
        (active, duration_minutes)).fetchone()[0]
    conn.close()
    return n


def save_candles(active, duration_minutes, candles_df):
    """candles_df: colunas timestamp, from, open, close, low, high, volume
    (o formato ja retornado por broker.get_candles_df)."""
    conn = _connect()
    rows = [
        (active, duration_minutes, int(r["from"]), float(r["open"]), float(r["close"]),
         float(r["low"]), float(r["high"]), float(r["volume"]))
        for _, r in candles_df.iterrows()
    ]
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candles "
            "(active, duration_minutes, from_ts, open, close, low, high, volume) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.close()
    return len(rows)


def load_candles(active, duration_minutes, limit=None):
    """Le do cache local - NUNCA toca rede. Retorna no mesmo formato que
    broker.get_candles_df (timestamp, from, open, close, low, high, volume)."""
    conn = _connect()
    if limit:
        query = (
            "SELECT * FROM (SELECT from_ts, open, close, low, high, volume FROM candles "
            "WHERE active=? AND duration_minutes=? ORDER BY from_ts DESC LIMIT ?) "
            "ORDER BY from_ts")
        rows = conn.execute(query, (active, duration_minutes, limit)).fetchall()
    else:
        query = ("SELECT from_ts, open, close, low, high, volume FROM candles "
                  "WHERE active=? AND duration_minutes=? ORDER BY from_ts")
        rows = conn.execute(query, (active, duration_minutes)).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "from", "open", "close", "low", "high", "volume"])
    df = pd.DataFrame(rows, columns=["from", "open", "close", "low", "high", "volume"])
    df["timestamp"] = pd.to_datetime(df["from"], unit="s")
    return df[["timestamp", "from", "open", "close", "low", "high", "volume"]]


def sync(broker, active, duration_minutes, initial_backfill=3000):
    """Busca so os candles NOVOS desde a ultima sincronizacao (incremental).
    Se o cache estiver vazio, faz um backfill inicial maior. Essa e a UNICA
    funcao deste modulo que toca rede."""
    last_ts = latest_timestamp(active, duration_minutes)
    if last_ts is None:
        logger.info("cache vazio para %s/%dmin, fazendo backfill inicial de %d candles...",
                    active, duration_minutes, initial_backfill)
        fresh = broker.get_candles_df(active, duration_minutes * 60, count=initial_backfill)
    else:
        gap_seconds = time.time() - last_ts
        gap_candles = max(10, int(gap_seconds / (duration_minutes * 60)) + 20)
        logger.info("sincronizando %s/%dmin: buscando ate %d candles novos desde %s...",
                    active, duration_minutes, gap_candles,
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ts)))
        fresh = broker.get_candles_df(active, duration_minutes * 60, count=gap_candles)
        fresh = fresh[fresh["from"] > last_ts]

    if len(fresh) == 0:
        logger.info("nada novo pra sincronizar")
        return 0

    saved = save_candles(active, duration_minutes, fresh)
    total = count_candles(active, duration_minutes)
    logger.info("sincronizado: %d candles novos salvos, %d no total no cache local", saved, total)
    return saved
