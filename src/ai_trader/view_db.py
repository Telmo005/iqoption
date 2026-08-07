"""Visualizador rapido da base de estrategias, sem precisar de ferramenta
externa.

Uso:
    cd iqoptionapi/src
    python -m ai_trader.view_db
    python -m ai_trader.view_db --limit 30 --min-trades 20
    python -m ai_trader.view_db --all-sessions
"""
import argparse
import sqlite3

from . import registry


def print_sessions_summary():
    conn = sqlite3.connect(registry.DB_PATH)
    rows = conn.execute(
        "SELECT session_id, MIN(created_at), MAX(created_at), COUNT(*), "
        "SUM(CASE WHEN split='test' THEN 1 ELSE 0 END) "
        "FROM trials GROUP BY session_id ORDER BY MIN(created_at) DESC"
    ).fetchall()
    conn.close()
    print("=== Sessoes ja rodadas ===")
    print(f"{'session_id':<16}{'quando':<22}{'tentativas':>12}{'validas(test)':>15}")
    import datetime
    for session_id, created_min, created_max, total, test_count in rows:
        when = datetime.datetime.fromtimestamp(created_min).strftime("%Y-%m-%d %H:%M")
        print(f"{session_id:<16}{when:<22}{total:>12}{test_count:>15}")


def main():
    parser = argparse.ArgumentParser(description="Ver a base de estrategias do ai_trader")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--active", default=None, help="filtrar por ativo, ex: EURUSD-OTC")
    parser.add_argument("--all-sessions", action="store_true", help="lista as sessoes ja rodadas")
    parser.add_argument(
        "--split", default="holdout", choices=["train", "validation", "test", "holdout"],
        help="qual particao mostrar no ranking (padrao: holdout - a unica "
             "avaliada uma unica vez, a mais confiavel)")
    args = parser.parse_args()

    print(f"base de dados: {registry.DB_PATH}")
    print(f"total de tentativas registradas: {registry.count_trials()}\n")

    if args.all_sessions:
        print_sessions_summary()
        print()

    rows = registry.best_overall(
        active=args.active, min_trades=args.min_trades, limit=args.limit, split=args.split)
    if not rows:
        print("nenhuma estrategia com trades suficientes ainda.")
        return
    print(registry.format_leaderboard(rows))


if __name__ == "__main__":
    main()
