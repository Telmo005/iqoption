"""Painel web do AI trader. Pensado originalmente pra rede local (sem
senha - so alcanca quem esta na mesma rede). Se IQ_DASHBOARD_PASSWORD
estiver definida no .env, todas as rotas passam a exigir autenticacao HTTP
Basic (usuario "admin") - obrigatorio antes de expor a porta na internet
(ex: numa VPS).

Uso:
    cd iqoptionapi/src
    python -m ai_trader.webapp
Depois abra http://<ip-desta-maquina>:5000 em qualquer dispositivo na
mesma rede (ou na internet, se a porta estiver exposta e a senha configurada).
"""
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from . import config, registry, status

app = Flask(__name__)


@app.before_request
def _require_auth():
    """So exige senha se IQ_DASHBOARD_PASSWORD estiver configurada - vazio
    mantem o comportamento original (sem senha, uso em rede local/confiavel)."""
    if not config.DASHBOARD_PASSWORD:
        return
    auth = request.authorization
    valid = auth is not None and secrets.compare_digest(auth.password, config.DASHBOARD_PASSWORD)
    if not valid:
        return Response(
            "Autenticacao necessaria para acessar o AI Trader.", 401,
            {"WWW-Authenticate": 'Basic realm="AI Trader"'})

SRC_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent
STALE_SECONDS = 40

COMPONENT_MODULES = {
    "sync": "ai_trader.sync",
    "generator": "ai_trader.generator",
    "executor": "ai_trader.executor",
    "learner": "ai_trader.learner",
}

# ---- configuracoes editaveis pela tela (gravadas no .env) ----
# valores lidos uma vez na inicializacao de cada processo (config.py), entao
# mudar aqui exige reiniciar Executor/Gerador pra valer - avisado na tela.
ENV_PATH = Path(__file__).resolve().parent / ".env"

# pares OTC principais - so pares "-OTC" funcionam pra abrir operacao nesta
# biblioteca hoje (ver broker.py). Trocar de par zera o progresso: cache de
# candles, campeao e historico de busca sao todos por par - o novo par
# comeca do zero (sync faz backfill, gerador busca de novo).
ACTIVE_OPTIONS = [
    "EURUSD-OTC", "EURGBP-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURJPY-OTC",
    "GBPJPY-OTC", "USDCHF-OTC", "NZDUSD-OTC", "AUDCAD-OTC",
]

SETTINGS_SPEC = [
    {"key": "IQ_ACTIVE", "label": "Par de moeda", "type": "select",
     "options": ACTIVE_OPTIONS, "default": config.ACTIVE},
    {"key": "IQ_AMOUNT", "label": "Valor por operacao", "type": "number", "default": config.AMOUNT},
    {"key": "IQ_MAX_AMOUNT", "label": "Valor maximo por operacao", "type": "number",
     "default": config.MAX_AMOUNT_PER_TRADE},
    {"key": "IQ_MAX_DAILY_LOSS", "label": "Limite de perda diaria", "type": "number",
     "default": config.MAX_DAILY_LOSS},
    {"key": "IQ_GENERATOR_CANDLE_WINDOW", "label": "Janela de candles do Gerador",
     "type": "number", "default": config.GENERATOR_CANDLE_WINDOW, "show_days": True},
    {"key": "IQ_BANKROLL_START", "label": "Saldo inicial de simulacao (por estrategia)",
     "type": "number", "default": config.BANKROLL_STARTING_BALANCE},
]


def _read_env_lines():
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def _env_dict(lines):
    d = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        d[k.strip()] = v.strip()
    return d


def _pid_alive(pid):
    """Confere no Windows se o PID ainda existe de verdade (nao so se o
    heartbeat e recente). Isso e o que impede duplicar processo: um
    componente pode ficar minutos sem escrever status (ex: sync.py dormindo
    ate 1h entre sincronizacoes) sem estar parado de verdade - confiar so na
    idade do heartbeat pra decidir se pode iniciar um novo ja causou
    processos duplicados em producao."""
    if pid is None:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return str(pid) in out.stdout
    except Exception:
        return False


def _is_running(data):
    if not data or data.get("phase") == "encerrado":
        return False
    pid = data.get("pid")
    if pid is not None:
        return _pid_alive(pid)
    # ainda na janela curta entre "cliquei em iniciar" e o processo filho
    # escrever seu proprio pid - usa idade do heartbeat como fallback
    age = time.time() - data.get("updated_at", 0)
    return age < STALE_SECONDS


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.route("/")
def index():
    return render_template("dashboard.html", lan_ip=_lan_ip())


@app.route("/api/status")
def api_status():
    components = {}
    for key in COMPONENT_MODULES:
        data = status.read_status(key)
        components[key] = {"data": data, "running": _is_running(data)}

    champion = registry.get_champion(config.ACTIVE, config.DURATION_MINUTES)
    if champion is not None:
        champion = {k: v for k, v in champion.items() if k != "strategy"}
        champion["strategy_repr"] = None
        # saldo final simulado (replay real do backtest holdout, partindo do
        # saldo de referencia configurado) - propositalmente SEM ligacao com
        # o saldo real da conta PRACTICE/REAL, que so existe pra manter o
        # sistema operando de verdade.
        final_balance = registry.replay_final_balance(
            champion["id"], config.BANKROLL_STARTING_BALANCE, config.AMOUNT)
        if final_balance is not None:
            champion["projection"] = {
                "starting_balance": config.BANKROLL_STARTING_BALANCE,
                "amount": config.AMOUNT,
                "payout": config.DEFAULT_PAYOUT,
                "final_balance": final_balance,
                "total_pnl": final_balance - config.BANKROLL_STARTING_BALANCE,
                "num_trades": champion.get("num_trades"),
            }

    errors = registry.get_recent_errors(limit=20)
    trade_stats_today = registry.get_trade_stats(active=config.ACTIVE, since_seconds=86400)
    # estrategias GP ja confirmadas no holdout - material de entrada do
    # Aprendiz (learner.py, processo separado). Exposto aqui com o MESMO
    # nivel de detalhe (testes/vitorias/derrotas/saldo final) do ranking,
    # pra dar pra validar cada uma sem precisar cruzar duas telas.
    hall_of_fame = registry.get_hall_of_fame(config.ACTIVE, config.DURATION_MINUTES, limit=6)
    for entry in hall_of_fame:
        entry["sim_starting_balance"] = config.BANKROLL_STARTING_BALANCE
        entry["final_balance"] = registry.replay_final_balance(
            entry["trial_id"], config.BANKROLL_STARTING_BALANCE, config.AMOUNT)

    # escolha ATUAL do Aprendiz (uma unica estrategia refinada a partir do
    # hall da fama) - so uma sugestao, nunca promovida automaticamente pro
    # campeao real usado pelo Executor.
    learner_best = registry.get_learner_best(config.ACTIVE, config.DURATION_MINUTES)
    if learner_best is not None:
        learner_best["sim_starting_balance"] = config.BANKROLL_STARTING_BALANCE
        learner_best["final_balance"] = registry.replay_final_balance(
            learner_best["trial_id"], config.BANKROLL_STARTING_BALANCE, config.AMOUNT)

    return jsonify({
        "config": {
            "active": config.ACTIVE,
            "duration_minutes": config.DURATION_MINUTES,
            "account_mode": "REAL" if config.ARM_REAL_MONEY else "PRACTICE",
        },
        "components": components,
        "champion": champion,
        "errors": errors,
        "trade_stats_today": trade_stats_today,
        "hall_of_fame": hall_of_fame,
        "learner_best": learner_best,
        "server_time": time.time(),
    })


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    env = _env_dict(_read_env_lines())
    result = []
    for spec in SETTINGS_SPEC:
        raw = env.get(spec["key"])
        if spec["type"] == "select":
            value = raw if raw in spec["options"] else spec["default"]
        else:
            try:
                value = float(raw) if raw is not None else spec["default"]
            except ValueError:
                value = spec["default"]
        entry = {"key": spec["key"], "label": spec["label"], "type": spec["type"], "value": value}
        if spec["type"] == "select":
            entry["options"] = spec["options"]
        if spec.get("show_days"):
            entry["show_days"] = True
        result.append(entry)
    return jsonify({"settings": result, "duration_minutes": config.DURATION_MINUTES})


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    body = request.get_json(force=True, silent=True) or {}
    spec_by_key = {s["key"]: s for s in SETTINGS_SPEC}
    updates = {}
    for key, value in body.items():
        spec = spec_by_key.get(key)
        if spec is None:
            continue
        if spec["type"] == "select":
            if value not in spec["options"]:
                return jsonify({"ok": False, "error": f"valor invalido para {key}"}), 400
            updates[key] = value
        else:
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": f"valor invalido para {key}"}), 400
            if fval <= 0:
                return jsonify({"ok": False, "error": f"{key} precisa ser maior que zero"}), 400
            updates[key] = fval

    if not updates:
        return jsonify({"ok": False, "error": "nenhum valor valido enviado"}), 400

    lines = _read_env_lines()
    seen = set()
    new_lines = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        new_lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/reports")
def api_reports():
    leaderboard = registry.best_overall(
        active=config.ACTIVE, min_trades=20, limit=40, split="holdout")
    # saldo final de cada estrategia partindo do MESMO ponto configurado
    # (independente entre elas - nenhuma "continua" o saldo da anterior),
    # replay literal da sequencia real de vitorias/derrotas do backtest.
    for row in leaderboard:
        row["sim_starting_balance"] = config.BANKROLL_STARTING_BALANCE
        row["final_balance"] = registry.replay_final_balance(
            row["id"], config.BANKROLL_STARTING_BALANCE, config.AMOUNT)
    champion_history = registry.get_champion_history(active=config.ACTIVE, limit=50)
    trade_history = registry.get_trade_history(active=config.ACTIVE, limit=100)
    trials_per_hour = registry.get_trials_per_hour(active=config.ACTIVE, hours=24)
    total_trials = registry.count_trials()

    return jsonify({
        "leaderboard": leaderboard,
        "champion_history": champion_history,
        "trade_history": trade_history,
        "trials_per_hour": trials_per_hour,
        "total_trials": total_trials,
    })


_start_lock = threading.Lock()


@app.route("/api/start/<component>", methods=["POST"])
def api_start(component):
    if component not in COMPONENT_MODULES:
        return jsonify({"ok": False, "error": "componente desconhecido"}), 400

    # o lock + a escrita de status "iniciando" ANTES de criar o processo
    # fecham a janela de corrida: sem isso, 2 cliques rapidos (ou o usuario
    # clicando de novo por achar que nao aconteceu nada) passavam os dois
    # pelo check "ja esta rodando" antes de qualquer processo novo escrever
    # seu proprio status - isso ja causou operacoes duplicadas em producao.
    with _start_lock:
        data = status.read_status(component)
        if _is_running(data):
            return jsonify({"ok": False, "error": "ja esta rodando"}), 409
        status.write_status(component, phase="iniciando", pid=None)

    log_path = LOG_DIR / f"{component}_run.log"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n\n=== iniciado pelo painel web em "
                   f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.flush()
    try:
        subprocess.Popen(
            [sys.executable, "-u", "-m", COMPONENT_MODULES[component]],
            cwd=str(SRC_DIR), stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/api/stop/<component>", methods=["POST"])
def api_stop(component):
    if component not in COMPONENT_MODULES:
        return jsonify({"ok": False, "error": "componente desconhecido"}), 400

    data = status.read_status(component)
    pid = data.get("pid") if data else None
    if pid is None:
        return jsonify({"ok": False, "error": "nao esta rodando"}), 409

    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    status.clear_status(component)
    return jsonify({"ok": True})


def _stop_components(keys):
    stopped = []
    for key in keys:
        data = status.read_status(key)
        pid = data.get("pid") if data else None
        if pid is not None:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            status.clear_status(key)
            stopped.append(key)
    return stopped


def _stop_all_components():
    return _stop_components(list(COMPONENT_MODULES))


@app.route("/api/champion/pin", methods=["POST"])
def api_pin_champion():
    """Escolha manual: forca um trial especifico (linha da tabela de
    estrategias) a virar o campeao usado pelo Executor agora, ignorando qual
    seria a escolha automatica do Gerador. Fica pinado ate o usuario
    explicitamente voltar pro modo automatico."""
    body = request.get_json(force=True, silent=True) or {}
    trial_id = body.get("trial_id")
    if trial_id is None:
        return jsonify({"ok": False, "error": "trial_id nao informado"}), 400
    try:
        registry.pin_champion(config.ACTIVE, config.DURATION_MINUTES, int(trial_id))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


@app.route("/api/champion/unpin", methods=["POST"])
def api_unpin_champion():
    registry.unpin_champion(config.ACTIVE, config.DURATION_MINUTES)
    return jsonify({"ok": True})


@app.route("/api/reset/strategies", methods=["POST"])
def api_reset_strategies():
    """So zera tentativas/campeao/historico de campeao - NAO mexe em saldo,
    operacoes ou erros. Separado do reset de saldo de proposito: estrategias
    sao dificeis de achar, entao resetar o saldo nao deveria jogar fora uma
    ja encontrada."""
    stopped = _stop_components(["generator", "executor", "learner"])  # sync nao precisa parar, candles nao mudam
    backup_name = registry.create_backup(label="reset_estrategias")
    try:
        registry.reset_strategy_data()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"falha ao zerar estrategias: {exc}"}), 500
    return jsonify({"ok": True, "backup": backup_name, "stopped_components": stopped})


@app.route("/api/reset/balance", methods=["POST"])
def api_reset_balance():
    """So reseta o saldo da conta PRACTICE na IQ Option e zera o historico
    de operacoes (que nao faz mais sentido contra um saldo novo). NAO mexe
    em estrategias/campeao/erros."""
    stopped = _stop_components(["executor"])  # so ele depende do saldo/contador de trades
    backup_name = registry.create_backup(label="reset_saldo")

    try:
        registry.reset_balance_data()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"falha ao zerar operacoes: {exc}"}), 500

    balance = None
    balance_error = None
    try:
        from .broker import Broker
        broker = Broker().connect()
        balance = broker.reset_practice_balance()
    except Exception as exc:
        balance_error = str(exc)

    return jsonify({
        "ok": True, "backup": backup_name, "stopped_components": stopped,
        "practice_balance": balance, "balance_error": balance_error,
    })


@app.route("/api/backups", methods=["GET"])
def api_backups():
    return jsonify({"backups": registry.list_backups()})


@app.route("/api/backups/<name>", methods=["DELETE"])
def api_delete_backup(name):
    try:
        registry.delete_backup(name)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "backup nao encontrado"}), 404
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/api/backups/clear_old", methods=["POST"])
def api_clear_old_backups():
    body = request.get_json(force=True, silent=True) or {}
    keep = int(body.get("keep", 2))
    deleted = registry.clear_old_backups(keep=keep)
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/errors/clear", methods=["POST"])
def api_clear_errors():
    registry.clear_errors()
    return jsonify({"ok": True})


@app.route("/api/restore", methods=["POST"])
def api_restore():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name")
    if not name:
        return jsonify({"ok": False, "error": "nome do backup nao informado"}), 400

    stopped = _stop_all_components()
    try:
        registry.restore_backup(name)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "backup nao encontrado"}), 404
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "stopped_components": stopped})


if __name__ == "__main__":
    ip = _lan_ip()
    print(f"\n  Painel disponivel em:")
    print(f"    neste PC:        http://127.0.0.1:5000")
    print(f"    na sua rede:     http://{ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
