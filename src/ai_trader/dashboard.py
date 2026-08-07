"""Painel de controle do AI trader - arquitetura de 3 processos
independentes (sincronizador de dados, gerador de estrategias, executor de
operacoes). Cada um tem seu proprio botao ligar/desligar. So LE status_*.json
e a base (registry.py); nunca faz nada sozinho alem do que os botoes mandam.

Uso:
    cd iqoptionapi/src
    python -m ai_trader.dashboard
"""
import subprocess
import sys
import time as _time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont, messagebox

from . import registry, status

BG = "#0f1117"
PANEL_BG = "#171a24"
CARD_BG = "#1d2130"
FG = "#e6e8ef"
MUTED = "#7d8296"
GREEN = "#3ddc84"
RED = "#ff5c5c"
YELLOW = "#f5c542"
BLUE = "#5b8cff"

SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]
STALE_SECONDS = 25

SRC_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent

COMPONENTS = {
    "sync": {
        "label": "SINCRONIZADOR", "module": "ai_trader.sync",
        "log": LOG_DIR / "sync_run.log",
        "phases": {
            "conectando": ("conectando...", YELLOW),
            "em_dia": ("dados em dia", GREEN),
            "aguardando": ("aguardando proximo ciclo", MUTED),
            "erro": ("erro - tentando de novo", RED),
        },
    },
    "generator": {
        "label": "GERADOR DE ESTRATEGIAS", "module": "ai_trader.generator",
        "log": LOG_DIR / "generator_run.log",
        "phases": {
            "aguardando_dados": ("esperando dados do sincronizador", YELLOW),
            "gerando": ("gerando e testando estrategias...", BLUE),
            "erro": ("erro - tentando de novo", RED),
        },
    },
    "executor": {
        "label": "EXECUTOR", "module": "ai_trader.executor",
        "log": LOG_DIR / "executor_run.log",
        "phases": {
            "iniciando": ("iniciando", YELLOW),
            "aguardando_estrategia": ("esperando o gerador achar uma estrategia", YELLOW),
            "bloqueado_dinheiro_real": ("bloqueado: campeao nao confirmado", RED),
            "verificando_sinal": ("verificando sinal...", BLUE),
            "aguardando_sinal": ("aguardando proximo candle", MUTED),
            "operando": ("operando...", YELLOW),
            "aguardando_resultado": ("aguardando resultado...", YELLOW),
            "parado": ("parado (limite atingido)", RED),
            "erro": ("erro - tentando de novo", RED),
            "encerrado": ("encerrado", MUTED),
        },
    },
}
BUSY_PHASES = {
    "conectando", "gerando", "verificando_sinal", "operando",
    "aguardando_resultado", "iniciando",
}


class ComponentCard(tk.Frame):
    def __init__(self, parent, key, spec, dashboard):
        super().__init__(parent, bg=CARD_BG, highlightthickness=0)
        self.key = key
        self.spec = spec
        self.dashboard = dashboard
        self.process = None
        self._spin_idx = 0

        pad = tk.Frame(self, bg=CARD_BG)
        pad.pack(fill="x", padx=14, pady=10)

        top = tk.Frame(pad, bg=CARD_BG)
        top.pack(fill="x")
        tk.Label(top, text=spec["label"], font=dashboard.f_label_bold, fg=FG, bg=CARD_BG).pack(side="left")
        self.pid_lbl = tk.Label(top, text="", font=dashboard.f_label, fg=MUTED, bg=CARD_BG)
        self.pid_lbl.pack(side="right")

        mid = tk.Frame(pad, bg=CARD_BG)
        mid.pack(fill="x", pady=(4, 8))
        self.spinner_lbl = tk.Label(mid, text="●", font=("Segoe UI", 12), fg=MUTED, bg=CARD_BG)
        self.spinner_lbl.pack(side="left")
        self.phase_lbl = tk.Label(mid, text="parado", font=dashboard.f_value_sm, fg=MUTED, bg=CARD_BG)
        self.phase_lbl.pack(side="left", padx=(8, 0))

        btns = tk.Frame(pad, bg=CARD_BG)
        btns.pack(fill="x")
        self.start_btn = tk.Button(
            btns, text="▶ Iniciar", font=dashboard.f_btn_sm, bg=GREEN, fg="#0a0a0a",
            bd=0, padx=12, pady=5, cursor="hand2", command=self.on_start)
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(
            btns, text="■ Parar", font=dashboard.f_btn_sm, bg=RED, fg="#0a0a0a",
            bd=0, padx=12, pady=5, cursor="hand2", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.detail_lbl = tk.Label(pad, text="", font=dashboard.f_mono, fg=MUTED, bg=CARD_BG,
                                   wraplength=250, justify="left")
        self.detail_lbl.pack(anchor="w", pady=(6, 0))

    def on_start(self):
        data = status.read_status(self.key)
        if self.dashboard.is_running(data):
            messagebox.showinfo("AI Trader", f"{self.spec['label']} ja esta rodando.")
            return
        log_path = self.spec["log"]
        log_file = open(log_path, "a", encoding="utf-8")
        log_file.write(f"\n\n=== iniciado pelo painel em {_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.flush()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", self.spec["module"]],
                cwd=str(SRC_DIR), stdout=log_file, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            messagebox.showerror("AI Trader", f"Falha ao iniciar {self.spec['label']}: {exc}")

    def on_stop(self):
        data = status.read_status(self.key)
        pid = data.get("pid") if data else None
        if pid is None and self.process is None:
            messagebox.showinfo("AI Trader", f"{self.spec['label']} nao esta rodando.")
            return
        if not messagebox.askyesno("AI Trader", f"Parar {self.spec['label']} agora?"):
            return
        if pid is not None:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        if self.process is not None:
            self.process.terminate()
            self.process = None
        status.clear_status(self.key)

    def render(self, data, running, spin_idx):
        self.start_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")

        if not data or not running:
            self.phase_lbl.config(text="parado", fg=MUTED)
            self.spinner_lbl.config(text="●", fg=MUTED)
            self.pid_lbl.config(text="")
            self.detail_lbl.config(text="")
            return

        phase = data.get("phase", "?")
        label, color = self.spec["phases"].get(phase, (phase, FG))
        busy = phase in BUSY_PHASES
        self.spinner_lbl.config(text=SPINNER_FRAMES[spin_idx] if busy else "●", fg=color)
        self.phase_lbl.config(text=label, fg=color)
        self.pid_lbl.config(text=f"pid {data.get('pid')}")

        if self.key == "sync":
            n = data.get("candles_cached", 0)
            self.detail_lbl.config(text=f"{n} candles no cache local")
        elif self.key == "generator":
            trials = data.get("trials_tested_total")
            champ = data.get("champion_name")
            txt = f"{trials or 0} estrategias testadas"
            if champ:
                conf = "confirmado" if data.get("champion_confirmed") else "nao confirmado"
                edge = data.get("champion_edge")
                txt += f"\ncampeao: {champ} ({conf}, edge {edge*100:+.2f}%)" if edge is not None else f"\ncampeao: {champ}"
            self.detail_lbl.config(text=txt)
        elif self.key == "executor":
            wins = data.get("wins", 0) or 0
            losses = data.get("losses", 0) or 0
            pnl = data.get("daily_pnl", 0.0) or 0.0
            self.detail_lbl.config(text=f"vitorias {wins} · derrotas {losses} · pnl {pnl:+.2f}")


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Trader - IQ Option")
        self.configure(bg=BG)
        self.geometry("620x820")
        self.minsize(560, 700)

        self._spin_idx = 0
        self._build_fonts()
        self._build_layout()
        self._tick()

    def _build_fonts(self):
        self.f_title = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.f_label = tkfont.Font(family="Segoe UI", size=9)
        self.f_label_bold = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.f_value = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.f_value_sm = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=8)
        self.f_btn_sm = tkfont.Font(family="Segoe UI", size=9, weight="bold")

    def _panel(self, parent, **kw):
        p = tk.Frame(parent, bg=PANEL_BG, highlightthickness=0)
        p.pack(fill="x", padx=14, pady=6, **kw)
        return p

    def _build_layout(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(header, text="AI TRADER", font=self.f_title, fg=FG, bg=BG).pack(side="left")
        self.pair_lbl = tk.Label(header, text="", font=self.f_label, fg=MUTED, bg=BG)
        self.pair_lbl.pack(side="right")

        # ---- 3 cartoes de componente ----
        cards_row = tk.Frame(self, bg=BG)
        cards_row.pack(fill="x", padx=10, pady=4)
        self.cards = {}
        for key, spec in COMPONENTS.items():
            card = ComponentCard(cards_row, key, spec, self)
            card.pack(side="left", fill="both", expand=True, padx=4)
            self.cards[key] = card

        # ---- campeao atual ----
        champ_panel = self._panel(self)
        champ_inner = tk.Frame(champ_panel, bg=PANEL_BG)
        champ_inner.pack(fill="x", padx=14, pady=10, anchor="w")
        tk.Label(champ_inner, text="CAMPEAO ATUAL (usado pelo executor)", font=self.f_label,
                 fg=MUTED, bg=PANEL_BG).pack(anchor="w")
        self.champ_name_lbl = tk.Label(champ_inner, text="-", font=self.f_value_sm, fg=FG, bg=PANEL_BG)
        self.champ_name_lbl.pack(anchor="w", pady=(2, 2))
        self.champ_badge_lbl = tk.Label(champ_inner, text="", font=self.f_label, fg=MUTED, bg=PANEL_BG)
        self.champ_badge_lbl.pack(anchor="w")

        # ---- stats do executor ----
        stats_panel = self._panel(self)
        stats_inner = tk.Frame(stats_panel, bg=PANEL_BG)
        stats_inner.pack(fill="x", padx=14, pady=10)
        self.balance_val, _ = self._stat(stats_inner, "SALDO", 0)
        self.pnl_val, _ = self._stat(stats_inner, "PNL DO DIA", 1)
        self.wl_val, _ = self._stat(stats_inner, "VIT/DERR", 2, small=True)
        self.trades_val, _ = self._stat(stats_inner, "OPERACOES", 3, small=True)

        # ---- eventos ----
        tk.Label(self, text="EVENTOS (todos os componentes)", font=self.f_label, fg=MUTED, bg=BG).pack(
            anchor="w", padx=14, pady=(8, 2))
        events_frame = tk.Frame(self, bg=PANEL_BG)
        events_frame.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self.events_text = tk.Text(
            events_frame, bg=PANEL_BG, fg=FG, font=self.f_mono, bd=0,
            highlightthickness=0, wrap="word", state="disabled", height=8)
        self.events_text.pack(fill="both", expand=True, padx=10, pady=8)
        self.events_text.tag_configure("win", foreground=GREEN)
        self.events_text.tag_configure("loss", foreground=RED)
        self.events_text.tag_configure("muted", foreground=MUTED)
        self.events_text.tag_configure("comp", foreground=BLUE)

        # ---- erros / anomalias ----
        tk.Label(self, text="ERROS E ANOMALIAS (pra investigar depois)", font=self.f_label,
                 fg=MUTED, bg=BG).pack(anchor="w", padx=14, pady=(4, 2))
        errors_frame = tk.Frame(self, bg=PANEL_BG)
        errors_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.errors_text = tk.Text(
            errors_frame, bg=PANEL_BG, fg=RED, font=self.f_mono, bd=0,
            highlightthickness=0, wrap="word", state="disabled", height=5)
        self.errors_text.pack(fill="both", expand=True, padx=10, pady=8)

    def _stat(self, parent, label, col, small=False):
        cell = tk.Frame(parent, bg=PANEL_BG)
        cell.grid(row=0, column=col, sticky="w", padx=(0, 24))
        parent.grid_columnconfigure(col, weight=1)
        tk.Label(cell, text=label, font=self.f_label, fg=MUTED, bg=PANEL_BG).pack(anchor="w")
        val = tk.Label(cell, text="-", font=(self.f_value_sm if small else self.f_value),
                       fg=FG, bg=PANEL_BG)
        val.pack(anchor="w")
        return val, cell

    @staticmethod
    def is_running(data):
        if not data:
            return False
        age = _time.time() - data.get("updated_at", 0)
        return age < STALE_SECONDS and data.get("phase") != "encerrado"

    def _tick(self):
        all_status = {key: status.read_status(key) for key in COMPONENTS}
        running = {key: self.is_running(data) for key, data in all_status.items()}

        for key, card in self.cards.items():
            card.render(all_status[key], running[key], self._spin_idx)

        exec_data = all_status.get("executor")
        self.pair_lbl.config(
            text=f"{exec_data.get('active', 'EURUSD-OTC')} · conta {exec_data.get('account_mode', '?')}"
            if exec_data else "")

        self._render_champion()
        self._render_executor_stats(exec_data, running.get("executor"))
        self._render_events(all_status)
        self._render_errors()

        self._spin_idx = (self._spin_idx + 1) % len(SPINNER_FRAMES)
        self.after(700, self._tick)

    def _render_champion(self):
        try:
            from . import config
            champ = registry.get_champion(config.ACTIVE, config.DURATION_MINUTES)
        except Exception:
            champ = None
        if champ is None:
            self.champ_name_lbl.config(text="(nenhum ainda)")
            self.champ_badge_lbl.config(text="")
            return
        self.champ_name_lbl.config(text=champ["name"])
        confirmed = champ["confirmed"]
        age_min = (_time.time() - champ["promoted_at"]) / 60
        badge = "CONFIRMADO NO HOLDOUT" if confirmed else "NAO CONFIRMADO (demo)"
        self.champ_badge_lbl.config(
            text=f"{badge}  ·  edge {champ['edge']*100:+.2f}%  ·  achado ha {age_min:.0f} min",
            fg=GREEN if confirmed else YELLOW)

    def _render_executor_stats(self, data, running):
        if not data or not running:
            self.balance_val.config(text="-")
            self.pnl_val.config(text="-", fg=FG)
            self.wl_val.config(text="-")
            self.trades_val.config(text="-")
            return
        balance = data.get("balance")
        self.balance_val.config(text=f"{balance:.2f}" if balance is not None else "-")
        pnl = data.get("daily_pnl", 0.0) or 0.0
        self.pnl_val.config(text=f"{pnl:+.2f}", fg=GREEN if pnl > 0 else (RED if pnl < 0 else FG))
        wins = data.get("wins", 0) or 0
        losses = data.get("losses", 0) or 0
        self.wl_val.config(text=f"{wins}/{losses}")
        trades_done = data.get("trades_done", 0) or 0
        max_trades = data.get("max_trades")
        self.trades_val.config(text=f"{trades_done}/{max_trades}" if max_trades else str(trades_done))

    def _render_events(self, all_status):
        merged = []
        for key, data in all_status.items():
            if not data:
                continue
            label = COMPONENTS[key]["label"]
            for line in (data.get("events") or [])[:8]:
                merged.append((line[:8], f"[{label}] {line[9:]}" if len(line) > 9 else f"[{label}] {line}"))
        merged.sort(key=lambda t: t[0], reverse=True)

        self.events_text.config(state="normal")
        self.events_text.delete("1.0", "end")
        for _, line in merged[:60]:
            tag = None
            if "GANHOU" in line:
                tag = "win"
            elif "PERDEU" in line:
                tag = "loss"
            elif "sem sinal" in line or "aguardando" in line:
                tag = "muted"
            self.events_text.insert("end", line + "\n", tag)
        self.events_text.config(state="disabled")

    def _render_errors(self):
        try:
            errors = registry.get_recent_errors(limit=15)
        except Exception:
            errors = []
        self.errors_text.config(state="normal")
        self.errors_text.delete("1.0", "end")
        if not errors:
            self.errors_text.insert("end", "(nenhum erro registrado ainda)")
        else:
            for e in errors:
                when = _time.strftime("%H:%M:%S", _time.localtime(e["created_at"]))
                self.errors_text.insert(
                    "end", f"{when} [{e['component']}] {e['error_type']}: {e['message']}\n")
        self.errors_text.config(state="disabled")


if __name__ == "__main__":
    Dashboard().mainloop()
