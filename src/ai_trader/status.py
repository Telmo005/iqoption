"""Escreve/le o estado de cada componente (sync, generator, executor) em
arquivos JSON separados - um por componente, pra tres processos
independentes poderem escrever o proprio status sem pisar um no outro.
Escrita atomica (escreve em .tmp e renomeia) pra o dashboard nunca ler um
arquivo pela metade.

Cada write_status() carimba o PID do processo chamador automaticamente -
assim o dashboard sabe, so lendo o status, qual processo parar."""
import json
import os
import time
from pathlib import Path

_DIR = Path(__file__).parent
_PID = os.getpid()

COMPONENTS = ("sync", "generator", "executor", "learner")


def _path(component):
    if component not in COMPONENTS:
        raise ValueError(f"componente desconhecido: {component}")
    return _DIR / f"status_{component}.json"


def write_status(component, **fields):
    """Escrita atomica (escreve em .tmp e renomeia) - mas o rename em si pode
    falhar de forma transitoria no Windows (antivirus/indexador segurando o
    arquivo por uma fracao de segundo, PermissionError WinError 5). Isso NAO
    pode derrubar o processo chamador so porque um heartbeat falhou: melhor
    perder uma escrita de status do que crashar o sincronizador/gerador/
    executor inteiro (ja aconteceu de verdade em producao)."""
    fields.setdefault("pid", _PID)
    fields["component"] = component
    fields["updated_at"] = time.time()
    path = _path(component)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(fields, f)
        os.replace(tmp_path, path)
    except OSError:
        time.sleep(0.2)
        try:
            os.replace(tmp_path, path)
        except OSError:
            pass  # heartbeat perdido nao e fatal - a proxima escrita corrige


def read_status(component):
    try:
        with open(_path(component), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_all():
    return {c: read_status(c) for c in COMPONENTS}


def clear_status(component):
    try:
        _path(component).unlink()
    except FileNotFoundError:
        pass
