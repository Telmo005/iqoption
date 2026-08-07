"""Bootstrap de primeira instalacao (ex: numa VPS nova, depois de git clone).

strategies.db (2GB+, com todo o historico de treino/validacao) e candles.db
NAO vao pro git (veja .gitignore) - grandes demais e faceis de refazer do
zero (sync.py reconstroi candles.db sozinho). O que realmente vale a pena
preservar entre maquinas - campeao atual, historico de campeoes, escolha do
Aprendiz, e as avaliacoes de holdout que embasam tudo isso - fica guardado
num arquivo bem menor (strategies.seed.sqlite, so a fatia 'holdout' dos
trials) que ESSE sim vai pro git.

Este script copia os arquivos ".seed.sqlite" pro nome real (.db) SO se o
arquivo real ainda nao existir - nunca sobrescreve progresso ja feito na
maquina onde estiver rodando.

Uso (uma vez, logo apos o git clone):
    cd iqoptionapi/src
    python -m ai_trader.seed_setup
"""
import shutil
from pathlib import Path

_DIR = Path(__file__).parent

PAIRS = [
    ("strategies.seed.sqlite", "strategies.db"),
    ("candles.seed.sqlite", "candles.db"),
]


def main():
    for seed_name, real_name in PAIRS:
        seed_path = _DIR / seed_name
        real_path = _DIR / real_name
        if not seed_path.exists():
            print(f"[pular] {seed_name} nao existe neste checkout")
            continue
        if real_path.exists():
            print(f"[pular] {real_name} ja existe - nao sobrescrevendo progresso existente")
            continue
        shutil.copy2(seed_path, real_path)
        print(f"[ok] {seed_name} -> {real_name}")


if __name__ == "__main__":
    main()
