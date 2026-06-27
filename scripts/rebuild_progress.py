#!/usr/bin/env python3
"""
Reconstrói o progress.json a partir dos modelos e checkpoints já existentes.

Use isto UMA VEZ após atualizar para a versão com manifesto, para registrar
os treinos que você já completou ANTES do manifesto existir.

Heurística por experimento:
  - Se há modelo final (.zip) E checkpoints → progresso = maior valor entre
    o último checkpoint e o alvo inferido. Por padrão assume que o modelo
    final corresponde ao --assume-target (default 2_000_000), já que o último
    checkpoint costuma ficar atrás do alvo real (ex: 1.9M para um alvo de 2M).
  - Se há modelo final SEM checkpoints → registra --assume-target.
  - Se há só checkpoints (sem modelo final) → registra o maior checkpoint
    (treino incompleto; o resume continuará dali).

Uso:
    # Assume que todo modelo final completo chegou a 2M
    python scripts/rebuild_progress.py --assume-target 2000000

    # Para um projeto que parou em 500k
    python scripts/rebuild_progress.py --assume-target 500000

    # Ver o que faria sem gravar
    python scripts/rebuild_progress.py --assume-target 2000000 --dry-run
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.training import (
    _find_latest_checkpoint, update_progress, load_progress, PROGRESS_FILE,
)


def parse_model_name(stem: str):
    m = re.match(r"^(DQN|PPO|A2C)_stage([^_]+)_seed(\d+)(?:_(.+))?$", stem)
    if not m:
        return None
    algo, stage, seed_str, variant = m.groups()
    return algo, stage, int(seed_str), variant


def main() -> int:
    p = argparse.ArgumentParser(description="Reconstrói progress.json de modelos existentes.")
    p.add_argument("--assume-target", type=int, default=2_000_000,
                   help="Timesteps a registrar p/ modelos finais completos (default 2M)")
    p.add_argument("--dry-run", action="store_true", help="Mostra sem gravar")
    args = p.parse_args()

    if not config.MODELS_DIR.exists():
        print(f"Pasta de modelos não existe: {config.MODELS_DIR}")
        return 1

    existing = load_progress()
    print(f"Manifesto atual: {len(existing)} entradas em {PROGRESS_FILE}")
    print(f"Alvo assumido p/ modelos completos: {args.assume_target:,}\n")

    n = 0
    for model_path in sorted(config.MODELS_DIR.glob("*.zip")):
        parsed = parse_model_name(model_path.stem)
        if parsed is None:
            continue
        algo, stage, seed, variant = parsed
        exp_id = model_path.stem  # já inclui sufixo _shape se houver

        _, ckpt_steps = _find_latest_checkpoint(exp_id)

        # Modelo final existe → assume alvo (a menos que checkpoint o exceda)
        registered = max(args.assume_target, ckpt_steps)

        extra = {"algo": algo, "stage": stage, "seed": seed,
                 "reward_shaping": bool(variant == "shape"),
                 "model": model_path.name, "source": "rebuild"}

        if args.dry_run:
            print(f"  [dry] {exp_id}: {registered:,} steps "
                  f"(ckpt={ckpt_steps:,})")
        else:
            update_progress(exp_id, timesteps=registered, extra=extra)
            print(f"  ✓ {exp_id}: {registered:,} steps")
        n += 1

    print(f"\n{'(dry-run) ' if args.dry_run else ''}{n} experimento(s) processado(s).")
    if not args.dry_run:
        print(f"Manifesto salvo em {PROGRESS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
