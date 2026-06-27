#!/usr/bin/env python3
"""
Reconstrói o progress.json a partir do estado REAL dos treinos no disco.

Fonte de verdade, em ordem de prioridade por experimento:
  1. Último timestep registrado no CSV de avaliação (logs/{exp_id}.csv)
     — é o valor mais confiável: reflete até onde o treino realmente avaliou.
  2. Maior checkpoint salvo (models/checkpoints/{exp_id}_{N}_steps.zip)
     — usado se não houver CSV.
  3. Se nada disso existir mas há modelo final, registra --fallback-target.

Por que NÃO assume um alvo fixo: treinos podem ter parado em pontos diferentes
(ex: 500k, 1.93M, 2M). Carimbar o mesmo valor em todos corromperia o manifesto
e faria o skip pular treinos incompletos.

Uso:
    # Reconstrói tudo a partir dos CSVs (recomendado)
    python scripts/rebuild_progress.py

    # Ver o que faria sem gravar
    python scripts/rebuild_progress.py --dry-run

    # Valor de fallback p/ modelos sem CSV nem checkpoint (raro)
    python scripts/rebuild_progress.py --fallback-target 500000
"""
from __future__ import annotations
import argparse
import csv
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


def last_timestep_from_csv(exp_id: str) -> int | None:
    """Último timestep do CSV de avaliação. None se ausente/vazio."""
    csv_path = config.LOGS_DIR / f"{exp_id}.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return int(rows[-1][0])   # coluna 'timestep' da última linha
    except (ValueError, OSError, IndexError):
        return None


def resolve_real_progress(exp_id: str, fallback_target: int) -> tuple[int, str]:
    """
    Determina o progresso real de um experimento.
    Returns (timesteps, source_label).
    """
    csv_ts = last_timestep_from_csv(exp_id)
    if csv_ts is not None:
        return csv_ts, "csv"

    _, ckpt_steps = _find_latest_checkpoint(exp_id)
    if ckpt_steps > 0:
        return ckpt_steps, "checkpoint"

    return fallback_target, "fallback"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Reconstrói progress.json a partir do estado real (CSV > checkpoint).")
    p.add_argument("--fallback-target", type=int, default=500_000,
                   help="Valor p/ modelos sem CSV nem checkpoint (default 500k)")
    p.add_argument("--dry-run", action="store_true", help="Mostra sem gravar")
    args = p.parse_args()

    if not config.MODELS_DIR.exists():
        print(f"Pasta de modelos não existe: {config.MODELS_DIR}")
        return 1

    existing = load_progress()
    print(f"Manifesto atual: {len(existing)} entradas em {PROGRESS_FILE}")
    print(f"Fonte de verdade: CSV > checkpoint > fallback ({args.fallback_target:,})\n")

    n = 0
    for model_path in sorted(config.MODELS_DIR.glob("*.zip")):
        parsed = parse_model_name(model_path.stem)
        if parsed is None:
            continue
        algo, stage, seed, variant = parsed
        exp_id = model_path.stem

        timesteps, source = resolve_real_progress(exp_id, args.fallback_target)

        extra = {"algo": algo, "stage": stage, "seed": seed,
                 "reward_shaping": bool(variant == "shape"),
                 "model": model_path.name, "source": f"rebuild_{source}"}

        if args.dry_run:
            print(f"  [dry] {exp_id}: {timesteps:,} steps (via {source})")
        else:
            update_progress(exp_id, timesteps=timesteps, extra=extra)
            print(f"  ✓ {exp_id}: {timesteps:,} steps (via {source})")
        n += 1

    print(f"\n{'(dry-run) ' if args.dry_run else ''}{n} experimento(s) processado(s).")
    if not args.dry_run:
        print(f"Manifesto salvo em {PROGRESS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
