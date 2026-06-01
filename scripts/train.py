#!/usr/bin/env python3
"""
CLI de treinamento. Suporta:
  - smoke test: --profile smoke
  - experimento completo: --profile full
  - filtrar algos/fases/seeds via flags

Exemplos:
  # Smoke test (1 algo, 1 fase, 1 seed, ~2 min)
  python scripts/train.py --profile smoke

  # Treino completo da matriz inteira (36 runs)
  python scripts/train.py --profile full

  # Apenas PPO em todas as fases × seeds
  python scripts/train.py --profile full --algos PPO

  # PPO + A2C só no stage 1-1 (todas as seeds)
  python scripts/train.py --profile full --algos PPO A2C --stages 1-1

  # Retreinar um único run específico (overwrite)
  python scripts/train.py --profile full --algos DQN --stages 4-1 --seeds 42 --overwrite
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Garante que src/ é importável independente de onde o script é chamado
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.training import run_matrix
from src.utils import check_python_version, print_env_info, setup_warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treinamento Mario DRL (DQN/PPO/A2C × fases × seeds)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--profile", choices=list(config.PROFILES),
                   default="smoke",
                   help="Perfil de execução")
    p.add_argument("--algos", nargs="+", default=None,
                   choices=config.ALGOS,
                   help="Subset de algoritmos a treinar (default: todos do perfil)")
    p.add_argument("--stages", nargs="+", default=None,
                   help="Subset de fases (default: as do perfil)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset de seeds (default: as do perfil)")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignora estado prévio e retreina do zero")
    p.add_argument("--reward-shaping", action="store_true",
                   help=("Adiciona ProgressRewardWrapper (bonus por avanço em x_pos). "
                         "Cria artefatos com sufixo _shape, separados do baseline."))
    p.add_argument("--no-check-python", action="store_true",
                   help="Desativa verificação de versão do Python")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_warnings()
    if not args.no_check_python:
        check_python_version()
    print_env_info()

    profile = config.get_profile(args.profile)

    algos = args.algos or config.ALGOS
    stages = args.stages or profile["stages_to_run"]
    seeds = args.seeds or profile["seeds_to_run"]

    print()
    print(f"Profile         : {args.profile}")
    print(f"Algos           : {algos}")
    print(f"Stages          : {stages}")
    print(f"Seeds           : {seeds}")
    print(f"Total timesteps : {profile['total_timesteps']:,}")
    print(f"Total treinos   : {len(algos) * len(stages) * len(seeds)}")
    print(f"Reward shaping  : {'SIM (artefatos com sufixo _shape)' if args.reward_shaping else 'não'}")
    print(f"Output em       : {config.ROOT_DIR}")
    print()

    config.ensure_dirs()

    results = run_matrix(
        algos=algos,
        stages_to_run=stages,
        seeds_to_run=seeds,
        total_timesteps=profile["total_timesteps"],
        eval_freq=profile["eval_freq"],
        n_eval_episodes=profile["n_eval_episodes"],
        n_checkpoints=profile["n_checkpoints"],
        overwrite=args.overwrite,
        reward_shaping=args.reward_shaping,
    )

    # Sumário final
    n_ok = sum(1 for r in results if r[3] is not None)
    n_fail = len(results) - n_ok
    print(f"\n{'='*70}")
    print(f"RESUMO: {n_ok}/{len(results)} treinos com sucesso, {n_fail} falharam.")
    print('='*70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
