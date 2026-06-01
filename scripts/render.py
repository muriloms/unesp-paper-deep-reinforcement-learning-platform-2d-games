#!/usr/bin/env python3
"""
CLI de visualização: gera GIFs do agente jogando.

3 modos:
  agent      — 1 modelo, 1 fase, 1 GIF
  compare    — DQN vs PPO vs A2C lado-a-lado (mesma fase/seed)
  evolution  — mesmo algoritmo em diferentes checkpoints

Exemplos:
  # GIF do PPO na fase 1-1 com seed 42
  python scripts/render.py agent --algo PPO --stage 1-1 --seed 42

  # Comparação DQN/PPO/A2C na fase 4-1 (seed 42 dos modelos)
  python scripts/render.py compare --stage 4-1 --seed 42

  # Evolução temporal do PPO (5 checkpoints uniformes)
  python scripts/render.py evolution --algo PPO --stage 1-1 --seed 42

  # Gera GIFs de todas as configurações treinadas
  python scripts/render.py all
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.visualization import (
    render_agent_episode,
    render_algos_comparison,
    render_temporal_evolution,
)
from src.utils import setup_warnings


def cmd_agent(args):
    render_agent_episode(
        algo=args.algo, stage=args.stage, seed=args.seed,
        max_steps=args.max_steps, fps=args.fps,
        render_seed=args.render_seed,
        variant=args.variant,
    )
    return 0


def cmd_compare(args):
    out = render_algos_comparison(
        stage=args.stage, seed=args.seed,
        algos=args.algos,
        max_steps=args.max_steps, fps=args.fps,
        render_seed=args.render_seed,
        variant=args.variant,
    )
    return 0 if out else 1


def cmd_evolution(args):
    out = render_temporal_evolution(
        algo=args.algo, stage=args.stage, seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        max_steps=args.max_steps, fps=args.fps,
        render_seed=args.render_seed,
        variant=args.variant,
    )
    return 0 if out else 1


def _parse_model_name(stem: str):
    """
    Parseia nomes como:
        DQN_stage1-1_seed42                → ("DQN", "1-1", 42, None)
        PPO_stage1-1_seed42_shape          → ("PPO", "1-1", 42, "shape")
        A2C_stage4-1_seed2024_<futuro>     → ("A2C", "4-1", 2024, "<futuro>")

    Retorna (algo, stage, seed, variant) ou None se padrão inválido.
    """
    import re
    m = re.match(r"^(DQN|PPO|A2C)_stage([^_]+)_seed(\d+)(?:_(.+))?$", stem)
    if not m:
        return None
    algo, stage, seed_str, variant = m.groups()
    return algo, stage, int(seed_str), variant   # variant é None se ausente


def cmd_all(args):
    """Gera GIF de cada modelo treinado + compare por configuração."""
    print("Modo ALL — gerando GIFs para todos os modelos disponíveis...\n")
    n = 0
    skipped = 0
    for path in sorted(config.MODELS_DIR.glob("*.zip")):
        parsed = _parse_model_name(path.stem)
        if parsed is None:
            skipped += 1
            continue
        algo, stage, seed, variant = parsed
        tag = f" [{variant}]" if variant else ""
        print(f"\n▶ {path.stem}{tag}")
        try:
            render_agent_episode(
                algo=algo, stage=stage, seed=seed,
                max_steps=args.max_steps, fps=args.fps,
                render_seed=args.render_seed,
                variant=variant,
            )
            n += 1
        except Exception as e:
            print(f"  ✗ FALHOU: {e}")

    # compare por (stage, seed, variant) — só compara entre algos da MESMA variante
    configs_seen = set()
    for path in sorted(config.MODELS_DIR.glob("*.zip")):
        parsed = _parse_model_name(path.stem)
        if parsed is None:
            continue
        _, stage, seed, variant = parsed
        configs_seen.add((stage, seed, variant))

    for stage, seed, variant in sorted(configs_seen, key=lambda t: (t[0], t[1], t[2] or "")):
        tag = f" [{variant}]" if variant else ""
        print(f"\n▶ comparison stage={stage} seed={seed}{tag}")
        try:
            render_algos_comparison(
                stage=stage, seed=seed,
                max_steps=args.max_steps, fps=args.fps,
                render_seed=args.render_seed,
                variant=variant,
            )
        except Exception as e:
            print(f"  ✗ FALHOU: {e}")

    print(f"\n✓ {n} GIFs individuais gerados.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Renderização de GIFs (agente individual, comparação, evolução)",
    )
    p.add_argument("--max-steps", type=int, default=3000,
                   help="Máximo de passos do policy por episódio (default: 3000)")
    p.add_argument("--fps", type=int, default=15,
                   help="FPS do GIF — 15 ≈ 60 FPS efetivos do NES (default: 15)")
    p.add_argument("--render-seed", type=int, default=999,
                   help="Seed do episódio renderizado (default: 999)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # agent
    sa = sub.add_parser("agent", help="GIF de UM modelo")
    sa.add_argument("--algo", required=True, choices=config.ALGOS)
    sa.add_argument("--stage", required=True)
    sa.add_argument("--seed", type=int, required=True)
    sa.add_argument("--variant", default=None,
                    help="Variante do modelo (ex: 'shape' p/ reward shaping)")
    sa.set_defaults(func=cmd_agent)

    # compare
    sc = sub.add_parser("compare", help="Comparação DQN vs PPO vs A2C lado-a-lado")
    sc.add_argument("--stage", required=True)
    sc.add_argument("--seed", type=int, required=True)
    sc.add_argument("--algos", nargs="+", choices=config.ALGOS, default=None,
                    help="Subset de algos a comparar (default: todos)")
    sc.add_argument("--variant", default=None,
                    help="Variante (ex: 'shape')")
    sc.set_defaults(func=cmd_compare)

    # evolution
    se = sub.add_parser("evolution", help="Evolução temporal (checkpoints)")
    se.add_argument("--algo", required=True, choices=config.ALGOS)
    se.add_argument("--stage", required=True)
    se.add_argument("--seed", type=int, required=True)
    se.add_argument("--n-checkpoints", type=int, default=5)
    se.add_argument("--variant", default=None,
                    help="Variante (ex: 'shape')")
    se.set_defaults(func=cmd_evolution)

    # all
    sall = sub.add_parser("all", help="Renderiza tudo que estiver disponível")
    sall.set_defaults(func=cmd_all)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_warnings()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
