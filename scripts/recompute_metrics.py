#!/usr/bin/env python3
"""
Re-avalia todos os modelos `.zip` salvos, gerando CSVs corrigidos com o
contador de mortes correto. Não retreina nada — só carrega cada modelo e
roda N episódios determinísticos, persistindo as métricas.

Por que existe: o callback original (durante o treino) contava `deaths` apenas
quando `info['life']` decrementava. Mas no Super Mario Bros, colisão com Goomba
pequeno termina o episódio com `done=True` SEM decrementar life — então a coluna
`deaths` ficava 0 em quase todos os episódios. Este script aplica a heurística
"se o episódio terminou sem flag e sem death, conta como 1 morte".

Uso:
    # Re-avalia todos os modelos (escreve em logs/<exp_id>_eval.csv)
    python scripts/recompute_metrics.py

    # Apenas alguns algos
    python scripts/recompute_metrics.py --algos PPO A2C

    # Sobrescreve o CSV original em vez de criar `_eval`
    python scripts/recompute_metrics.py --overwrite-original

    # Mais episódios por avaliação (default 5)
    python scripts/recompute_metrics.py --n-episodes 10

Após rodar, execute novamente:
    python scripts/analyze.py
para gerar as métricas/plots com os números corretos.
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C

from src import config
from src.env import make_vec_env_mario
from src.utils import get_device, setup_warnings

_ALGO_CLS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


def parse_model_name(stem: str):
    """Mesma lógica do render.py — tolera o sufixo _shape opcional."""
    m = re.match(r"^(DQN|PPO|A2C)_stage([^_]+)_seed(\d+)(?:_(.+))?$", stem)
    if not m:
        return None
    algo, stage, seed_str, variant = m.groups()
    return algo, stage, int(seed_str), variant


def evaluate_model(
    model, stage: str, seed: int,
    n_episodes: int,
    verbose: bool = True,
) -> list[dict]:
    """Roda N episódios determinísticos e retorna lista de métricas por episódio."""
    eval_env = make_vec_env_mario(
        stage=stage, n_envs=1, seed=seed + 9999, use_subproc=False,
    )

    results = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        max_x = 0
        deaths = 0
        frames = 0
        flag_get = False
        time_left = None
        prev_life = None

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            info0 = info[0]
            ep_reward += float(reward[0])
            frames += 1

            x = int(info0.get("x_pos", 0))
            if x > max_x:
                max_x = x
            life = info0.get("life", None)
            if prev_life is not None and life is not None and life < prev_life:
                deaths += 1
            prev_life = life
            if info0.get("flag_get", False):
                flag_get = True
            time_left = info0.get("time", time_left)

        # **CORREÇÃO**: game-over por Goomba pequeno não decrementa life.
        # Se episódio terminou sem flag e sem death contabilizada, é morte.
        if not flag_get and deaths == 0 and frames > 0:
            deaths = 1

        results.append(dict(
            episode=ep,
            reward=ep_reward,
            max_x_pos=max_x,
            flag_get=int(flag_get),
            deaths=deaths,
            frames=frames,
            time_left=time_left,
        ))
        if verbose:
            flag_tag = "✓" if flag_get else "✗"
            print(f"    ep{ep}: r={ep_reward:+7.1f}  max_x={max_x:4d}  "
                  f"flag={flag_tag}  deaths={deaths}  frames={frames}")

    eval_env.close()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-avalia modelos treinados gerando CSVs com deaths corretos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algos", nargs="+", default=None, choices=config.ALGOS,
                   help="Subset de algoritmos a reavaliar")
    p.add_argument("--stages", nargs="+", default=None,
                   help="Subset de fases (default: todas presentes)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset de seeds")
    p.add_argument("--variant", default=None,
                   help="Se especificado, processa só modelos com este sufixo (ex: 'shape')")
    p.add_argument("--n-episodes", type=int, default=5,
                   help="Episódios por modelo")
    p.add_argument("--timestep-tag", type=int, default=500_000,
                   help="Timestep registrado no CSV (default: 500000, o final)")
    p.add_argument("--overwrite-original", action="store_true",
                   help="Sobrescreve o CSV original em logs/. Default: salva como _eval.csv")
    p.add_argument("--skip-existing", action="store_true",
                   help="Pula modelos cujo CSV de reavaliação já existe")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_warnings()
    config.ensure_dirs()

    device = get_device()
    print(f"Device: {device}")
    print(f"Modelos em: {config.MODELS_DIR}")
    print(f"CSVs serão escritos em: {config.LOGS_DIR}\n")

    # Lista todos os modelos finais (não checkpoints intermediários)
    model_paths = sorted(config.MODELS_DIR.glob("*.zip"))
    if not model_paths:
        print("Nenhum modelo encontrado.")
        return 1

    # Filtra por critério do usuário
    todo = []
    for path in model_paths:
        parsed = parse_model_name(path.stem)
        if parsed is None:
            continue
        algo, stage, seed, variant = parsed

        if args.algos and algo not in args.algos:
            continue
        if args.stages and stage not in args.stages:
            continue
        if args.seeds and seed not in args.seeds:
            continue
        if args.variant is not None and variant != args.variant:
            continue
        todo.append((path, algo, stage, seed, variant))

    print(f"Vai reavaliar {len(todo)} modelo(s).\n")
    if not todo:
        return 1

    t0_total = time.time()
    n_ok = 0
    n_fail = 0

    for idx, (path, algo, stage, seed, variant) in enumerate(todo, 1):
        tag = f"_{variant}" if variant else ""
        exp_id_full = f"{algo}_stage{stage}_seed{seed}{tag}"

        suffix = "" if args.overwrite_original else "_eval"
        out_csv = config.LOGS_DIR / f"{exp_id_full}{suffix}.csv"

        if args.skip_existing and out_csv.exists():
            print(f"[{idx}/{len(todo)}] {exp_id_full}  →  já existe ({out_csv.name}), pulando.")
            continue

        print(f"[{idx}/{len(todo)}] {exp_id_full}")
        try:
            cls = _ALGO_CLS[algo]
            model = cls.load(path, device=device)
            t0 = time.time()
            results = evaluate_model(
                model, stage=stage, seed=seed,
                n_episodes=args.n_episodes,
                verbose=True,
            )
            elapsed = time.time() - t0
            n_ok += 1

            # Salva no mesmo formato dos CSVs do callback (timestep,episode,...)
            df = pd.DataFrame([
                {"timestep": args.timestep_tag, **r} for r in results
            ])
            df.to_csv(out_csv, index=False)

            # Sumário
            tau = df["flag_get"].mean()
            mean_x = df["max_x_pos"].mean()
            mean_d = df["deaths"].mean()
            print(f"  ✓ {elapsed:.1f}s  |  τ={tau:.0%}  max_x_avg={mean_x:.0f}  "
                  f"deaths_avg={mean_d:.1f}  →  {out_csv.name}\n")
        except Exception as e:
            print(f"  ✗ FALHOU: {e}\n")
            n_fail += 1

    total_elapsed = time.time() - t0_total
    print(f"{'='*70}")
    print(f"RESUMO: {n_ok} ok, {n_fail} falhas em {total_elapsed/60:.1f} min")
    if not args.overwrite_original:
        print(f"\nPróximo passo: ")
        print(f"  - Para usar os CSVs corrigidos na análise, renomeie ou edite")
        print(f"    scripts/analyze.py para apontar para *_eval.csv.")
        print(f"  - Alternativa: rode novamente com --overwrite-original")
        print(f"    (cuidado: substitui os CSVs originais).")
    print('='*70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
