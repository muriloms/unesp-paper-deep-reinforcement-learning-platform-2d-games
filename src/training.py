"""
Função de treinamento e loop completo.

`train_one()` é idempotente E retomável:
- Se modelo final + CSV existem → pula
- Se existe checkpoint parcial → carrega e treina só os timesteps restantes
- Caso contrário → começa do zero

Essencial para experimentos longos que podem ser interrompidos.
"""
from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from .config import (
    HPARAMS, MODELS_DIR, CKPT_DIR, LOGS_DIR, TB_DIR,
    experiment_id, ensure_dirs,
)
from .env import make_vec_env_mario
from .callbacks import MarioEvalCallback
from .utils import set_global_seed, get_device


# Mapeia string → classe SB3
_ALGO_CLS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


def _find_latest_checkpoint(exp_id: str) -> tuple[Optional[Path], int]:
    """
    Procura o checkpoint mais recente de um exp_id.

    Returns:
        (path, timesteps_done) — ou (None, 0) se nada encontrado.

    Checkpoints seguem o padrão do CheckpointCallback do SB3:
        CKPT_DIR/{exp_id}_{N}_steps.zip
    """
    if not CKPT_DIR.exists():
        return None, 0
    pattern = re.compile(rf"^{re.escape(exp_id)}_(\d+)_steps\.zip$")
    candidates = []
    for p in CKPT_DIR.glob(f"{exp_id}_*_steps.zip"):
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None, 0
    candidates.sort()
    return candidates[-1][1], candidates[-1][0]


def train_one(
    algo: str,
    stage: str,
    seed: int,
    *,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    n_checkpoints: int = 5,
    overwrite: bool = False,
    verbose: int = 1,
    reward_shaping: bool = False,
) -> Path:
    """
    Treina UM agente (algo, stage, seed). Idempotente E retomável.

    Args:
        algo:             "DQN" | "PPO" | "A2C"
        stage:            Fase, ex: "1-1"
        seed:             Semente
        total_timesteps:  Quantos timesteps treinar no total
        eval_freq:        Frequência de avaliação (em timesteps por env, não global)
        n_eval_episodes:  Episódios por avaliação
        n_checkpoints:    Quantos checkpoints uniformes durante o treino (0 desativa)
        overwrite:        Se True, ignora estado prévio e retreina do zero
        verbose:          0=silencioso, 1=normal
        reward_shaping:   Se True, adiciona ProgressRewardWrapper + sufixo "_shape"
                          ao exp_id (artefatos separados do baseline).

    Returns:
        Path para o CSV de métricas de avaliação.
    """
    ensure_dirs()

    exp_id = experiment_id(algo, stage, seed)
    if reward_shaping:
        exp_id = f"{exp_id}_shape"   # diferencia do baseline para não sobrescrever

    log_csv  = LOGS_DIR / f"{exp_id}.csv"
    model_pt = MODELS_DIR / f"{exp_id}.zip"
    tb_path  = TB_DIR / exp_id

    # 1) Treino completo já feito?
    if log_csv.exists() and model_pt.exists() and not overwrite:
        if verbose:
            print(f"  → {exp_id} já completo, pulando.")
        return log_csv

    device = get_device()
    set_global_seed(seed)
    hp = HPARAMS[algo].copy()
    n_envs = hp.pop("n_envs")

    # 2) Cria o env de treino
    train_env = make_vec_env_mario(
        stage=stage, n_envs=n_envs, seed=seed,
        use_subproc=(n_envs > 1),
        reward_shaping=reward_shaping,
    )

    cls = _ALGO_CLS[algo]
    ckpt_path, ckpt_steps = _find_latest_checkpoint(exp_id)

    # 3) Lógica de RESUME — se há checkpoint parcial e não fizemos overwrite
    if ckpt_path is not None and not overwrite and ckpt_steps < total_timesteps:
        if verbose:
            print(f"  ↻ Retomando {exp_id} do checkpoint: {ckpt_path.name} "
                  f"({ckpt_steps:,} steps já feitos)")
        model = cls.load(
            str(ckpt_path),
            env=train_env,
            device=device,
            tensorboard_log=str(tb_path),
        )
        remaining = total_timesteps - ckpt_steps
        reset_num_timesteps = False   # mantém contagem do TensorBoard contínua
    else:
        common_kwargs = dict(
            policy="CnnPolicy",
            env=train_env,
            verbose=0,
            seed=seed,
            device=device,
            tensorboard_log=str(tb_path),
        )
        model = cls(**common_kwargs, **hp)
        remaining = total_timesteps
        reset_num_timesteps = True

    # 4) Callbacks: avaliação + checkpoint periódico
    eval_cb = MarioEvalCallback(
        eval_stage=stage,
        eval_freq=max(eval_freq // n_envs, 1),   # ajusta p/ n_envs
        n_eval_episodes=n_eval_episodes,
        log_path=log_csv,
        seed=seed,
        verbose=verbose,
    )
    callbacks: list[BaseCallback] = [eval_cb]

    if n_checkpoints > 0:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        # save_freq é em "chamadas a _on_step" (= passos de policy);
        # multiplicar por n_envs converte para timesteps globais.
        ckpt_save_freq = max(total_timesteps // n_envs // n_checkpoints, 1)
        callbacks.append(CheckpointCallback(
            save_freq=ckpt_save_freq,
            save_path=str(CKPT_DIR),
            name_prefix=exp_id,
            save_replay_buffer=False,
            save_vecnormalize=False,
        ))

    cb = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    # 5) Treinamento
    t0 = time.time()
    if verbose:
        print(f"\n► [{exp_id}] treinando ({remaining:,} timesteps restantes, n_envs={n_envs})")
    try:
        model.learn(
            total_timesteps=remaining,
            callback=cb,
            tb_log_name=exp_id,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(model_pt)
        if verbose:
            elapsed = time.time() - t0
            print(f"✓ [{exp_id}] concluído em {elapsed/60:.1f} min — modelo: {model_pt}")
    finally:
        train_env.close()
    return log_csv


def run_matrix(
    algos: list[str],
    stages_to_run: list[str],
    seeds_to_run: list[int],
    *,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    n_checkpoints: int = 5,
    overwrite: bool = False,
    reward_shaping: bool = False,
) -> list[tuple]:
    """
    Loop sobre algos × stages × seeds. Idempotente.

    Returns:
        Lista de (algo, stage, seed, csv_path_ou_None).
    """
    total = len(algos) * len(stages_to_run) * len(seeds_to_run)
    done = 0
    results = []

    shape_tag = "  [shape]" if reward_shaping else ""
    for algo in algos:
        for stage in stages_to_run:
            for seed in seeds_to_run:
                done += 1
                print(f"\n{'='*70}")
                print(f"[{done}/{total}] {algo} | stage {stage} | seed {seed}{shape_tag}")
                print('='*70)
                try:
                    csv_path = train_one(
                        algo=algo, stage=stage, seed=seed,
                        total_timesteps=total_timesteps,
                        eval_freq=eval_freq,
                        n_eval_episodes=n_eval_episodes,
                        n_checkpoints=n_checkpoints,
                        overwrite=overwrite,
                        reward_shaping=reward_shaping,
                    )
                    results.append((algo, stage, seed, csv_path))
                except Exception as e:
                    print(f"✗ FALHOU [{algo}/{stage}/{seed}]: {e}")
                    results.append((algo, stage, seed, None))

    return results
