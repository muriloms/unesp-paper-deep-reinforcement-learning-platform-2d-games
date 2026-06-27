"""
Função de treinamento e loop completo.

`train_one()` é idempotente E retomável, com um MANIFESTO DE PROGRESSO
(`mario_drl_results/progress.json`) que registra, por experimento, até quantos
timesteps foi de fato treinado.

Por que o manifesto existe:
- Os checkpoints intermediários (CheckpointCallback) param em múltiplos de
  `save_freq`. Num resume curto (ex: 1.9M → 2M), o trecho final de 100k é
  pequeno demais para disparar um checkpoint em 2M, então o maior checkpoint
  fica em 1.9M. Sem o manifesto, o skip acharia 1.9M < 2M e retreinaria os
  100k finais para sempre.
- O manifesto grava o progresso REAL (timesteps concluídos) quando o treino
  termina, então o skip passa a confiar nele.
- É um único arquivo de texto (JSON), versionável no git. Você pode commitar
  `progress.json` e ignorar os `.zip` pesados — em outro PC dá pra ver o estado
  do projeto e saber de onde retomar.

Fonte de verdade para "quanto já treinei", em ordem de prioridade:
  1. progress.json  (registro autoritativo de conclusão)
  2. maior checkpoint salvo
  3. nada → começa do zero
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from .config import (
    HPARAMS, MODELS_DIR, CKPT_DIR, LOGS_DIR, TB_DIR, ROOT_DIR,
    experiment_id, ensure_dirs,
)
from .env import make_vec_env_mario
from .callbacks import MarioEvalCallback
from .utils import set_global_seed, get_device


# Mapeia string → classe SB3
_ALGO_CLS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}

# Manifesto de progresso (versionável no git)
PROGRESS_FILE = ROOT_DIR / "progress.json"


class ProgressManifestCallback(BaseCallback):
    """
    Atualiza o manifesto (progress.json) com o progresso REAL durante o treino,
    não só ao final. Garante que, se o treino for interrompido (Ctrl+C, queda,
    fechar o PC), o manifesto reflita até onde realmente chegou.

    Grava em intervalos (update_every timesteps) para não escrever em disco a
    cada passo. O valor gravado é `base_timesteps + num_timesteps`, onde
    base_timesteps é de onde o resume começou (mantém a contagem absoluta).
    """
    def __init__(self, exp_id: str, base_timesteps: int, meta: dict,
                 update_every: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.exp_id = exp_id
        self.base_timesteps = base_timesteps
        self.meta = meta
        self.update_every = update_every
        self._last_written = 0

    def _absolute_steps(self) -> int:
        # num_timesteps conta os passos desta chamada learn(); soma a base
        # para obter o total absoluto desde o início do experimento.
        return self.base_timesteps + self.num_timesteps

    def _on_step(self) -> bool:
        abs_steps = self._absolute_steps()
        if abs_steps - self._last_written >= self.update_every:
            update_progress(self.exp_id, timesteps=abs_steps, extra=self.meta)
            self._last_written = abs_steps
        return True


# ============================================================================
# MANIFESTO DE PROGRESSO
# ============================================================================
def load_progress() -> dict:
    """Lê progress.json. Retorna {} se não existir ou estiver corrompido."""
    if not PROGRESS_FILE.exists():
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_progress(exp_id: str) -> int:
    """Timesteps concluídos para um experimento, segundo o manifesto. 0 se ausente."""
    data = load_progress()
    entry = data.get(exp_id)
    if isinstance(entry, dict):
        return int(entry.get("timesteps", 0))
    if isinstance(entry, int):   # formato antigo simples
        return entry
    return 0


def update_progress(exp_id: str, timesteps: int, extra: dict | None = None) -> None:
    """Registra/atualiza o progresso de um experimento no manifesto."""
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_progress()
    entry = {"timesteps": int(timesteps), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        entry.update(extra)
    data[exp_id] = entry
    # Escrita atômica: grava em temp e renomeia (evita corromper se cair no meio)
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(PROGRESS_FILE)


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


def _resolve_progress(exp_id: str) -> tuple[int, Optional[Path], int]:
    """
    Determina o progresso real do experimento combinando manifesto + checkpoints.

    Returns:
        (effective_steps, ckpt_path, ckpt_steps)
          effective_steps : maior entre manifesto e último checkpoint
          ckpt_path/steps : info do checkpoint (p/ retomar pesos se necessário)
    """
    manifest_steps = get_progress(exp_id)
    ckpt_path, ckpt_steps = _find_latest_checkpoint(exp_id)
    effective_steps = max(manifest_steps, ckpt_steps)
    return effective_steps, ckpt_path, ckpt_steps


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

    # Progresso real: combina manifesto (autoritativo) + checkpoints
    effective_steps, ckpt_path, ckpt_steps = _resolve_progress(exp_id)

    # 1) Já atingiu o alvo? Usa o manifesto como fonte de verdade.
    #    Resolve o loop dos "100k finais": mesmo que o maior checkpoint seja
    #    1.9M, se o manifesto registra 2M concluído, pula de verdade.
    if not overwrite and effective_steps >= total_timesteps and model_pt.exists():
        if verbose:
            print(f"  → {exp_id} já completo ({effective_steps:,} ≥ "
                  f"{total_timesteps:,}), pulando.")
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

    # 3) RESUME — escolhe a melhor fonte de pesos disponível.
    #    Prioriza checkpoint (tem pesos + optimizer state). Se o manifesto diz
    #    que há progresso mas o checkpoint está atrás, usa o modelo final.
    resume_source = None
    resume_steps = 0
    if not overwrite and effective_steps > 0:
        if ckpt_path is not None and ckpt_steps >= effective_steps:
            resume_source, resume_steps = ckpt_path, ckpt_steps
        elif model_pt.exists():
            # Modelo final está à frente (ou igual) ao checkpoint → retoma dele.
            resume_source, resume_steps = model_pt, effective_steps
        elif ckpt_path is not None:
            resume_source, resume_steps = ckpt_path, ckpt_steps

    if resume_source is not None and resume_steps < total_timesteps:
        if verbose:
            print(f"  ↻ Retomando {exp_id} de: {resume_source.name} "
                  f"({resume_steps:,} steps já feitos → alvo {total_timesteps:,})")
        model = cls.load(
            str(resume_source),
            env=train_env,
            device=device,
            tensorboard_log=str(tb_path),
        )
        remaining = total_timesteps - resume_steps
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
        resume_steps = 0

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

    # Grava progresso REAL no manifesto durante o treino (não só ao final).
    # Se o treino for interrompido, o progress.json reflete até onde chegou —
    # essencial p/ o fluxo entre PCs (interromper, subir no git, retomar).
    progress_meta = {
        "algo": algo, "stage": stage, "seed": seed,
        "reward_shaping": bool(reward_shaping),
        "model": model_pt.name,
        "source": "live",
    }
    callbacks.append(ProgressManifestCallback(
        exp_id=exp_id,
        base_timesteps=resume_steps,   # de onde o resume começou (0 se do zero)
        meta=progress_meta,
        update_every=max(eval_freq, 10_000),  # alinha com a freq de avaliação
        verbose=0,
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
        # Registra progresso REAL no manifesto — fonte de verdade para o skip.
        update_progress(
            exp_id,
            timesteps=total_timesteps,
            extra={
                "algo": algo, "stage": stage, "seed": seed,
                "reward_shaping": bool(reward_shaping),
                "model": model_pt.name,
            },
        )
        if verbose:
            elapsed = time.time() - t0
            print(f"✓ [{exp_id}] concluído em {elapsed/60:.1f} min — modelo: {model_pt}")
            print(f"  progresso registrado em {PROGRESS_FILE.name}: {total_timesteps:,} steps")
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
