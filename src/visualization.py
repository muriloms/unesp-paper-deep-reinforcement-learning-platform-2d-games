"""
Visualizações: GIFs do agente jogando.

- render_agent_episode      : 1 modelo, 1 fase → 1 GIF
- render_models_side_by_side: N modelos lado-a-lado (mesma fase, mesma seed)
- render_temporal_evolution : mesmo algoritmo em diferentes checkpoints

Cuidados técnicos importantes (custaram muito tempo descobrir):
  - SEMPRE fazer frame.copy() — env.render() retorna referência ao buffer interno
    do emulador; sem copy o GIF fica estático (todos os frames apontam ao mesmo
    array, sobrescrito).
  - imageio.mimsave(...) com duration em MILISSEGUNDOS, não segundos.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from .config import (
    FRAME_STACK, MODELS_DIR, CKPT_DIR, PLOTS_DIR,
    experiment_id,
)
from .env import _GymCompat, CompatJoypadSpace
from .utils import get_device

_ALGO_CLS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


# ============================================================================
# Env de render — pipeline igual ao de treino mas com render_mode="rgb_array"
# ============================================================================
def _make_render_env(stage: str):
    env_id = f"SuperMarioBros-{stage}-v0"
    base = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")
    base = CompatJoypadSpace(base, SIMPLE_MOVEMENT)
    base = _GymCompat(env=base)
    base = MaxAndSkipEnv(base, skip=4)
    base = WarpFrame(base, width=84, height=84)
    base = Monitor(base)
    vec_env = DummyVecEnv([lambda: base])
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    return vec_env


# ============================================================================
# Helper: roda 1 episódio capturando frames + métricas
# ============================================================================
def run_episode_for_render(model, stage: str = "1-1", max_steps: int = 3000, seed: int = 999):
    """
    Roda 1 episódio do modelo e devolve (frames RGB, metrics dict).

    O frame.copy() é CRÍTICO — sem ele, todos os frames acabam apontando
    para o mesmo buffer de memória do emulador (que é sobrescrito a cada step),
    e o GIF fica estático.
    """
    vec_env = _make_render_env(stage)
    render_env = vec_env.venv.envs[0]
    # seed do action_space (estabilidade reprodutível das ações sample)
    render_env.action_space.seed(seed)

    frames = []
    metrics = {"reward": 0.0, "max_x": 0, "deaths": 0,
               "flag_get": False, "steps": 0}
    prev_life = None

    obs = vec_env.reset()
    for _ in range(max_steps):
        frame = render_env.render()
        if frame is not None:
            frames.append(frame.copy())   # ← CRÍTICO

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        info0 = info[0]
        metrics["reward"] += float(reward[0])
        metrics["steps"] += 1
        x = int(info0.get("x_pos", 0))
        if x > metrics["max_x"]:
            metrics["max_x"] = x
        life = info0.get("life", None)
        if prev_life is not None and life is not None and life < prev_life:
            metrics["deaths"] += 1
        prev_life = life
        if info0.get("flag_get", False):
            metrics["flag_get"] = True

        if done[0]:
            frame = render_env.render()
            if frame is not None:
                frames.append(frame.copy())
            break

    vec_env.close()
    return frames, metrics


def _model_suffix(variant: Optional[str]) -> str:
    """Sufixo do nome do modelo conforme variant (None → vanilla, 'shape' → _shape)."""
    return f"_{variant}" if variant else ""


# ============================================================================
# 1. Render de UM agente
# ============================================================================
def render_agent_episode(
    algo: str, stage: str, seed: int,
    *,
    max_steps: int = 2000,
    fps: int = 15,
    out_path: Optional[Path] = None,
    render_seed: int = 999,
    variant: Optional[str] = None,
) -> tuple[Path, dict]:
    """Carrega modelo treinado e salva GIF de 1 episódio.

    Args:
        variant: None p/ baseline; "shape" p/ modelos com reward shaping;
                 outras strings p/ futuras variantes (sufixo do nome do modelo).
    """
    cls = _ALGO_CLS[algo]
    suffix = _model_suffix(variant)
    model_path = MODELS_DIR / f"{experiment_id(algo, stage, seed)}{suffix}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = cls.load(model_path, device=get_device())
    frames, metrics = run_episode_for_render(
        model, stage=stage, max_steps=max_steps, seed=render_seed,
    )

    if out_path is None:
        out_path = PLOTS_DIR / f"agent_{algo}_stage{stage}_seed{seed}{suffix}.gif"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANTE: duration em MS (imageio v2 + PIL)
    duration_ms = int(round(1000.0 / fps))
    imageio.mimsave(out_path, frames, duration=duration_ms, loop=0)

    print(f"✓ {len(frames)} frames @ {fps} FPS → {out_path}")
    print(f"   reward={metrics['reward']:+.1f}  max_x={metrics['max_x']}  "
          f"flag={metrics['flag_get']}  deaths={metrics['deaths']}")
    return out_path, metrics


# ============================================================================
# 2. Render de N modelos lado-a-lado
# ============================================================================
def _add_label_bar(frame, label_top: str, metrics_text: str = ""):
    """Adiciona barra preta no topo com label, e opcional barra inferior com métrica."""
    import cv2   # import local — evita exigir opencv quando só se treina
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, label_top, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if metrics_text:
        cv2.rectangle(out, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(out, metrics_text, (8, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_models_side_by_side(
    models: dict,
    *,
    stage: str = "1-1",
    max_steps: int = 3000,
    fps: int = 15,
    render_seed: int = 999,
    out_path: Optional[Path] = None,
    show_running_metrics: bool = True,
) -> tuple[Path, dict]:
    """
    Renderiza múltiplos modelos jogando a mesma fase com o mesmo seed,
    compõe lado-a-lado num único GIF.

    Args:
        models: dict {label: model_obj}
                ex: {"DQN": dqn_model, "PPO": ppo_model, "A2C": a2c_model}
                ex: {"100k": m100k, "300k": m300k, "500k": m500k}
    """
    labels = list(models.keys())
    print(f"Renderizando {len(labels)} modelos: {labels}  |  stage {stage}")

    all_frames = {}
    all_metrics = {}
    for label, model in models.items():
        print(f"  ► {label}...")
        frames, metrics = run_episode_for_render(model, stage, max_steps, render_seed)
        all_frames[label] = frames
        all_metrics[label] = metrics
        flag = "✓" if metrics["flag_get"] else "✗"
        print(f"    {len(frames):4d} frames | r={metrics['reward']:+7.1f} "
              f"| max_x={metrics['max_x']:5d} | flag={flag} | deaths={metrics['deaths']}")

    # Padding p/ sincronizar tamanhos (alguns morrem antes)
    max_len = max(len(f) for f in all_frames.values())
    for label in labels:
        frames = all_frames[label]
        if len(frames) < max_len:
            pad = frames[-1] if frames else np.zeros((240, 256, 3), dtype=np.uint8)
            frames.extend([pad.copy() for _ in range(max_len - len(frames))])

    # Composição lado-a-lado
    composed = []
    for t in range(max_len):
        panels = []
        for label in labels:
            text = ""
            if show_running_metrics:
                final = all_metrics[label]
                progress = min((t + 1) / max(final["steps"], 1), 1.0)
                approx_x = int(final["max_x"] * progress)
                approx_r = final["reward"] * progress
                text = f"x={approx_x}  r={approx_r:+.0f}"
            panels.append(_add_label_bar(all_frames[label][t], label, text))
        composed.append(np.concatenate(panels, axis=1))

    if out_path is None:
        out_path = PLOTS_DIR / f"comparison_{stage}.gif"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(round(1000.0 / fps))
    imageio.mimsave(out_path, composed, duration=duration_ms, loop=0)
    print(f"\n✓ Salvo: {out_path}  ({len(composed)} frames, "
          f"{composed[0].shape[1]}×{composed[0].shape[0]} px)")
    return out_path, all_metrics


# ============================================================================
# 3. Comparação DQN vs PPO vs A2C (helper de alto nível)
# ============================================================================
def render_algos_comparison(
    stage: str, seed: int,
    *,
    algos: list[str] | None = None,
    max_steps: int = 3000,
    fps: int = 15,
    render_seed: int = 999,
    out_path: Optional[Path] = None,
    variant: Optional[str] = None,
) -> Optional[Path]:
    """Carrega DQN/PPO/A2C (do mesmo stage/seed/variant) e renderiza lado-a-lado."""
    algos = algos or ["DQN", "PPO", "A2C"]
    suffix = _model_suffix(variant)
    models = {}
    for algo in algos:
        cls = _ALGO_CLS[algo]
        path = MODELS_DIR / f"{experiment_id(algo, stage, seed)}{suffix}.zip"
        if path.exists():
            models[algo] = cls.load(path, device=get_device())
            print(f"  ✓ {algo}: {path.name}")
        else:
            print(f"  ✗ {algo} não encontrado: {path.name}")

    if len(models) < 2:
        print(f"⚠ Pelo menos 2 algoritmos necessários, encontrados {len(models)}.")
        return None

    if out_path is None:
        out_path = PLOTS_DIR / f"comparison_algos_stage{stage}_seed{seed}{suffix}.gif"
    path, _ = render_models_side_by_side(
        models, stage=stage, max_steps=max_steps, fps=fps,
        render_seed=render_seed, out_path=out_path,
    )
    return path


# ============================================================================
# 4. Evolução temporal — mesmo algoritmo em diferentes checkpoints
# ============================================================================
def _find_checkpoints(
    algo: str, stage: str, seed: int, variant: Optional[str] = None
) -> list[tuple[int, Path]]:
    """Lista os (timesteps, path) de todos os checkpoints de um exp_id, ordenados."""
    exp_id = experiment_id(algo, stage, seed)
    if variant:
        exp_id = f"{exp_id}_{variant}"
    if not CKPT_DIR.exists():
        return []
    pattern = re.compile(rf"^{re.escape(exp_id)}_(\d+)_steps\.zip$")
    ckpts = []
    for p in sorted(CKPT_DIR.glob(f"{exp_id}_*.zip")):
        m = pattern.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    return ckpts


def render_temporal_evolution(
    algo: str, stage: str, seed: int,
    *,
    n_checkpoints: int = 5,
    max_steps: int = 2500,
    fps: int = 15,
    render_seed: int = 999,
    out_path: Optional[Path] = None,
    variant: Optional[str] = None,
) -> Optional[Path]:
    """Renderiza o mesmo algoritmo em vários checkpoints lado-a-lado."""
    ckpts = _find_checkpoints(algo, stage, seed, variant=variant)
    if not ckpts:
        print(f"⚠ Nenhum checkpoint encontrado em {CKPT_DIR}")
        return None

    # Escolhe checkpoints uniformemente espaçados
    n_show = min(n_checkpoints, len(ckpts))
    step = max(len(ckpts) // n_show, 1)
    chosen = ckpts[::step][:n_show]

    cls = _ALGO_CLS[algo]
    models = {f"{s // 1000}k": cls.load(p, device=get_device()) for s, p in chosen}
    print(f"Checkpoints: {[s for s, _ in chosen]}")

    suffix = _model_suffix(variant)
    if out_path is None:
        out_path = PLOTS_DIR / f"evolution_{algo}_stage{stage}_seed{seed}{suffix}.gif"
    path, _ = render_models_side_by_side(
        models, stage=stage, max_steps=max_steps, fps=fps,
        render_seed=render_seed, out_path=out_path,
    )
    return path
