"""
Environment factory para o Super Mario Bros.

Pipeline de wrappers Atari clássica + ponte gym → gymnasium via shimmy.
Cuidado com a ordem — alterar wrappers quebra `info` (perde flag_get, x_pos, life,
etc.) e as métricas do Grupo II ficam vazias.
"""
from __future__ import annotations

import gym as _legacy_gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor,
)

from .config import FRAME_STACK


# ----------------------------------------------------------------------
# Auto-detect gym version → escolhe shim correto
# ----------------------------------------------------------------------
_gym_version = tuple(int(x) for x in _legacy_gym.__version__.split(".")[:2])
if _gym_version >= (0, 26):
    from shimmy import GymV26CompatibilityV0 as _GymCompat
    _COMPAT_API = "v0.26"
else:
    from shimmy import GymV21CompatibilityV0 as _GymCompat
    _COMPAT_API = "v0.21"


class CompatJoypadSpace(JoypadSpace):
    """
    JoypadSpace tolerante a `seed` e `options` no reset().

    O nes-py 8.2.1 não conhece esses kwargs (introduzidos em gym>=0.22),
    e quando SB3/shimmy os encaminham via reset(), dá TypeError. Esta
    subclasse engole esses kwargs e delega ao parent.
    """
    def reset(self, seed=None, options=None, **kwargs):
        return super().reset(**kwargs)


def make_mario_env(
    stage: str = "1-1",
    seed: int = 0,
    render_mode: str | None = None,
    reward_shaping: bool = False,
):
    """
    Cria UM env do Super Mario Bros já com a pipeline de pré-processamento.

    Args:
        stage:           Fase no formato "world-stage", ex: "1-1", "4-2"
        seed:            Semente para reprodutibilidade
        render_mode:     None para treino; "rgb_array" para captura de frames
        reward_shaping:  Se True, adiciona ProgressRewardWrapper (bonus por x_pos)
                         Resolve convergência prematura para política trivial.

    Returns:
        Env gymnasium pronto para SB3 (SEM frame-stack — esse é VecFrameStack).
    """
    env_id = f"SuperMarioBros-{stage}-v0"

    # 1) Env do Mario com API compatibility ativada
    make_kwargs = dict(apply_api_compatibility=True)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    env = gym_super_mario_bros.make(env_id, **make_kwargs)

    # 2) Restringe ação a SIMPLE_MOVEMENT (7 ações), com fix para seed/options
    env = CompatJoypadSpace(env, SIMPLE_MOVEMENT)

    # 2.5) [OPCIONAL] Reward shaping ANTES da ponte/skip
    if reward_shaping:
        from .wrappers import ProgressRewardWrapper
        env = ProgressRewardWrapper(env)

    # 3) Ponte gym → gymnasium
    env = _GymCompat(env=env)

    # 4) Skip-frame com max pixel-wise (protocolo Atari de Mnih et al., 2015)
    env = MaxAndSkipEnv(env, skip=4)

    # 5) Grayscale + resize 84x84
    env = WarpFrame(env, width=84, height=84)

    # 6) Monitor para logging de episódios
    env = Monitor(env)

    env.action_space.seed(seed)
    return env


def make_vec_env_mario(
    stage: str,
    n_envs: int,
    seed: int,
    use_subproc: bool = True,
    reward_shaping: bool = False,
):
    """
    Cria um VecEnv com n_envs cópias paralelas do Mario + frame stack.

    - n_envs == 1   → DummyVecEnv (mais leve, sem overhead de IPC)
    - n_envs > 1    → SubprocVecEnv (processos separados, paralelismo real)
    - reward_shaping → propaga para todos os envs paralelos
    """
    def make_one(rank: int):
        def _init():
            return make_mario_env(
                stage=stage, seed=seed + rank,
                reward_shaping=reward_shaping,
            )
        return _init

    env_fns = [make_one(i) for i in range(n_envs)]
    if n_envs == 1 or not use_subproc:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")

    # Frame-stack 4 no nível VecEnv (importante: depois do VecEnv, não antes)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    vec_env = VecMonitor(vec_env)
    return vec_env
