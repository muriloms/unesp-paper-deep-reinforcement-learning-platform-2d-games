"""
Configuração global do projeto Mario DRL.

Todas as constantes do experimento (fases, seeds, hyperparams, paths) em um único módulo.
Toda mudança aqui se propaga pelos scripts/notebooks.
"""
from __future__ import annotations
from pathlib import Path

# ============================================================================
# DIRETÓRIOS
# ============================================================================
# Por padrão, mario_drl_results/ na raiz do projeto (parent de src/).
# Pode ser sobrescrito via MARIO_DRL_ROOT no ambiente.
import os as _os

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "mario_drl_results"
ROOT_DIR    = Path(_os.environ.get("MARIO_DRL_ROOT", _DEFAULT_ROOT)).resolve()

MODELS_DIR  = ROOT_DIR / "models"
CKPT_DIR    = MODELS_DIR / "checkpoints"
LOGS_DIR    = ROOT_DIR / "logs"
TB_DIR      = ROOT_DIR / "tensorboard"
METRICS_DIR = ROOT_DIR / "metrics"
PLOTS_DIR   = ROOT_DIR / "plots"


def ensure_dirs():
    """Cria todos os diretórios de saída (idempotente)."""
    for d in (MODELS_DIR, CKPT_DIR, LOGS_DIR, TB_DIR, METRICS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EXPERIMENTO
# ============================================================================
STAGES = ["1-1", "1-2", "4-1", "8-1"]   # ordem canônica de dificuldade
SEEDS  = [42, 123, 2024]
ALGOS  = ["DQN", "PPO", "A2C"]

# Comprimentos canônicos das fases (em unidades de x_pos do emulador NES)
# Usado para normalizar a distância na métrica d_bar do Grupo II.
STAGE_LENGTH = {"1-1": 3266, "1-2": 3266, "4-1": 3866, "8-1": 3266}

# Frame-stack (parte do VecEnv, não do env base)
FRAME_STACK = 4

# ----------------------------------------------------------------------
# REWARD SHAPING — adiciona bonus por progresso (x_pos) e penalidade
# por morte. Resolve convergência prematura do PPO/A2C para política
# trivial (Mario parado). Default: False (mantém protocolo original).
# Para usar: passe --reward-shaping no scripts/train.py
# ----------------------------------------------------------------------
USE_REWARD_SHAPING = False


# ============================================================================
# MODOS DE EXECUÇÃO
# ============================================================================
# "smoke"    → ~2 min, valida pipeline (pouco timesteps p/ ver aprendizado)
# "mid"      → ~10-20 min, validação de aprendizado real (DQN começa a aprender)
# "long"     → ~30-60 min, progresso visível antes do full
# "full"     → ~60h, experimento completo do paper (500k × todas fases/seeds)
# "scaling"  → 2M em 2 fases (1-1, 4-1), experimento de escala de orçamento
# "full-2mi" → 2M em TODAS as fases/seeds (36 treinos)
# "full-4mi" → 4M em TODAS as fases/seeds — retoma do fim do full-2mi via manifesto

PROFILES = {
    "smoke": dict(
        total_timesteps = 10_000,
        eval_freq       = 2_000,
        n_eval_episodes = 2,
        stages_to_run   = ["1-1"],
        seeds_to_run    = [42],
        n_checkpoints   = 0,   # smoke não gera checkpoints intermediários
    ),
    "mid": dict(
        # ~10-20 min por algo (1 fase, 1 seed) — suficiente para PPO/A2C
        # mostrarem aprendizado real e DQN começar a treinar (learning_starts=10k).
        # Use p/ validar reward_shaping antes de disparar o full.
        total_timesteps = 100_000,
        eval_freq       = 10_000,
        n_eval_episodes = 3,
        stages_to_run   = ["1-1"],
        seeds_to_run    = [42],
        n_checkpoints   = 3,
    ),
    "long": dict(
        # ~30-60 min por algo (1 fase, 1 seed) — esperado para PPO/A2C
        # já mostrarem progresso visível em Mario (passar do 1º Goomba e tal).
        # Boa zona de validação antes do full.
        total_timesteps = 250_000,
        eval_freq       = 10_000,
        n_eval_episodes = 5,
        stages_to_run   = ["1-1"],
        seeds_to_run    = [42],
        n_checkpoints   = 5,
    ),
    "full": dict(
        total_timesteps = 500_000,
        eval_freq       = 10_000,
        n_eval_episodes = 5,
        stages_to_run   = STAGES,
        seeds_to_run    = SEEDS,
        n_checkpoints   = 5,
    ),
    "scaling": dict(
        # Experimento de escala de orçamento: testa se MAIS timesteps resolvem
        # a não-conclusão observada em 500k. Foca em 2 fases (1 fácil + 1 difícil)
        # × 3 algos × 3 seeds = 18 treinos de 2M timesteps.
        #
        # APROVEITA OS CHECKPOINTS DE 500k: se já existe checkpoint de 500k de um
        # treino, o resume continua dali e treina só os 1.5M restantes.
        #
        # Tempo estimado (RTX 4070, retomando de 500k):
        #   DQN (1 env):  ~6-9h × 6 treinos  = ~45h
        #   PPO (8 envs): ~3-5h × 6 treinos  = ~25h
        #   A2C (16 envs):~3-5h × 6 treinos  = ~25h
        # Fatie por algoritmo em terminais paralelos.
        total_timesteps = 2_000_000,
        eval_freq       = 20_000,    # menos avaliações (eval custa tempo); 100 pontos
        n_eval_episodes = 5,
        stages_to_run   = ["1-1", "4-1"],   # 1 fácil + 1 difícil
        seeds_to_run    = SEEDS,
        n_checkpoints   = 10,        # checkpoint a cada 200k (resume granular)
    ),
    "full-2mi": dict(
        # 2M timesteps em TODAS as fases × seeds (36 treinos).
        # Retoma do fim do "full" (500k) via manifesto/modelo final.
        total_timesteps = 2_000_000,
        eval_freq       = 20_000,
        n_eval_episodes = 5,
        stages_to_run   = STAGES,
        seeds_to_run    = SEEDS,
        n_checkpoints   = 10,        # checkpoint a cada 200k
    ),
    "full-4mi": dict(
        # 4M timesteps em TODAS as fases × seeds (36 treinos).
        # Retoma do fim do "full-2mi" (2M) via manifesto/modelo final —
        # treina só os 2M restantes de cada treino que já chegou a 2M.
        total_timesteps = 4_000_000,
        eval_freq       = 20_000,
        n_eval_episodes = 5,
        stages_to_run   = STAGES,
        seeds_to_run    = SEEDS,
        n_checkpoints   = 20,        # checkpoint a cada 200k (mantém granularidade)
    ),
}


# ============================================================================
# HIPERPARÂMETROS (Tabela 2 do artigo)
# ============================================================================
HPARAMS = {
    "DQN": dict(
        learning_rate     = 1e-4,
        buffer_size       = 100_000,
        learning_starts   = 10_000,
        batch_size        = 32,
        tau               = 1.0,
        gamma             = 0.99,
        train_freq        = 4,
        gradient_steps    = 1,
        target_update_interval = 10_000,
        exploration_fraction   = 0.10,
        exploration_initial_eps = 1.0,
        exploration_final_eps   = 0.01,
        max_grad_norm     = 10.0,
        n_envs            = 1,
    ),
    "PPO": dict(
        learning_rate     = 3e-4,    # aumentado de 2.5e-4
        n_steps           = 128,
        batch_size        = 256,
        n_epochs          = 4,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.1,
        ent_coef          = 0.1,     # exploração FORTE (era 0.05 → 0.01 original)
        vf_coef           = 0.5,
        max_grad_norm     = 0.5,
        n_envs            = 8,
    ),
    "A2C": dict(
        learning_rate     = 7e-4,
        n_steps           = 5,
        gamma             = 0.99,
        gae_lambda        = 1.0,
        ent_coef          = 0.01,
        vf_coef           = 0.25,
        max_grad_norm     = 0.5,
        rms_prop_eps      = 1e-5,
        use_rms_prop      = True,
        n_envs            = 16,
    ),
}


# ============================================================================
# HELPERS
# ============================================================================
def experiment_id(algo: str, stage: str, seed: int) -> str:
    """ID único de um treino — usado para todos os artefatos (modelo, logs, TB)."""
    return f"{algo}_stage{stage}_seed{seed}"


def get_profile(name: str) -> dict:
    """Retorna o dict do perfil de execução (smoke/full)."""
    if name not in PROFILES:
        raise ValueError(f"Perfil desconhecido: {name!r}. Opções: {list(PROFILES)}")
    return PROFILES[name]