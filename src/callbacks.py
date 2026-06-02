"""
Callbacks customizados para SB3.

`MarioEvalCallback` é o coração da coleta de dados — avalia o agente
periodicamente, captura métricas por episódio (reward, max_x_pos,
flag_get, deaths, frames, time_left) e persiste em CSV de forma incremental.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from .env import make_vec_env_mario


class MarioEvalCallback(BaseCallback):
    """
    Avalia o agente periodicamente e registra métricas detalhadas em CSV.

    Cada chamada de avaliação roda `n_eval_episodes` episódios determinísticos
    (`deterministic=True` no predict) e grava uma linha por episódio com:
      timestep, episode, reward, max_x_pos, flag_get, deaths, frames, time_left
    """

    def __init__(
        self,
        eval_stage: str,
        eval_freq: int,
        n_eval_episodes: int,
        log_path: Path,
        seed: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_stage = eval_stage
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = Path(log_path)
        self.seed = seed
        self.eval_env = None
        self._rows = []
        self._eval_count = 0

    def _on_training_start(self) -> None:
        # Env de avaliação separado, sempre n_envs=1, determinístico
        # Seed deslocada (+9999) p/ descorrelacionar do env de treino.
        self.eval_env = make_vec_env_mario(
            stage=self.eval_stage,
            n_envs=1,
            seed=self.seed + 9999,
            use_subproc=False,
        )

    def _run_eval(self):
        """Executa n_eval_episodes e devolve uma lista de dicts com métricas."""
        results = []
        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = [False]
            ep_reward = 0.0
            max_x = 0
            deaths = 0
            frames = 0
            flag_get = False
            time_left = None
            prev_life = None
            last_info = {}

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                info0 = info[0]
                last_info = info0
                ep_reward += float(reward[0])
                frames += 1

                # Métricas do Mario (extraídas do info do nes-py)
                x = int(info0.get("x_pos", 0))
                if x > max_x:
                    max_x = x
                life = info0.get("life", None)
                # Morte detectada por perda de vida (queda em buraco, Bowser)
                if prev_life is not None and life is not None and life < prev_life:
                    deaths += 1
                prev_life = life
                if info0.get("flag_get", False):
                    flag_get = True
                time_left = info0.get("time", time_left)

            # Morte por colisão com Goomba pequeno: done=True sem decrementar `life`.
            # Se o episódio terminou sem flag E sem death registrado por life-loss,
            # considera como uma morte.
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
        return results

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        eval_results = self._run_eval()
        self._eval_count += 1

        for r in eval_results:
            self._rows.append({"timestep": int(self.num_timesteps), **r})

        if self.verbose >= 1:
            rewards = [r["reward"] for r in eval_results]
            flags = [r["flag_get"] for r in eval_results]
            xs = [r["max_x_pos"] for r in eval_results]
            print(
                f"  [eval @ {self.num_timesteps:>7,}] "
                f"reward={np.mean(rewards):+8.2f}±{np.std(rewards):5.2f}  "
                f"completion={np.mean(flags):.0%}  "
                f"max_x_avg={np.mean(xs):.0f}"
            )

        # Persistência incremental — sobrevive a crash do processo
        pd.DataFrame(self._rows).to_csv(self.log_path, index=False)
        return True

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
