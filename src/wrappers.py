"""
Wrappers customizados.

`ProgressRewardWrapper` adiciona um bonus de reward proporcional ao avanço de
`x_pos` no Super Mario Bros, e uma penalidade clara por morte. Resolve o
problema de convergência prematura para política trivial (Mario parado).

Justificativa: o reward nativo do `gym-super-mario-bros` é truncado em
[-15, +15] por frame e dominado por penalidades de tempo. Em horizontes de
desconto de gamma=0.99 (~100 steps), o agente "aprende" que ficar imóvel é
melhor do que tentar e ser punido. Adicionar um sinal contínuo proporcional
ao progresso espacial faz o gradiente "puxar" o agente para a direita.

Referências:
- Kauten (2018), gym-super-mario-bros docs
- Mnih et al. (2015), Atari RL — uso de reward clipping similar
"""
from __future__ import annotations
import gym


class ProgressRewardWrapper(gym.Wrapper):
    """
    Adiciona ao reward nativo um sinal proporcional ao avanço em x_pos.

    Composição do reward final:
        r_total = r_nativo
                + progress_coef * (x_pos_t - x_pos_{t-1})
                + completion_bonus * flag_get_event
                - death_penalty * death_event

    Aplicado ANTES do shimmy (no env gym antigo). A ordem importa: precisa
    estar antes de MaxAndSkipEnv porque o skip agrega 4 frames de reward,
    e queremos o shaping aplicado a cada frame.

    Args:
        env:               env gym antigo (JoypadSpace ou raw mario)
        progress_coef:     multiplicador do delta x_pos (default 0.1)
                           Com x avançando ~5-10 pixels por frame bom, isso
                           dá ~0.5-1.0 de bonus por frame de progresso real.
        death_penalty:     punição por perder vida (default 15, já é o que
                           o env faz, deixamos explícito)
        completion_bonus:  bonus extra por completar (default 50)
    """

    def __init__(
        self,
        env,
        progress_coef: float = 1.0,
        death_penalty: float = 25.0,
        completion_bonus: float = 200.0,
        time_penalty: float = 0.2,
        backward_penalty: float = 1.0,
        stuck_penalty: float = 0.5,
        stuck_window: int = 60,
    ):
        super().__init__(env)
        self.progress_coef = progress_coef
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus
        self.time_penalty = time_penalty       # custo por frame — incentiva fazer algo
        self.backward_penalty = backward_penalty   # custo por andar para a esquerda
        self.stuck_penalty = stuck_penalty
        self.stuck_window = stuck_window
        self._x_prev = 0
        self._life_prev = None
        self._x_max = 0
        self._frames_since_progress = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._x_prev = 0
        self._life_prev = None
        self._x_max = 0
        self._frames_since_progress = 0
        return out

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs, reward, done, info = ret
            terminated, truncated = done, False

        # Progresso em x — só conta avanço para NOVO recorde
        x_curr = int(info.get("x_pos", self._x_prev))
        if x_curr > self._x_max:
            delta_progress = x_curr - self._x_max
            self._x_max = x_curr
            self._frames_since_progress = 0
        else:
            delta_progress = 0
            self._frames_since_progress += 1
        progress_bonus = self.progress_coef * delta_progress

        # Movimento para trás (delta_x negativo)
        delta_x = x_curr - self._x_prev
        backward_pen = 0.0
        if delta_x < 0:
            backward_pen = self.backward_penalty * abs(delta_x)
        self._x_prev = x_curr

        # Penalidade por travar (sem progresso por stuck_window frames)
        stuck_pen = 0.0
        if self._frames_since_progress > self.stuck_window:
            stuck_pen = self.stuck_penalty

        # Morte
        life = info.get("life", None)
        death_event = 0.0
        if self._life_prev is not None and life is not None and life < self._life_prev:
            death_event = 1.0
        # Game over por colisão sem perda de "life" (Goomba pequeno)
        if done and not info.get("flag_get", False) and death_event == 0.0:
            death_event = 1.0
        self._life_prev = life

        # Conclusão
        completion_event = 1.0 if info.get("flag_get", False) else 0.0

        shaped_reward = (
            float(reward)
            + progress_bonus
            - self.time_penalty          # custo por frame, fixo (substitui survival_bonus)
            - backward_pen               # custo por andar pra esquerda
            - stuck_pen                  # custo por travar
            - self.death_penalty * death_event
            + self.completion_bonus * completion_event
        )

        if len(ret) == 5:
            return obs, shaped_reward, terminated, truncated, info
        return obs, shaped_reward, done, info