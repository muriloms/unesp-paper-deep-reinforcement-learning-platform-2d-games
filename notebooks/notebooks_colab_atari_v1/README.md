# Atari DRL Playtester — Pacote de Implementação

Pacote de execução para o projeto científico **"Avaliação Automática de Dificuldade
em Jogos Atari 2600 por meio de Agentes de Deep Reinforcement Learning"**
(disciplina de Redes Neurais Artificiais — PPGCC/UNESP).

Treina **DQN, PPO e A2C** em 4 jogos Atari de dificuldade canônica crescente
× 3 seeds = **36 runs** de 500 000 timesteps cada, e analisa as métricas
agregadas via Spearman + Mann-Whitney U para validar a hipótese central:
**as métricas dos agentes refletem a dificuldade canônica reconhecida na literatura.**

---

## Os 4 jogos

| # | Jogo | env_id | Dificuldade | Referência |
|---|------|--------|-------------|-----------|
| 1 | Pong | `ALE/Pong-v5` | Fácil | Mnih et al. 2015 (resolvido ~100k ts) |
| 2 | Breakout | `ALE/Breakout-v5` | Médio | Mnih et al. 2015 (~500k–1M ts) |
| 3 | Ms.Pac-Man | `ALE/MsPacman-v5` | Difícil | Bellemare et al. 2016 |
| 4 | Montezuma's Revenge | `ALE/MontezumaRevenge-v5` | Extremo (sparse reward) | Badia et al. 2020 |

Pong/Breakout têm reward denso; Ms.Pac-Man tem espaço de estados maior;
Montezuma's Revenge é o *poster child* de *hard exploration* e raramente é
resolvido por DQN/PPO/A2C baseline — um **negative result informativo** que
fortalece a hipótese H1.

---

## Arquivos do pacote

| Arquivo | Função | Células |
|---|---|---|
| `01_train_dqn.ipynb` | Treina DQN nos 4 jogos × 3 seeds | 22 |
| `02_train_ppo.ipynb` | Treina PPO nos 4 jogos × 3 seeds | 22 |
| `03_train_a2c.ipynb` | Treina A2C nos 4 jogos × 3 seeds | 22 |
| `04_analysis.ipynb` | Consolida métricas, gera tabelas/figuras/GIFs | 36 |
| `requirements.txt` | Versões pinadas da stack | — |

Os 3 notebooks de treino compartilham estrutura idêntica (só mudam o título
e os hiperparâmetros). Produzem em `BASE_DIR/`: modelos finais, checkpoints
intermediários, logs TensorBoard e CSVs de avaliação. O notebook de análise
consome esses CSVs e gera os outputs do paper.

---

## Stack

Migrou de stack legada (gym 0.26.2 + gym-super-mario-bros 7.4.0 + nes-py +
shimmy + condacolab + Python 3.10 forçado) para **stack gymnasium nativa**:

- `gymnasium ≥ 1.0` — API moderna oficial
- `ale-py ≥ 0.10` — Atari Learning Environment, **ROMs já incluídas no pacote**
- `stable-baselines3 ≥ 2.4` — DQN/PPO/A2C com `CnnPolicy`
- `torch ≥ 2.4`

**Zero hacks**: sem condacolab, sem downgrade de Python, sem shimmy, sem
nes-py, sem AutoROM, sem ROMs a importar manualmente. Funciona em Colab Pro
Python 3.11+ nativo.

---

## Roteiro de execução em Colab Pro

### 1. Setup inicial (uma vez)

1. Upload dos 4 `.ipynb` (+ `requirements.txt`) para `MyDrive/atari_drl/notebooks/`
   no Google Drive.

2. Abra `01_train_dqn.ipynb` no Colab Pro:
   - **Runtime → Change runtime type → T4 GPU** (ou L4)
   - **Runtime → Manage sessions → Premium** (sessão de até ~24h)

`BASE_DIR` detectado automaticamente: `/content/drive/MyDrive/atari_drl_results`
(tudo persiste entre sessões — sobrevive a queda do Colab).

### 2. Smoke test (~5 min)

Antes de disparar ~26h de treino, valide o pipeline em escala reduzida:

1. **Execute a célula 1 (Instalação)**.
   - Primeira execução: instala `gymnasium`, `ale-py` (com ROMs!),
     `stable-baselines3`, NumPy 2.x e dependências (~2 min).
   - **Reinicia o kernel automaticamente** após o install (necessário para
     evitar conflito de numpy-em-memória vs numpy-em-disco — bug clássico do Colab).
   - **RE-EXECUTE a célula 1** após o restart. Na 2ª passagem ela pula tudo
     e roda um smoke test do `ALE/Pong-v5` (`reset` + `step`).

2. Mantenha `SMOKE_TEST = True` na célula 9.

3. Execute o notebook **até a célula 19** (`Loop de treinamento`).

4. Descomente a linha `results = run_dqn_experiments()` e execute.

5. Em ~3 min, deve completar 1 run de 10k timesteps em Pong com seed 42.
   Espere ver `✓ [DQN_Pong_42] concluído em 2.x min`.

### 3. Treino completo

Para **cada notebook de treino** (01, 02, 03):

1. Reabra o notebook (já no Drive).
2. Célula 9: **mude `SMOKE_TEST = False`**.
3. Execute todas as células 1–19.
4. **Descomente** a linha `results = run_{algo}_experiments()` na célula 19.
5. Execute. O loop roda as 12 combinações (4 jogos × 3 seeds) sequencialmente.

**Tempo estimado em T4** (500k timesteps por run):

| Algoritmo | n_envs | Tempo por run | Total (12 runs) |
|---|---:|---:|---:|
| DQN | 1 | ~80 min | ~16h |
| PPO | 8 | ~30 min | ~6h |
| A2C | 16 | ~20 min | ~4h |
| **Total** | | | **~26h** |

DQN provavelmente exige 2 sessões Colab (~24h cada). Isto é tratado
automaticamente:

> **Resume from checkpoint:** se a sessão cair no meio de um run, reabra o
> notebook, execute as células 1–19, e a célula 19 **retoma o run incompleto
> do último checkpoint salvo** (1 a cada 100k timesteps).

### 4. Verificação visual (~1 min por GIF)

Célula 21 dos notebooks 01/02/03 gera um GIF de 1 episódio do agente jogando.
Descomente o bloco no fim da célula, ajuste a `tag` (`PPO_Pong_42` etc.) e
execute. Confere qualitativamente o aprendizado.

### 5. Análise (~10 min)

Quando os 36 runs estiverem completos (36 arquivos em `models/{algo}/*.zip`),
abra `04_analysis.ipynb`:

1. Execute todas as células sequencialmente.
2. Outputs em `BASE_DIR`:
   - `figures/learning_curves.png` — Fig. 1 do paper
   - `figures/group1_bars.png` — boxplots Grupo I
   - `figures/group2_boxplots.png` — boxplots Grupo II
   - `figures/table_group1.csv` — métricas comparativas
   - `figures/table_group2.csv` — métricas de dificuldade
   - `figures/spearman.csv` — **ρ + p-value (validação H1)**
   - `figures/mann_whitney.csv` — comparações pareadas
   - `videos/side_by_side_*.gif` — DQN×PPO×A2C lado a lado
   - `videos/temporal_*.gif` — evolução temporal

3. Para gerar os GIFs comparativos: descomente as chamadas no fim das
   células 33 e 35.

---

## Protocolo experimental

| Item | Valor |
|---|---|
| Jogos | Pong, Breakout, Ms.Pac-Man, Montezuma's Revenge |
| Seeds | 42, 123, 2024 |
| Algoritmos | DQN, PPO, A2C (Stable-Baselines3 ≥ 2.4) |
| Orçamento | 500 000 timesteps por run |
| Avaliação | a cada 10 000 timesteps, 5 episódios determinísticos |
| Checkpoints | 5 uniformes por treino |
| Observação | `(4, 84, 84)` uint8 grayscale com frame stack |
| Pré-processamento | `AtariWrapper` (Mnih 2015 preset) |
| γ | 0.99 |
| Recompensa (treino) | clipped em sign({-1, 0, +1}) — padrão Atari |
| Recompensa (eval) | SCORE REAL (sem clip) — para métricas Grupo II |

Pipeline de wrappers:
```
gymnasium.make(ALE/<game>-v5)
  → AtariWrapper(noop=30, frame_skip=4, screen=84,
                 terminal_on_life_loss=True, clip_reward=True)
    → Monitor
      → DummyVecEnv / SubprocVecEnv
        → VecFrameStack(n_stack=4, channels_order="last")
          → VecTransposeImage     # (4, 84, 84) — input do CnnPolicy
```

### Diferença entre env de treino e env de avaliação

| Config | Treino | Avaliação |
|---|---|---|
| `clip_reward` | `True` (sinal denso) | `False` (score real) |
| `terminal_on_life_loss` | `True` (episódios curtos, mais updates) | `False` (rodar até game over) |
| `seed` | `seed` | `seed + 9999` (descorrelacionada) |

Esse desacoplamento é essencial para que a Métrica `τ` (taxa de "score ≥ threshold")
e `d̄` (score humano-normalizado) reflitam o desempenho REAL do agente — não
o sinal truncado de treino.

---

## Métricas

**Grupo I — comparação entre arquiteturas:**
- $\bar{R}_{\text{final}}$ — média dos últimos 50k ts
- AUC normalizada — eficiência amostral
- $t_{80\%}$ — tempo para 80% do máximo (frac. de T)
- $\sigma_{\text{final}}$ — desvio-padrão final

**Grupo II — dificuldade:**
- $\tau$ — taxa de sucesso (score ≥ threshold do jogo)
- $\bar{d}$ — score humano-normalizado (Mnih 2015 Eq. 1):
  $\bar{d} = (\text{score}_{\text{agent}} - \text{score}_{\text{random}}) /
             (\text{score}_{\text{human}} - \text{score}_{\text{random}})$
- Mortes por episódio (`info['lives']`)
- Tempo em frames (`ep_length × 4` — frame skip)

**Análise estatística:**
- **Spearman** entre rank canônico dos jogos (1=Pong → 4=Montezuma) e
  rank induzido pelas métricas: $|\rho| \geq 0.8$ valida H1
- **Mann-Whitney U** pareado (DQN×PPO, DQN×A2C, PPO×A2C) por jogo

---

## Troubleshooting

**"AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'"** —
conflito numpy-em-memória vs numpy-em-disco no Colab após pip install. A célula 1
detecta isso e **reinicia o kernel automaticamente**. Re-execute a célula 1
após o restart.

**"AttributeError: module 'ale_py' has no attribute 'ALEInterface'"** —
versão antiga de ale-py. Confirme `ale_py.__version__ ≥ 0.10.0` na célula 1.
ROMs vêm incluídas a partir dessa versão.

**"NamespaceNotFound: Namespace ALE not found"** — falta chamar
`gym.register_envs(ale_py)` ANTES de `gym.make("ALE/...")`. A célula 5
(imports) já faz isso.

**"Session crashed: out of memory"** — provável estouro do replay buffer DQN.
Mitigação aplicada: `optimize_memory_usage=True`. Se ainda crashar, reduza
`buffer_size` de 100_000 para 50_000 na célula 11.

**"Cannot pickle local object" em PPO/A2C** — alguns ambientes têm problema
com `SubprocVecEnv` (spawn vs fork). Os notebooks usam imports defensivos
dentro do `_thunk` para sobreviver a `spawn`. Se ainda falhar, troque
`SubprocVecEnv` por `DummyVecEnv` na célula 13 — vai treinar mais devagar
mas funciona em qualquer ambiente.

**Sessão Colab caiu no meio de um run** — não faça nada. Reabra o notebook,
execute as células 1–19, e a célula 19 retoma do último checkpoint salvo
(100k em 100k). O CSV de avaliação também é incremental e sobrevive a crash.

**Quero recomeçar um run do zero** — apague manualmente
`models/{algo}/{algo}_{game}_{seed}.zip` (idempotência usa esse arquivo
como flag). Apague também os checkpoints intermediários
em `models/checkpoints/{algo}/{algo}_{game}_{seed}/` se quiser zerar
completamente.

---

## Estrutura de saída

```
atari_drl_results/                   # BASE_DIR (Drive ou local)
├── models/
│   ├── DQN/
│   │   ├── DQN_Pong_42.zip         # modelo final
│   │   ├── DQN_Pong_123.zip
│   │   └── ...
│   ├── PPO/
│   ├── A2C/
│   └── checkpoints/
│       ├── DQN/
│       │   └── DQN_Pong_42/
│       │       ├── DQN_Pong_42_100000_steps.zip
│       │       ├── DQN_Pong_42_200000_steps.zip
│       │       └── ...
│       └── ...
├── logs/
│   ├── DQN/
│   │   ├── DQN_Pong_42_eval.csv     # 1 linha por episódio
│   │   └── ...
│   ├── PPO/
│   ├── A2C/
│   └── tb/                           # TensorBoard por algoritmo
├── figures/                          # gerados pelo 04_analysis
└── videos/                           # GIFs do 04_analysis
```

---

## Citações para o paper

Stack:
- Towers et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032.
- Raffin et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Bellemare et al. (2013). *The Arcade Learning Environment: An Evaluation Platform for General Agents*. JAIR 47.
- Machado et al. (2018). *Revisiting the Arcade Learning Environment*. JAIR 61.

Algoritmos e baselines:
- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML.
- Badia et al. (2020). *Agent57: Outperforming the Atari Human Benchmark*. ICML.

Métricas / human baselines:
- Mnih et al. (2015) — Table S2 (human/random scores)
- Wang et al. (2016). *Dueling Network Architectures for Deep RL*. ICML — refinamentos de baseline.
