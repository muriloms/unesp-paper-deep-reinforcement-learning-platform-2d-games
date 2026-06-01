# Mario DRL Playtester — Pacote de Implementação

Pacote de execução para o projeto científico **"Avaliação Automática de Dificuldade
em Jogos de Plataforma 2D por meio de Agentes de Deep Reinforcement Learning"**
(disciplina de Redes Neurais Artificiais — PPGCC/UNESP).

Treina **DQN, PPO e A2C** no Super Mario Bros (NES) sob orçamento computacional
fixo de 500 000 timesteps × 4 fases × 3 seeds = **36 runs**, e analisa as métricas
agregadas via Spearman + Mann-Whitney U para validar a hipótese de que métricas
dos agentes refletem a dificuldade canônica das fases.

---

## Arquivos

| Arquivo | Função | Células |
|---|---|---|
| `01_train_dqn.ipynb` | Treina DQN nas 4 fases × 3 seeds | 22 |
| `02_train_ppo.ipynb` | Treina PPO nas 4 fases × 3 seeds | 22 |
| `03_train_a2c.ipynb` | Treina A2C nas 4 fases × 3 seeds | 22 |
| `04_analysis.ipynb` | Consolida métricas, gera tabelas/figuras/GIFs | 36 |
| `requirements.txt` | Versões pinadas da stack | — |

Os 3 notebooks de treino compartilham a mesma estrutura. Cada um produz, no
`BASE_DIR/`, modelos finais, checkpoints intermediários, logs TensorBoard e
CSVs de avaliação. O notebook de análise consome esses CSVs.

---

## Stack

Migrou da stack legada (`gym 0.26.2` + `gym-super-mario-bros 7.4.0` + `nes-py 8.2.1`
+ `shimmy` + `condacolab` para Python 3.10) para uma stack **gymnasium-nativa**:

- `gymnasium ≥ 1.0`
- `stable-baselines3 ≥ 2.4`
- `torch ≥ 2.4`
- `nes-py` + `gym-super-mario-bros` — forks gymnasium-compatible do
  [tooichitake/gymnasium-mario](https://github.com/tooichitake/gymnasium-mario)

Sem condacolab, sem downgrade de Python, sem shimmy. Funciona com Python 3.10–3.12
(Colab Pro default ≥ 3.11). Mantém o `info` dict completo do Kauten original
(`x_pos`, `y_pos`, `life`, `score`, `flag_get`, `time`, ...) — drop-in com o
código pré-existente.

---

## Roteiro de execução em Colab Pro

### 1. Setup inicial (uma vez)

1. Faça upload dos 4 `.ipynb` (e do `requirements.txt`) para uma pasta do
   Google Drive — recomendado: `MyDrive/mario_drl/notebooks/`. O **`BASE_DIR`**
   detectado automaticamente pelos notebooks é
   `/content/drive/MyDrive/mario_drl_results` — todos os modelos, logs e
   figuras vão para lá (sobrevive à queda da sessão Colab).

2. Abra `01_train_dqn.ipynb` no Colab Pro. Confirme:
   - **Runtime → Change runtime type → T4 GPU** (ou L4, se disponível).
   - **Runtime → Manage sessions → Premium** (sessão de até ~24h).

### 2. Smoke test (~5 min)

Antes de disparar 47h de treino, **rode o pipeline em escala reduzida**:

1. Execute a **célula 1 (Instalação)**.
   - Na **primeira execução**, ela instala cmake, clona o repo do fork,
     compila `nes-py` (~30s), instala a stack RL e os pacotes de análise.
   - Em seguida ela **reinicia automaticamente o kernel** (necessário para
     evitar conflito entre o `numpy` que o Colab já tinha em memória e o
     que pode ter sido atualizado durante o install).
   - **RE-EXECUTE a célula 1.** Na segunda passagem, ela detecta que tudo
     está instalado, pula direto para a validação e roda um smoke test do
     ambiente (`reset` + `step` em 1-1).
2. Na célula 9 (`# Configuração global`), mantenha `SMOKE_TEST = True`.
3. Execute o notebook **até a célula 19** (`Loop de treinamento`).
4. Descomente a linha `results = run_dqn_experiments()` e execute.
5. Em ~3 min, deve completar 1 run de 10k timesteps em `1-1` com seed 42.
   Espere ver `✓ [DQN_1-1_42] concluído em 2.x min`.

### 3. Treino completo (~47h cumulativos, espalhados em múltiplas sessões)

Para **cada um dos 3 notebooks de treino**:

1. Reabra o notebook (ele já está na pasta do Drive).
2. Na célula 9, **mude `SMOKE_TEST = False`**.
3. Execute todas as células de 1 a 19.
4. **Descomente** a linha `results = run_{algo}_experiments()` na célula 19.
5. Execute. O loop roda as 12 combinações (4 fases × 3 seeds) sequencialmente.

**Tempo estimado em T4** (sob orçamento de 500k timesteps):

| Algoritmo | n_envs | Tempo por run | Total (12 runs) |
|---|---:|---:|---:|
| DQN | 1 | ~120 min | ~24h |
| PPO | 8 | ~50 min | ~10h |
| A2C | 16 | ~35 min | ~7h |

Como cada sessão Colab dura no máximo ~24h, o treino do **DQN provavelmente vai
exigir 2 sessões**. Isto é tratado automaticamente:

> **Resume from checkpoint:** se a sessão cair no meio de um run, basta reabrir
> o notebook, re-executar as células 1-19 (instalação pula, Drive remonta,
> modelos finais já existentes são pulados) e o loop **retoma o run incompleto
> do último checkpoint salvo** (1 a cada 100k timesteps).

### 4. Verificação visual (opcional, ~2 min por GIF)

Após treinar, a célula 21 dos notebooks 01/02/03 gera um GIF do agente jogando.
Descomente o bloco de exemplo no fim da célula, ajuste a `tag`
(`PPO_1-1_42` por exemplo) e execute. Confere qualitativamente que o agente
não está apenas parado pulando no mesmo lugar.

### 5. Análise (~10 min)

Depois que **os 36 runs estiverem completos** (`models/{algo}/*.zip` para
36 arquivos), abra `04_analysis.ipynb`:

1. Execute todas as células sequencialmente.
2. Outputs gerados em `BASE_DIR`:
   - `figures/learning_curves.png` — Fig. 1 do paper (curvas de aprendizado)
   - `figures/group1_bars.png` — boxplots Grupo I
   - `figures/group2_boxplots.png` — boxplots Grupo II
   - `figures/table_group1.csv` — métricas comparativas
   - `figures/table_group2.csv` — métricas de dificuldade
   - `figures/spearman.csv` — ρ + p-value (validação de H1)
   - `figures/mann_whitney.csv` — comparações pareadas
   - `videos/side_by_side_*.gif` — DQN×PPO×A2C lado a lado
   - `videos/temporal_*.gif` — evolução temporal (mesmo agente, checkpoints)

3. Para gerar os GIFs comparativos (opcional), descomente as chamadas no fim
   das células 33 e 35.

---

## Protocolo experimental preservado

Todos os elementos científicos do projeto original foram mantidos:

| Item | Valor |
|---|---|
| Fases | 1-1, 1-2, 4-1, 8-1 |
| Seeds | 42, 123, 2024 |
| Algoritmos | DQN, PPO, A2C (Stable-Baselines3 ≥ 2.4) |
| Orçamento | 500 000 timesteps por run |
| Avaliação | a cada 10 000 timesteps, 5 episódios determinísticos |
| Checkpoints | 5 uniformes por treino (para resume e evolução temporal) |
| Observação | `(4, 84, 84)` uint8 grayscale com frame stack |
| Ações | `SIMPLE_MOVEMENT` (7 ações discretas) |
| γ | 0.99 |
| Recompensa | sinal nativo `gym-super-mario-bros`, truncado em [-15, +15] |

Pipeline de wrappers (idêntico ao da stack antiga):
```
gym_super_mario_bros.make()
  → JoypadSpace(SIMPLE_MOVEMENT)
    → MaxAndSkipEnv(skip=4)
      → WarpFrame(84,84)
        → Monitor
          → DummyVecEnv / SubprocVecEnv
            → VecFrameStack(n_stack=4, channels_order="last")
              → VecTransposeImage     # (4, 84, 84) — input do CnnPolicy
```

Hiperparâmetros (Tabela 2 do paper) replicados na célula 11 de cada notebook,
incluindo `optimize_memory_usage=True` no DQN (crítico para Colab Free com
buffer_size=100k: economiza ~1.4 GB de RAM).

---

## Troubleshooting

**"AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'"** —
conflito entre o numpy que o Colab já tinha em memória e o numpy de disco
após o `pip install`. A célula 1 detecta isso e **reinicia o kernel
automaticamente** após o primeiro install. Basta re-executar a célula 1
depois do restart e seguir.

**"Session crashed: out of memory"** — provável estouro do replay buffer do DQN.
Mitigações já aplicadas: `optimize_memory_usage=True`. Se ainda crashar, reduza
`buffer_size` de 100_000 para 50_000 na célula 11.

**"Cannot pickle local object" em PPO/A2C** — alguns ambientes têm problema com
`SubprocVecEnv`. Os notebooks usam imports defensivos dentro do `_thunk` para
sobreviver a `spawn` vs `fork`. Se ainda assim falhar, troque `SubprocVecEnv`
por `DummyVecEnv` na célula 13 (`make_vec_env_mario`) — vai treinar mais devagar
mas funciona.

**"cmake: command not found"** — só ocorre fora do Colab. No Colab a célula 3
instala via `apt-get`. Em outra máquina: `sudo apt install cmake build-essential`.

**Sessão Colab caiu no meio de um run** — não faça nada. Reabra o notebook,
execute as células 1–19, e a célula 19 vai automaticamente retomar do último
checkpoint salvo (cada 100k timesteps). O CSV de avaliação também é incremental
e sobrevive a crashes.

**Logs antigos atrapalhando uma nova rodada** — apague manualmente o arquivo
final `models/{algo}/{algo}_{stage}_{seed}.zip` (a checagem de idempotência
usa esse arquivo como flag). Os checkpoints intermediários podem ser apagados
também se quiser recomeçar do zero.

---

## Estrutura de saída

```
mario_drl_results/                   # BASE_DIR (em Drive ou local)
├── models/
│   ├── DQN/
│   │   ├── DQN_1-1_42.zip          # modelo final (1 por run)
│   │   ├── DQN_1-1_123.zip
│   │   └── ...
│   ├── PPO/
│   ├── A2C/
│   └── checkpoints/
│       ├── DQN/
│       │   └── DQN_1-1_42/
│       │       ├── DQN_1-1_42_100000_steps.zip
│       │       ├── DQN_1-1_42_200000_steps.zip
│       │       └── ...
│       └── ...
├── logs/
│   ├── DQN/
│   │   ├── DQN_1-1_42_eval.csv     # 1 linha por episódio de avaliação
│   │   └── ...
│   ├── PPO/
│   ├── A2C/
│   └── tb/                          # TensorBoard logs por algoritmo
├── figures/                         # gerados pelo 04_analysis
└── videos/                          # GIFs gerados pelo 04_analysis
```

---

## Citações

Stack:
- Towers et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032.
- Raffin et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR.
- Kauten, C. (2018). *Super Mario Bros for OpenAI Gym*. github.com/Kautenja/gym-super-mario-bros.
- Zhao, Z. (2025). *Gymnasium Mario*. github.com/tooichitake/gymnasium-mario.
