# Avaliação Automática de Dificuldade em Jogos de Plataforma 2D
## Agentes de Deep Reinforcement Learning como *playtesters* automáticos

Repositório de código e dados experimentais do artigo **"Avaliação Automática de Dificuldade em Jogos de Plataforma 2D por meio de Agentes de Deep Reinforcement Learning"** (Murilo Mazzotti Silvestrini, PPGCC/UNESP).

O trabalho investiga se três arquiteturas de Deep RL (DQN, PPO, A2C) podem servir como *playtesters* automáticos para estimar dificuldade em jogos de plataforma 2D, usando o Super Mario Bros como bancada experimental.

## Resumo

O balanceamento de dificuldade é um desafio central no design de jogos, tradicionalmente realizado por *playtesting* humano — custoso, subjetivo e pouco escalável. Este trabalho treina e compara DQN, PPO e A2C em quatro fases do Super Mario Bros (1-1, 1-2, 4-1, 8-1), com três seeds cada (36 treinamentos), sob orçamento fixo de 500k timesteps. O objetivo é duplo: (i) comparar as arquiteturas; (ii) investigar se métricas derivadas dos agentes acompanham a progressão canônica de dificuldade.

Os resultados mostram que **DQN é consistentemente a arquitetura mais eficiente em amostras sob o orçamento restrito**, superando A2C e PPO em todas as fases. Nenhum agente completou qualquer fase dentro do orçamento, tornando degeneradas as métricas dependentes de conclusão. A métrica de progresso normalizado exibe a tendência negativa esperada em relação à dificuldade canônica, porém com correlação fraca e estatisticamente inconclusiva.

## Estrutura do projeto

```
.
├── src/                          # Pacote Python — lógica do experimento
│   ├── config.py                 # Constantes globais (fases, seeds, hyperparams, paths)
│   ├── env.py                    # Environment factory + CompatJoypadSpace
│   ├── wrappers.py               # ProgressRewardWrapper (reward shaping experimental)
│   ├── callbacks.py              # MarioEvalCallback (persistência incremental dos CSVs)
│   ├── training.py               # train_one + resume from checkpoint + run_matrix
│   ├── analysis.py               # Grupos I/II + Spearman + Mann-Whitney + load_all_logs
│   ├── plots.py                  # Geração de PNGs (1 arquivo por gráfico)
│   ├── visualization.py          # GIFs de agente (individual, comparação, evolução)
│   └── utils.py                  # set_global_seed, get_device, validações
│
├── scripts/                      # CLIs executáveis
│   ├── train.py                  # Dispara treinos (perfis smoke/mid/long/full)
│   ├── analyze.py                # Carrega logs → métricas → plots
│   ├── render.py                 # Gera GIFs (agent / compare / evolution / all)
│   └── recompute_metrics.py      # Re-avalia modelos com contagem de mortes corrigida
│
├── notebooks/
│   ├── 99_analysis_explore.ipynb # Análise interativa importando de src/
│   ├── notebook_v1/              # Versões históricas (notebook monolítico inicial)
│   ├── notebook_colab_v1/        # Tentativa de split em 4 notebooks para Colab
│   ├── notebook_colab_v2/        # Iteração subsequente
│   └── notebooks_colab_atari_v1/ # Experimentos paralelos
│
├── mario_drl_results/            # Saídas do experimento principal (vanilla)
│   ├── models/                   # *.zip — 36 modelos finais
│   │   └── checkpoints/          # *_{N}_steps.zip — 5 checkpoints por treino
│   ├── logs/                     # *.csv (trajetória) + *_eval.csv (deaths corrigidos)
│   ├── tensorboard/              # Eventos do TensorBoard por experimento
│   ├── metrics/                  # CSVs consolidados (Grupos I e II, Spearman, MW)
│   └── plots/                    # PNGs (curvas, métricas) + GIFs (rollouts)
│
├── mario_drl_results_v1/         # Snapshot inicial (smoke tests, validação do pipeline)
├── mario_drl_results_v2/         # Experimentos com reward shaping (PPO + ProgressRewardWrapper)
│
├── docs/
│   ├── TRAINING_RECIPES.md       # Tabela completa de combinações de comandos
│   └── REWARD_SHAPING.md         # Diagnóstico + uso do ProgressRewardWrapper
│
├── requirements.txt              # Dependências travadas (Python 3.10/3.11)
├── pyproject.toml                # Metadata do projeto + entry points
└── README.md                     # Este arquivo
```

### Convenções de nomes

- **Modelo final:** `{ALGO}_stage{X-Y}_seed{N}.zip` (ex: `PPO_stage1-1_seed42.zip`)
- **Checkpoint intermediário:** `{exp_id}_{TIMESTEPS}_steps.zip`
- **Variante com reward shaping:** sufixo `_shape` (ex: `PPO_stage1-1_seed42_shape.zip`)
- **CSV re-avaliado com deaths corrigido:** sufixo `_eval` (ex: `PPO_stage1-1_seed42_eval.csv`)

## Pré-requisitos

- **Linux** (Ubuntu/Debian) com NVIDIA GPU + driver CUDA
- **Python 3.10 ou 3.11** — `nes-py 8.2.1` não compila em 3.12+
- **`uv`** instalado:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Bibliotecas do sistema** para compilar `nes-py`:
  ```bash
  sudo apt update
  sudo apt install -y build-essential python3-dev libsdl2-dev
  ```

## Instalação

```bash
# 1) Python 3.10 gerenciado pelo uv
uv python install 3.10

# 2) Venv local
uv venv --python 3.10
source .venv/bin/activate

# 3) Dependências travadas
uv pip install -r requirements.txt
```

Para o kernel do Jupyter:

```bash
python -m ipykernel install --user --name=mario_drl --display-name "Python (mario_drl)"
```

## Protocolo experimental

| Item | Valor |
|---|---|
| Algoritmos | DQN, PPO, A2C |
| Fases (Super Mario Bros NES) | 1-1, 1-2, 4-1, 8-1 |
| Seeds | 42, 123, 2024 |
| Total de treinamentos | 36 (3 × 4 × 3) |
| Timesteps por treino | 500.000 |
| Avaliações durante o treino | a cada 10.000 timesteps |
| Episódios por avaliação | 5 (determinísticos) |
| Observação | `(4, 84, 84)` grayscale + frame stack |
| Ações | `SIMPLE_MOVEMENT` (7 ações discretas) |
| `gamma` | 0.99 |
| Checkpoints intermediários | 5 por treino |

Hyperparams completos em `src/config.py` (dict `HPARAMS`).

## Uso via CLI

### Treinamento

Validação rápida do pipeline (~2 min):
```bash
python scripts/train.py --profile smoke
```

Experimento completo (36 treinos × 500k timesteps, ~60h em RTX 3060/3070):
```bash
python scripts/train.py --profile full
```

**Subsets** (para fatiar em sessões):
```bash
# Apenas PPO em todas as fases e seeds
python scripts/train.py --profile full --algos PPO

# DQN + PPO só na fase 1-1
python scripts/train.py --profile full --algos DQN PPO --stages 1-1

# Um treino específico, ignorando o que já foi feito
python scripts/train.py --profile full --algos DQN --stages 4-1 --seeds 42 --overwrite
```

Lista completa de combinações em [`docs/TRAINING_RECIPES.md`](docs/TRAINING_RECIPES.md).

### Análise

Gera CSVs consolidados (`metrics/`) e PNGs (`plots/`):
```bash
python scripts/analyze.py
```

Para usar os CSVs com mortes corrigidas (após rodar `recompute_metrics.py`):
```bash
python scripts/analyze.py --use-eval-csvs
```

A flag `--use-eval-csvs` faz com que **apenas o Grupo II** (τ, d̄, mortes, tempo) use os CSVs `_eval` — curvas de aprendizado e Grupo I continuam usando a trajetória completa do treino.

### Re-avaliação dos modelos

Por uma limitação do `nes-py`, a contagem de mortes original ignora colisões com Goombas pequenos (não decrementam `info['life']`). Para regenerar os CSVs com contagem correta — **sem retreinar** — basta carregar cada modelo salvo e re-avaliar:

```bash
python scripts/recompute_metrics.py
```

Gera arquivos `*_eval.csv` em `mario_drl_results/logs/` preservando os CSVs originais. ~30 min para os 36 modelos.

### Visualizações

```bash
# GIF de UM modelo
python scripts/render.py agent --algo PPO --stage 1-1 --seed 42

# Comparação DQN/PPO/A2C lado-a-lado (mesma fase, mesmo seed)
python scripts/render.py compare --stage 4-1 --seed 42

# Evolução temporal do PPO (5 checkpoints uniformes)
python scripts/render.py evolution --algo PPO --stage 1-1 --seed 42

# Gera todos os GIFs disponíveis (agent + compare por configuração)
python scripts/render.py all
```

Para modelos com reward shaping, adicione `--variant shape`:
```bash
python scripts/render.py agent --algo PPO --stage 1-1 --seed 42 --variant shape
```

### TensorBoard

Acompanhamento em tempo real durante o treino:
```bash
tensorboard --logdir mario_drl_results/tensorboard
```

### Notebook interativo

```bash
jupyter lab notebooks/99_analysis_explore.ipynb
```

Importa diretamente de `src/`, então qualquer mudança nos módulos se propaga sem editar o notebook.

## Robustez do pipeline

O `train_one` em `src/training.py` é **idempotente e retomável**:

| Estado do output | Comportamento |
|---|---|
| Modelo final + CSV existem | Pula completamente |
| Não há final, mas há checkpoint parcial | **Retoma** dali, treina só os timesteps restantes |
| Nada existe | Começa do zero |
| `--overwrite` | Ignora tudo e retreina do zero |

Se uma sessão cair (terminal fechou, GPU travou), basta repetir o comando original — o script detecta o último checkpoint salvo e continua. Perde no máximo `500.000 / 5 = 100.000` timesteps de progresso por treino interrompido.

## Métricas coletadas

**Grupo I — comparação entre arquiteturas (calculadas sobre a trajetória de treino):**
- $\bar{R}_{\text{final}}$: média de recompensa nos últimos 50k timesteps
- AUC normalizada da curva de aprendizado
- $t_{80\%}$: timesteps até atingir 80% do máximo
- $\sigma_{\text{final}}$: desvio-padrão dos últimos 50k

**Grupo II — avaliação de dificuldade (sobre a última avaliação):**
- $\tau$: taxa de conclusão (média de `flag_get`)
- $\bar{d}$: distância normalizada (`max_x_pos` / comprimento da fase)
- Mortes por episódio
- Tempo até conclusão (frames, apenas episódios bem-sucedidos)

**Análise estatística:**
- Correlação de **Spearman** entre o ranking de cada métrica do Grupo II e o ranking canônico das fases
- Teste de **Mann-Whitney U** pareado entre pares de algoritmos, por fase

## Reward shaping (experimento secundário)

Durante o desenvolvimento, observou-se convergência prematura para política trivial em algumas configurações. Como contraponto científico, implementou-se `ProgressRewardWrapper` em `src/wrappers.py`, que adiciona ao reward nativo:

- bonus proporcional ao avanço em `x_pos` (apenas para novos recordes, não vai-e-vem)
- penalidade por andar para trás
- penalidade por travar sem progresso
- penalidade explícita de tempo
- bonus por completar a fase

Para treinar com shaping:
```bash
python scripts/train.py --profile full --algos PPO --reward-shaping
```

Modelos shaped ganham sufixo `_shape` e não sobrescrevem o baseline. Discussão completa em [`docs/REWARD_SHAPING.md`](docs/REWARD_SHAPING.md). Resultados parciais em `mario_drl_results_v2/`.

## Workflow típico

```bash
# 1. Validação rápida
python scripts/train.py --profile smoke

# 2. Experimento completo, fatiado por algoritmo (3 terminais paralelos)
nohup python scripts/train.py --profile full --algos DQN > dqn.log 2>&1 &
nohup python scripts/train.py --profile full --algos PPO > ppo.log 2>&1 &
nohup python scripts/train.py --profile full --algos A2C > a2c.log 2>&1 &

# 3. Re-avaliar para corrigir contagem de mortes
python scripts/recompute_metrics.py

# 4. Análise final (CSVs + plots)
python scripts/analyze.py --use-eval-csvs

# 5. Visualizações (GIFs)
python scripts/render.py all

# 6. Exploração interativa
jupyter lab notebooks/99_analysis_explore.ipynb
```

## Solução de problemas

**`OverflowError: Python integer 1024 out of bounds for uint8`**
→ Está usando Python 3.12+. O `nes-py 8.2.1` não suporta. Use 3.10 ou 3.11.

**`apply_api_compatibility` é argumento desconhecido**
→ Versão errada do `gym-super-mario-bros`. Reinstale com `pip install gym-super-mario-bros==7.4.0`.

**Crash de RAM aos 10k timesteps no DQN**
→ Buffer de replay (~2.8 GB). Reduza `buffer_size` no `HPARAMS["DQN"]` para 50_000 em `src/config.py`.

**`No module named 'src'`**
→ Rode a partir da raiz do projeto: `cd /caminho/do/projeto && python scripts/train.py`.

**Mortes sempre zero nos CSVs**
→ Limitação conhecida do `nes-py` (Goombas pequenos não decrementam `life`). Rode `python scripts/recompute_metrics.py` para gerar CSVs corrigidos.



## Licença e dados

Código sob licença MIT (ou conforme política institucional do PPGCC/UNESP). Os experimentos foram conduzidos com fins acadêmicos; ROMs do Super Mario Bros são propriedade da Nintendo e não são distribuídas com este repositório — o pacote `gym-super-mario-bros 7.4.0` traz a integração com a ROM original embutida no instalador para fins de pesquisa.

