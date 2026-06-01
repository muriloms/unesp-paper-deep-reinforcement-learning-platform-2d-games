# Mario DRL — Projeto Modular

Pipeline modular para o artigo *Avaliação Automática de Dificuldade em Jogos de Plataforma 2D por meio de Agentes de Deep Reinforcement Learning* (DQN vs PPO vs A2C no Super Mario Bros).

## Estrutura do projeto

```
mario_drl/
├── src/                          # Pacote Python — toda a lógica vive aqui
│   ├── config.py                 # Constantes (paths, fases, seeds, hyperparams)
│   ├── env.py                    # Environment factory + CompatJoypadSpace
│   ├── callbacks.py              # MarioEvalCallback (persistência incremental)
│   ├── training.py               # train_one + resume from checkpoint + run_matrix
│   ├── analysis.py               # Grupos I/II + Spearman + Mann-Whitney
│   ├── plots.py                  # learning curves, group I/II charts
│   ├── visualization.py          # GIF rendering (agent, side-by-side, evolution)
│   └── utils.py                  # seed, device, env validation
├── scripts/                      # CLIs executáveis
│   ├── train.py                  # python scripts/train.py
│   ├── analyze.py                # python scripts/analyze.py
│   └── render.py                 # python scripts/render.py
├── notebooks/
│   └── 99_analysis_explore.ipynb # Análise interativa (importa de src/)
├── mario_drl_results/            # Saídas (criado em runtime — ignorado pelo git)
│   ├── models/                   # *.zip — modelos finais
│   │   └── checkpoints/          # *_{N}_steps.zip — para resume e evolução
│   ├── logs/                     # *.csv — métricas por episódio
│   ├── tensorboard/              # TB logs
│   ├── metrics/                  # CSVs consolidados (Grupos I/II, Spearman, MW)
│   └── plots/                    # PNGs + GIFs
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Pré-requisitos

- **Linux** (Ubuntu/Debian) com NVIDIA GPU + driver CUDA
- **`uv`** instalado:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Bibliotecas do sistema** para compilar `nes-py`:
  ```bash
  sudo apt update
  sudo apt install -y build-essential python3-dev libsdl2-dev
  ```

> Use **Python 3.10 ou 3.11**. O `nes-py 8.2.1` não compila em 3.12+.

## Setup

A partir da raiz do projeto:

```bash
# 1) Python 3.10 gerenciado pelo uv (não interfere no Python do sistema)
uv python install 3.10

# 2) Venv local
uv venv --python 3.10
source .venv/bin/activate

# 3) Dependências travadas
uv pip install -r requirements.txt
```

Se precisar de uma versão específica de CUDA (ex: 12.1):
```bash
uv pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu121
```

## Uso via CLI

### Treinamento

Validação rápida (~2 min, 1 algo × 1 fase × 1 seed × 10k timesteps):
```bash
python scripts/train.py --profile smoke
```

Experimento completo (36 treinos × 500k timesteps, ~60h):
```bash
python scripts/train.py --profile full
```

**Subsets** (úteis para fatiar o trabalho em sessões):
```bash
# Apenas PPO em todas as fases e seeds
python scripts/train.py --profile full --algos PPO

# DQN + PPO só na fase 1-1
python scripts/train.py --profile full --algos DQN PPO --stages 1-1

# Um treino específico, ignorando o que já foi feito
python scripts/train.py --profile full --algos DQN --stages 4-1 --seeds 42 --overwrite
```

### Análise

Gera todos os CSVs (`metrics/`) e plots (`plots/`):
```bash
python scripts/analyze.py
```

Só CSVs (sem PNGs):
```bash
python scripts/analyze.py --no-plots
```

### Visualizações (GIFs)

```bash
# 1 modelo, 1 fase
python scripts/render.py agent --algo PPO --stage 1-1 --seed 42

# DQN vs PPO vs A2C lado-a-lado (mesma fase, mesmo seed)
python scripts/render.py compare --stage 4-1 --seed 42

# Evolução temporal do PPO (5 checkpoints uniformes)
python scripts/render.py evolution --algo PPO --stage 1-1 --seed 42

# Gera todos os GIFs disponíveis (agent + compare para cada config)
python scripts/render.py all
```

### Notebook de análise

```bash
python -m ipykernel install --user --name=mario_drl --display-name "Python (mario_drl)"
jupyter lab notebooks/99_analysis_explore.ipynb
```

Seleciona o kernel "Python (mario_drl)". O notebook importa diretamente de `src/`, então qualquer mudança nos módulos se propaga sem precisar editar o notebook.

## Resume from checkpoint

`scripts/train.py` é **idempotente E retomável**:

| Estado do output | Comportamento |
|---|---|
| Modelo final + CSV existem | **Pula** completamente |
| Não há final, mas há checkpoint parcial (ex: `*_300000_steps.zip`) | **Retoma** dali, treina só os 200k restantes |
| Nada existe | Começa do zero |

**Quanto perde no máximo se a sessão cair?** No perfil `full` salvamos 5 checkpoints uniformes em 500k timesteps, então uma queda perde **no máximo 100k timesteps de progresso** (~30 min de PPO ou A2C, ~1h de DQN).

Para perder ainda menos, edite `PROFILES["full"]["n_checkpoints"]` em `src/config.py` (ex: 10 → checkpoint a cada 50k timesteps).

## TensorBoard ao vivo

Em outro terminal, com o venv ativo:
```bash
tensorboard --logdir mario_drl_results/tensorboard
```
Abra `http://localhost:6006` no navegador. Acompanhe as curvas de recompensa e loss enquanto os treinos rodam.

## Estimativa de tempo (RTX 3060/3070, perfil full)

| Algoritmo | n_envs | Tempo / treino | × 12 treinos |
|---|---|---|---|
| DQN | 1 | ~2-3h | ~30h |
| PPO | 8 | ~1-1.5h | ~15h |
| A2C | 16 | ~1-1.5h | ~15h |
| **Total** | | | **~60h** |

Dispare em sessões fatiadas com `--algos` ou `--stages` — o resume retoma de onde parou.

## Workflow recomendado

1. **Validação:** `python scripts/train.py --profile smoke`
2. **Treino real fatiado por algoritmo (sessões paralelas em terminais):**
   - Term 1: `python scripts/train.py --profile full --algos DQN`
   - Term 2: `python scripts/train.py --profile full --algos PPO`
   - Term 3: `python scripts/train.py --profile full --algos A2C`
3. **Análise parcial enquanto ainda treina:** `python scripts/analyze.py` (funciona com dados parciais)
4. **Análise final + GIFs:** `python scripts/analyze.py && python scripts/render.py all`
5. **Exploração no notebook:** `jupyter lab notebooks/99_analysis_explore.ipynb`

## Solução de problemas

**`OverflowError: Python integer 1024 out of bounds for uint8`**
→ Está em Python 3.12+. Use 3.10 ou 3.11.

**`apply_api_compatibility` é argumento desconhecido**
→ Versão errada do `gym-super-mario-bros`. Reinstale com `pip install gym-super-mario-bros==7.4.0`.

**Crash de RAM aos 10k timesteps no DQN**
→ Buffer de replay (~2.8 GB). Reduza `buffer_size` no `HPARAMS["DQN"]` para 50_000 em `src/config.py`.

**`No module named 'src'`**
→ Rode a partir da raiz do projeto: `cd /caminho/do/projeto && python scripts/train.py`.
