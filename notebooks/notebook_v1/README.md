# Mario DRL — Fase 1 (Notebook)

Notebook ponta-a-ponta para o artigo *Avaliação Automática de Dificuldade em Jogos de Plataforma 2D por meio de Agentes de Deep Reinforcement Learning* (DQN vs PPO vs A2C no Super Mario Bros).

## Pré-requisitos (Linux + GPU)

- **`uv`** instalado:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **CUDA 11.8 ou 12.x** com driver NVIDIA compatível
- **Pacotes do sistema** para compilar `nes-py` (Cython + SDL2):
  ```bash
  sudo apt update
  sudo apt install -y build-essential python3-dev libsdl2-dev
  ```

> O `nes-py` **não compila** em Python 3.12+ no momento. Use **3.10 ou 3.11** — o passo seguinte garante isso via `uv`.

## Setup do ambiente com `uv`

A partir da pasta onde estão `mario_drl_phase1.ipynb` e `requirements.txt`:

```bash
# 1) Instala Python 3.10 gerenciado pelo uv (não interfere no Python do sistema)
uv python install 3.10

# 2) Cria o venv .venv/ na pasta do projeto, fixado em 3.10
uv venv --python 3.10

# 3) Ativa
source .venv/bin/activate

# 4) Instala as dependências travadas
uv pip install -r requirements.txt

# 5) Registra o kernel para o Jupyter encontrar
python -m ipykernel install --user --name=mario_drl --display-name "Python (mario_drl)"
```

### Nota sobre PyTorch + CUDA

O `pip install torch` padrão já traz binários com CUDA no Linux x86_64, então o passo 4 acima resolve na maioria dos casos. Se precisar forçar uma versão específica de CUDA (ex: CUDA 12.1), substitua a instalação do torch por:

```bash
uv pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt   # instala o resto
```

Verifique o link correto na [matriz oficial do PyTorch](https://pytorch.org/get-started/locally/) para sua versão de CUDA.

## Abrindo o notebook

```bash
jupyter lab
```

Selecione o kernel **"Python (mario_drl)"**. A célula `!pip install ...` (célula 3 do notebook) pode ser **ignorada** — o `uv` já instalou tudo.

## Roteiro de execução

1. **Smoke test** (~2 min): `SMOKE_TEST=True` na célula 7. Roda 1 treino PPO curto na fase 1-1 para validar o pipeline ponta-a-ponta. Se passar, está tudo OK.
2. **Experimento completo** (~60h GPU): set `SMOKE_TEST=False`, **reinicie o kernel** e descomente `results = run_all_experiments()` na célula 19. A função é idempotente — se cair, é só rerodar e ela pula o que já completou.
3. **Análise**: depois dos treinos, execute as células 20→35 sequencialmente. Os CSVs e PNGs ficam em `./mario_drl_results/`.

## Estrutura gerada

```
mario_drl_results/
├── models/       # *.zip — modelos SB3 salvos
├── logs/         # *.csv — métricas de avaliação por episódio
├── tensorboard/  # logs do TensorBoard
├── metrics/      # group1_*.csv, group2_*.csv, spearman_*.csv, mannwhitney_*.csv
└── plots/        # learning_curves.png, group1_comparison.png, group2_difficulty.png
```

## Dicas práticas

- **Rodando em lotes**: edite `STAGES_TO_RUN` na célula 7 para rodar uma fase de cada vez. Útil para fechar resultados parciais sem prender o terminal por dias.
- **TensorBoard ao vivo**: `tensorboard --logdir mario_drl_results/tensorboard` em outro terminal.
- **Memória de GPU**: PPO (8 envs) e A2C (16 envs) podem precisar de ~4–6 GB de VRAM. Se faltar memória, reduza `n_envs` no `HPARAMS` da célula 7 (custo: aumenta tempo total).
- **DQN é o mais lento** (apenas 1 env) — vale disparar primeiro PPO/A2C em todas as fases enquanto DQN roda em paralelo.
- **Reativar o ambiente depois**: basta `source .venv/bin/activate` na pasta do projeto.
- **Adicionar uma dep depois**: `uv pip install <pacote>` — fica registrada no venv ativo.

## Próxima fase (modular)

Quando os experimentos estiverem rodando, partimos para:

- `train.py` CLI com `argparse` (ou `typer`)
- Configs em YAML (`config/dqn.yaml`, etc.)
- `analyze.py` para gerar tabelas LaTeX direto para o artigo
- Conversão do `requirements.txt` para `pyproject.toml` gerenciado pelo `uv` em modo projeto (`uv init` + `uv add`)
- Estrutura `src/` com módulos separados (`env.py`, `callbacks.py`, `metrics.py`)
