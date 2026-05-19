# Mario DRL — Fluxo modular (4 notebooks)

Arquitetura dividida em 4 notebooks para permitir **execução paralela** dos três algoritmos e isolar a análise do treinamento.

## Visão geral

```
01_train_dqn.ipynb  ─┐
02_train_ppo.ipynb  ─┼─►  mario_drl_results/  ─►  04_analysis.ipynb  ─►  PNGs + GIFs + CSVs
03_train_a2c.ipynb  ─┘     (Drive em Colab,
                            pasta local fora)
```

| Notebook | O que faz | Tempo (500k timesteps, GPU mediana) |
|---|---|---|
| **`01_train_dqn.ipynb`** | Treina DQN em 4 fases × 3 seeds = 12 treinamentos | ~30h (1 env paralelo) |
| **`02_train_ppo.ipynb`** | Treina PPO em 4 fases × 3 seeds = 12 treinamentos | ~15h (8 envs paralelos) |
| **`03_train_a2c.ipynb`** | Treina A2C em 4 fases × 3 seeds = 12 treinamentos | ~15h (16 envs paralelos) |
| **`04_analysis.ipynb`** | Carrega CSVs, gera todas as métricas, plots e visualizações | ~10 min |

## Por que dividir?

- **Paralelização**: rode os 3 treinamentos em 3 sessões Colab simultâneas (mesma conta ou contas diferentes) — wall-clock vira ~30h em vez de 60h
- **Quota de GPU**: Colab Free dá ~12h/dia. Notebook menor cabe melhor na quota diária — vai dando passos
- **Isolamento de falhas**: se A2C crashar, DQN/PPO já estão salvos no Drive
- **Análise iterativa**: dá pra refinar plots/tabelas no `04_analysis` sem retreinar nada

## Setup inicial (uma vez)

1. **Crie a pasta** no Google Drive: `MyDrive/mario_drl_results/` (vazia — os notebooks criam subpastas sozinhos)
2. **Faça upload** dos 4 notebooks em qualquer pasta do Drive (ou abra-os direto via Colab)
3. **Configure GPU** no Colab: `Runtime → Change runtime type → GPU (T4 ou superior)`

## Roteiro de execução

### Fase 1 — Smoke tests (validação, ~5 min total)

Em cada notebook de treino (`01`, `02`, `03`), individualmente:

1. Rode a **célula 1** (instalação) — espera ~3 min
2. **Restart session** (`Ctrl+M .`)
3. Rode todas as células com `SMOKE_TEST=True` (default) — valida pipeline em ~2 min, treina 1 modelo curto
4. A última célula renderiza um GIF do agente — confirme que vê Mario fazendo coisas

### Fase 2 — Treinamento completo (~15–30h por notebook, em paralelo)

Em cada notebook de treino, depois do smoke test:

1. Edite **célula 4** (config) → `SMOKE_TEST = False`
2. Reinicie a sessão
3. Rode todas as células em sequência até a do "Loop de treinamento"
4. **Descomente** a linha `# run_{algo}_experiments()` e execute
5. Deixe rodando. É idempotente — se a sessão cair, é só reabrir e rerodar, o que já completou é pulado

**Dica para a quota Colab Free:** edite `STAGES_TO_RUN` e `SEEDS_TO_RUN` na célula 4 para fatiar o trabalho em sessões de ~4h. Exemplo:

```python
# Sessão 1: stage 1-1 com todas as seeds
STAGES_TO_RUN = ["1-1"]
SEEDS_TO_RUN  = [42, 123, 2024]

# Sessão 2 (no dia seguinte): stage 1-2
STAGES_TO_RUN = ["1-2"]
SEEDS_TO_RUN  = [42, 123, 2024]
```

### Fase 3 — Análise (~10 min)

Quando os 3 notebooks de treino terminarem (ou mesmo parcialmente):

1. Abra `04_analysis.ipynb`
2. Rode tudo de cima a baixo
3. Os artefatos finais ficam em `mario_drl_results/`:
   - `metrics/*.csv` → tabelas para inserir no artigo
   - `plots/*.png` → figuras
   - `plots/comparison_*.gif` → comparação DQN/PPO/A2C lado-a-lado
   - `plots/evolution_*.gif` → evolução temporal do agente

A análise **funciona com dados parciais** — se só DQN e PPO terminaram, os plots mostram só essas duas curvas. Tudo via groupby.

## Quando reusar `04_analysis` durante o treino

Pode (e deve) abrir o `04_analysis.ipynb` ao longo do caminho — mesmo com 30% dos treinos prontos, ele gera resultados parciais e ajuda a:
- Diagnosticar se um algoritmo está convergindo mal
- Decidir se vale rodar mais seeds
- Validar que o pipeline de métricas está correto

## Estrutura de saída compartilhada

Todos os 4 notebooks acessam `mario_drl_results/` via o mesmo bloco de path config (célula 3 de cada um):

```python
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT_DIR = Path("/content/drive/MyDrive/mario_drl_results")
except ImportError:
    ROOT_DIR = Path("./mario_drl_results").resolve()
```

Layout:
```
mario_drl_results/
├── models/                            # *.zip — modelos finais
│   └── checkpoints/                   # *_{N}_steps.zip — para evolução temporal
├── logs/                              # *.csv — métricas por episódio de avaliação
├── tensorboard/                       # logs do TensorBoard
├── metrics/                           # CSVs consolidados (gerados por 04_analysis)
└── plots/                             # PNGs + GIFs (gerados por 04_analysis)
```

## Dicas práticas

- **Conexão estável é essencial em Colab.** Use a extensão "Colab Auto Reconnect" ou rode local se possível.
- **TensorBoard ao vivo**: `%load_ext tensorboard` + `%tensorboard --logdir mario_drl_results/tensorboard` em qualquer um dos notebooks.
- **Mudar fase ou seed para visualizar**: nas demos do `04_analysis` (células 14.1 e 14.2), ajuste `stage_for_render` e `seed_for_render` para uma combinação que esteja treinada.
- **Recuperar de uma queda no meio do treino**: nada a fazer além de reabrir o notebook. `train_one` é idempotente.
