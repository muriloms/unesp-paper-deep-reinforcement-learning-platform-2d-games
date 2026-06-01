# Guia de execução — `scripts/train.py`

Referência completa de combinações para executar treinos específicos. Útil para fatiar o trabalho em sessões, retomar de quedas e refazer treinos pontuais.

## Flags disponíveis

| Flag | Valores possíveis | Default | Função |
|---|---|---|---|
| `--profile` | `smoke` \| `full` | `smoke` | Smoke = validação rápida (~2 min); full = experimento real (~60h) |
| `--algos` | `DQN` `PPO` `A2C` (1 ou mais) | todos | Filtra quais algoritmos treinar |
| `--stages` | `1-1` `1-2` `4-1` `8-1` (1 ou mais) | conforme perfil | Filtra quais fases |
| `--seeds` | `42` `123` `2024` (1 ou mais inteiros) | conforme perfil | Filtra quais seeds |
| `--overwrite` | (flag) | desativado | Ignora estado prévio e retreina do zero |
| `--no-check-python` | (flag) | desativado | Desativa checagem de Python 3.10/3.11 |

**Idempotência:** sem `--overwrite`, treinos já completos são pulados, e treinos parciais (com checkpoint) **retomam de onde pararam**.

## Universo de combinações possíveis

| Algoritmos | Fases | Seeds | Total de treinos |
|---|---|---|---|
| 3 (DQN, PPO, A2C) | 4 (1-1, 1-2, 4-1, 8-1) | 3 (42, 123, 2024) | **36** |

## Tabela de receitas prontas

### Validação e exploração rápida

| Cenário | Comando | Tempo aprox. |
|---|---|---|
| Smoke test (validar pipeline) | `python scripts/train.py --profile smoke` | ~2 min |
| Smoke só de um algoritmo | `python scripts/train.py --profile smoke --algos DQN` | ~2 min |

### Treino completo

| Cenário | Comando | Tempo aprox. |
|---|---|---|
| Tudo (matriz completa) | `python scripts/train.py --profile full` | ~60h |
| Tudo, retreinando do zero | `python scripts/train.py --profile full --overwrite` | ~60h |

### Por algoritmo (ideal para sessões paralelas)

| Cenário | Comando | Tempo aprox. |
|---|---|---|
| Apenas DQN (12 treinos) | `python scripts/train.py --profile full --algos DQN` | ~30h |
| Apenas PPO (12 treinos) | `python scripts/train.py --profile full --algos PPO` | ~15h |
| Apenas A2C (12 treinos) | `python scripts/train.py --profile full --algos A2C` | ~15h |
| DQN + PPO (24 treinos) | `python scripts/train.py --profile full --algos DQN PPO` | ~45h |
| PPO + A2C (24 treinos) | `python scripts/train.py --profile full --algos PPO A2C` | ~30h |

### Por fase

| Cenário | Comando | Total |
|---|---|---|
| Só a fase 1-1 (9 treinos) | `python scripts/train.py --profile full --stages 1-1` | 3 algos × 1 fase × 3 seeds |
| Só a fase 4-1 | `python scripts/train.py --profile full --stages 4-1` | 9 treinos |
| Fases 1-1 e 1-2 (mundo 1) | `python scripts/train.py --profile full --stages 1-1 1-2` | 18 treinos |
| Fases 4-1 e 8-1 (avançadas) | `python scripts/train.py --profile full --stages 4-1 8-1` | 18 treinos |
| Todas exceto 8-1 | `python scripts/train.py --profile full --stages 1-1 1-2 4-1` | 27 treinos |

### Por seed

| Cenário | Comando | Total |
|---|---|---|
| Só seed 42 (12 treinos) | `python scripts/train.py --profile full --seeds 42` | 3 algos × 4 fases × 1 seed |
| Seeds 42 e 123 (24 treinos) | `python scripts/train.py --profile full --seeds 42 123` | |
| Seed específica nova | `python scripts/train.py --profile full --seeds 999` | 12 treinos com seed 999 |

### Combinações refinadas

| Cenário | Comando |
|---|---|
| DQN só na fase 1-1, todas as seeds | `python scripts/train.py --profile full --algos DQN --stages 1-1` |
| PPO + A2C nas fases avançadas (4-1, 8-1) | `python scripts/train.py --profile full --algos PPO A2C --stages 4-1 8-1` |
| Só o DQN/1-1/seed42 | `python scripts/train.py --profile full --algos DQN --stages 1-1 --seeds 42` |
| Refazer DQN/4-1/seed42 do zero | `python scripts/train.py --profile full --algos DQN --stages 4-1 --seeds 42 --overwrite` |
| Apenas seed 42 em toda matriz | `python scripts/train.py --profile full --seeds 42` |
| Todos os algoritmos, fase 1-1, seed 42 (3 treinos) | `python scripts/train.py --profile full --stages 1-1 --seeds 42` |

### Retomar após queda

Não precisa fazer nada especial — basta repetir o comando original. O script detecta o último checkpoint salvo de cada treino e continua dali.

| Cenário | Comando |
|---|---|
| Cai durante DQN/4-1/123, retomar tudo | `python scripts/train.py --profile full` (vai pular o que está completo e retomar o parcial) |
| Forçar retreinar **apenas** o que estava em curso | `python scripts/train.py --profile full --algos DQN --stages 4-1 --seeds 123 --overwrite` |

## Estratégias de execução recomendadas

### Estratégia 1 — Pipeline conservadora (recomendada para 1 GPU)

Roda sequencial, dispara um algoritmo de cada vez no mesmo terminal:

```bash
python scripts/train.py --profile full --algos PPO    # ~15h
python scripts/train.py --profile full --algos A2C    # ~15h
python scripts/train.py --profile full --algos DQN    # ~30h
```

Total: ~60h, mas previsível. Boa para deixar rodando à noite.

### Estratégia 2 — Paralela com `nohup` (mais rápida, mesma GPU compartilhada)

Em 3 terminais paralelos, **mesma máquina**, mesma GPU sendo compartilhada por PyTorch:

```bash
# Terminal 1
nohup python scripts/train.py --profile full --algos DQN > dqn.log 2>&1 &

# Terminal 2 (depois de uns segundos para DQN ocupar VRAM primeiro)
nohup python scripts/train.py --profile full --algos PPO > ppo.log 2>&1 &

# Terminal 3
nohup python scripts/train.py --profile full --algos A2C > a2c.log 2>&1 &
```

Acompanhe com `tail -f dqn.log` (ou os outros). **Cuidado:** se a soma da VRAM exceder a GPU disponível, vai estourar. Monitore com `nvidia-smi`.

### Estratégia 3 — Por fase (priorizar fases fáceis primeiro para resultados rápidos)

```bash
python scripts/train.py --profile full --stages 1-1   # 9 treinos
# já dá pra rodar análise parcial enquanto continua
python scripts/train.py --profile full --stages 1-2 4-1 8-1
```

### Estratégia 4 — Por seed (todas as fases/algos com 1 seed por vez)

```bash
python scripts/train.py --profile full --seeds 42     # 12 treinos
python scripts/train.py --profile full --seeds 123    # 12 treinos
python scripts/train.py --profile full --seeds 2024   # 12 treinos
```

Vantagem: tem sempre o conjunto completo (3 algos × 4 fases) de uma seed antes de partir para a próxima. Permite **análise parcial mais robusta**.

## Casos especiais

### Refazer um único treino que deu errado

Suponha que `PPO_stage8-1_seed2024` tenha valores estranhos no CSV e você quer rerodar do zero:

```bash
python scripts/train.py --profile full --algos PPO --stages 8-1 --seeds 2024 --overwrite
```

Os outros 35 treinos ficam intocados.

### Adicionar uma seed nova depois (replicação extra)

Para fortalecer a análise estatística, posso querer mais uma seed:

```bash
python scripts/train.py --profile full --seeds 999
```

Vai rodar 3 algos × 4 fases × seed 999 = 12 treinos novos, sem mexer nos antigos. Depois, em `src/config.py`, adicione `999` em `SEEDS` para a análise capturar.

### Smoke test com configuração customizada

Smoke test default usa `--algos PPO --stages 1-1 --seeds 42` (definido em `PROFILES["smoke"]` no `src/config.py`). Para testar outra combinação rapidamente:

```bash
python scripts/train.py --profile smoke --algos DQN --stages 4-1
```

Roda DQN/4-1/seed42 com apenas 10k timesteps (~2 min).

## Como monitorar enquanto roda

### TensorBoard ao vivo

Em outro terminal, com o venv ativo:

```bash
tensorboard --logdir mario_drl_results/tensorboard
```

Abra `http://localhost:6006`. Vai ver curvas de loss, reward médio, exploração (DQN), todas atualizando em tempo real.

### CSV incremental dos eval episodes

Os CSVs em `mario_drl_results/logs/` são escritos **incrementalmente** (a cada chamada de avaliação). Pode inspecioná-los sem parar o treino:

```bash
tail -f mario_drl_results/logs/DQN_stage1-1_seed42.csv
```

### Análise parcial

Mesmo enquanto os treinos ainda rodam, pode gerar métricas e plots do que já existe:

```bash
python scripts/analyze.py
```

Funciona com dados parciais.

## Resumo rápido

- **Filtros:** `--algos`, `--stages`, `--seeds` aceitam um ou mais valores
- **Combinação:** os filtros se multiplicam — total = `len(algos) × len(stages) × len(seeds)`
- **Sem filtros:** roda a matriz inteira do perfil escolhido
- **Idempotente:** sem `--overwrite`, pula completos e retoma parciais
- **`--overwrite`:** força retreinar do zero (ignora tudo, inclusive checkpoints)



python scripts/render.py agent --algo PPO --stage 1-1 --seed 42 --variant shape

# Apaga o modelo ruim e retreina do zero
python scripts/train.py --profile long --algos PPO --reward-shaping --overwrite

# Em outro terminal, ver o CSV crescendo a cada eval (~10k timesteps)
tail -f mario_drl_results/logs/PPO_stage1-1_seed42_shape.csv

# 1) Olhar últimas avaliações
tail -20 mario_drl_results/logs/PPO_stage1-1_seed42_shape.csv

# 2) Gerar GIF
python scripts/render.py agent --algo PPO --stage 1-1 --seed 42 --variant shape

# 3) Teste estocástico (igual ao anterior, pra ter certeza que não colapsou)
python3 << 'PY'
import sys; sys.path.insert(0, '.')
from stable_baselines3 import PPO
from src.config import MODELS_DIR
from src.visualization import _make_render_env
from src.utils import get_device
model = PPO.load(MODELS_DIR / "PPO_stage1-1_seed42_shape.zip", device=get_device())
env = _make_render_env("1-1")
obs = env.reset()
xs = []
for step in range(2000):
    action, _ = model.predict(obs, deterministic=False)
    obs, r, done, info = env.step(action)
    xs.append(int(info[0].get("x_pos", 0)))
    if done[0]: break
env.close()
print(f"max_x={max(xs)}, final={xs[-1]}, steps={len(xs)}")
PY

# Apaga o modelo ruim e retreina do zero
python scripts/train.py --profile full --algos PPO --reward-shaping --overwrite