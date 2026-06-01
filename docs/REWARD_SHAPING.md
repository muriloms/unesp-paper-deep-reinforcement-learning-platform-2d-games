# Reward Shaping — diagnóstico e solução

## O problema observado

Após treinar PPO/stage1-1 com 500.000 timesteps no protocolo baseline, a inspeção dos CSVs revelou convergência para uma política trivial:

```
timestep,episode,reward,max_x_pos,flag_get,deaths,frames,time_left
5000,    0,       -415,  40,        0,       0,     2005,   0
10000,   0,       -455,  39,        0,       0,     2005,   0
...
500000,  0,       +250,  314,       0,       0,     30,     394
```

Mario:
- Não avançou além do pixel 314 (9,6% da fase, comprimento total 3266)
- Não morreu nenhuma vez
- Episódios terminam em ~30 passos (não chega nem na primeira ameaça)
- Reward idêntico em todas as 5 avaliações (determinístico travado)

## Diagnóstico

**Convergência prematura para mínimo local trivial.** Sequência:

1. **Fase 1 (0-30k timesteps):** Mario fica imóvel o jogo inteiro (`frames=2005`, `time_left=0`). Reward muito negativo (`-415`) porque ficar parado acumula penalidade de tempo.
2. **Fase 2 (>30k):** Mario "aprende" a evitar o pior dos cenários movendo-se pouco, atinge `max_x=314` e congela ali. Reward sobe para +250, mas comportamento continua trivial.

**Causa raiz:** o reward nativo do `gym-super-mario-bros` é truncado em `[-15, +15]` por frame e dominado por penalidades de tempo. Com `gamma=0.99` (horizonte de desconto ≈ 100 frames), o PPO não consegue propagar o sinal "ir mais longe" através de eventos esparsos como passar do primeiro Goomba ou completar a fase. O gradiente da política favorece soluções triviais de "não fazer nada de pior".

## Solução implementada: `ProgressRewardWrapper`

Wrapper que adiciona ao reward nativo um sinal contínuo proporcional ao **avanço em `x_pos`**, com penalidade clara por morte e bonus por completar a fase:

```
r_total = r_nativo
        + progress_coef * (x_pos_t - x_pos_{t-1})    # default 0.1
        - death_penalty * death_event                  # default 15
        + completion_bonus * flag_get_event            # default 50
```

**Por que isso resolve:**
- Cada passo dá um sinal claro: avançar para a direita = positivo, recuar = negativo
- A escala (`0.1` × pixels por frame) dá ~0.5-1.0 de bonus por frame de progresso real, comparável à magnitude do reward nativo truncado
- O gradiente do PPO agora "puxa" Mario para a direita desde os primeiros gradient updates

**Arquivos modificados:**
- `src/wrappers.py` — novo módulo com `ProgressRewardWrapper`
- `src/env.py` — `make_mario_env` e `make_vec_env_mario` aceitam `reward_shaping=True`
- `src/training.py` — `train_one` e `run_matrix` propagam o flag; cria artefatos com sufixo `_shape`
- `src/config.py` — flag global `USE_REWARD_SHAPING`
- `scripts/train.py` — nova flag `--reward-shaping`

## Como usar

### Treino com reward shaping

```bash
# PPO/1-1/seed42 com shaping
python scripts/train.py --profile full --algos PPO --stages 1-1 --seeds 42 --reward-shaping

# Toda a matriz com shaping (cria 36 modelos "_shape" sem mexer no baseline)
python scripts/train.py --profile full --reward-shaping

# Smoke test rápido para validar o wrapper (~2 min)
python scripts/train.py --profile smoke --reward-shaping
```

### Separação de artefatos

Para não sobrescrever o baseline, runs com shaping ganham sufixo `_shape`:

| Sem shaping | Com shaping |
|---|---|
| `models/PPO_stage1-1_seed42.zip` | `models/PPO_stage1-1_seed42_shape.zip` |
| `logs/PPO_stage1-1_seed42.csv` | `logs/PPO_stage1-1_seed42_shape.csv` |
| `models/checkpoints/PPO_stage1-1_seed42_*_steps.zip` | `models/checkpoints/PPO_stage1-1_seed42_shape_*_steps.zip` |

**Importante:** o reward gravado no CSV continua sendo o reward **nativo** do jogo (sem shaping), porque o `MarioEvalCallback` cria env de avaliação separado sem o wrapper. Isso garante comparação justa entre baseline e shaped.

## O que esperar do CSV com shaping ativo

Se o fix funcionar, os primeiros 10-50k timesteps devem mostrar `max_x_pos` subindo continuamente:

```
timestep,episode,reward,max_x_pos,flag_get,deaths,frames,time_left
5000,    0,        +50,  450,       0,       1,     180,   350
10000,   0,       +120,  720,       0,       2,     240,   320
20000,   0,       +200,  1100,      0,       3,     310,   280
50000,   0,       +400,  1850,      0,       2,     420,   210
100000,  0,       +500,  2400,      0,       1,     510,   170
200000,  0,       +800,  3266,      1,       0,     520,   180
```

Sinais positivos para procurar:
- `max_x_pos` crescendo timestep a timestep
- `deaths` > 0 (Mario está tentando, não parado)
- `frames` aumentando (episódios mais longos = mais exploração)
- `flag_get=1` aparecendo em algum momento

## Roteiro recomendado

Para o paper, sugiro:

1. **Manter o baseline atual como evidência negativa** — mostra que o protocolo "puro" (sem shaping) não converge em 500k timesteps no Mario. É um achado válido.
2. **Rodar shaping no PPO/1-1 só para validar o fix:**
   ```bash
   python scripts/train.py --profile full --algos PPO --stages 1-1 --reward-shaping
   ```
   Tempo: ~15h × 1 algo × 1 fase × 3 seeds = ~5h
3. **Se funcionar (Mario completa 1-1 com shaping), rodar a matriz completa shaped:**
   ```bash
   python scripts/train.py --profile full --reward-shaping
   ```
4. **No artigo, comparar baseline vs shaped** na Seção 3 — esse é o tipo de análise rigorosa que diferencia trabalhos sérios. Mostra:
   - Limitação do reward nativo do env
   - Importância do reward design em DRL aplicado a jogos
   - Resultados positivos onde antes não havia nada

## Comparação na análise

O `scripts/analyze.py` atual não distingue runs `_shape` automaticamente. Para a análise comparativa, você pode rodar análise duas vezes apontando para subsets diferentes via `MARIO_DRL_ROOT`:

```bash
# Baseline
MARIO_DRL_ROOT=./mario_drl_results python scripts/analyze.py

# Shaped (se mover os _shape para outra pasta)
mkdir -p ./mario_drl_shaped/logs
cp mario_drl_results/logs/*_shape.csv ./mario_drl_shaped/logs/
# rename para tirar o _shape (analyze.py espera padrão exp_id_stageX_seedY)
for f in ./mario_drl_shaped/logs/*_shape.csv; do
    mv "$f" "${f/_shape/}"
done
MARIO_DRL_ROOT=./mario_drl_shaped python scripts/analyze.py
```

Alternativa mais limpa que posso implementar depois: passar `--variant shape` para o `analyze.py` filtrar apenas runs com aquele sufixo, gerar tabelas separadas e plots comparativos baseline-vs-shaped.

## Parâmetros do wrapper (se precisar tunar)

Em `src/wrappers.py`:

```python
ProgressRewardWrapper(
    env,
    progress_coef    = 0.1,    # quanto mais alto, mais agressivo o sinal de progresso
    death_penalty    = 15.0,   # alinhado com truncamento do reward nativo
    completion_bonus = 50.0,   # bonus único ao alcançar a flag
)
```

Se Mario aprender mas ficar travado em um local específico (ex: salto difícil), aumente `progress_coef` para `0.2`. Se Mario virar suicida (corre na direção e morre), aumente `death_penalty` para `30`.

## Risco a notar

Reward shaping pode ser visto como **"engenharia de reward"** — alguns reviewers de RL podem criticar isso como "ajustar o jogo até funcionar". Recomendo abordar no paper:

- Citar literatura: reward shaping no Mario é prática estabelecida (Kauten 2018; PPO original em Atari também usa clipping similar)
- Apresentar baseline E shaped: mostra rigor metodológico
- Discutir explicitamente o tradeoff: comparabilidade entre algoritmos é mantida porque todos os 3 (DQN, PPO, A2C) recebem o mesmo shaping
