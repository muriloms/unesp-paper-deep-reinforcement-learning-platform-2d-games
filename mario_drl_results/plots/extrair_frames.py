#!/usr/bin/env python3
"""
Extrai um quadro representativo de cada GIF de rollout e salva como PNG,
para inclusão no artigo LaTeX (que não renderiza GIF animado).

Uso:
    pip install pillow
    # rode de dentro de mario_drl_results/plots/
    python extrair_frames.py

Gera: figuras/rollouts/rollout_<ALGO>_stage<FASE>.png  (uma seed, 12 arquivos)

O quadro escolhido fica perto do FIM do episódio (FRAME_FRAC), que ~ corresponde
ao ponto mais distante alcançado pelo agente — coerente com a métrica d̄, já que
nenhum agente conclui as fases. Ajuste SEED ou FRAME_FRAC se quiser outro recorte.
"""
import os
from PIL import Image, ImageSequence

# --------- configuração ---------
SEED        = 42          # seed única usada na montagem (42, 123 ou 2024)
FRAME_FRAC  = 0.90        # posição do quadro no episódio (0=início, 1=fim)
ALGOS       = ["DQN", "A2C", "PPO"]
STAGES      = ["1-1", "1-2", "4-1", "8-1"]
SRC_DIR     = "."                       # onde estão os .gif
OUT_DIR     = "figuras/rollouts"        # destino dos .png
# --------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def pick_frame(gif_path, frac):
    """Retorna um quadro RGB perto da fração 'frac' do GIF, evitando quadro final
    eventualmente em branco (tela de morte)."""
    im = Image.open(gif_path)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    n = len(frames)
    if n == 0:
        raise ValueError("GIF sem quadros")
    idx = min(int(n * frac), n - 1)
    # se o quadro escolhido for quase todo preto, recua até achar um não-vazio
    for j in range(idx, -1, -1):
        extrema = frames[j].convert("L").getextrema()  # (min, max)
        if extrema[1] > 20:   # tem conteúdo visível
            return frames[j]
    return frames[idx]

ok, faltando = 0, []
for algo in ALGOS:
    for stage in STAGES:
        src = os.path.join(SRC_DIR, f"agent_{algo}_stage{stage}_seed{SEED}.gif")
        if not os.path.exists(src):
            faltando.append(src)
            continue
        frame = pick_frame(src, FRAME_FRAC)
        out = os.path.join(OUT_DIR, f"rollout_{algo}_stage{stage}.png")
        frame.save(out)
        ok += 1
        print(f"  ok  {src}  ->  {out}")

print(f"\n{ok} quadros extraídos para {OUT_DIR}/ (seed {SEED}).")
if faltando:
    print("ATENÇÃO: GIFs não encontrados:")
    for f in faltando:
        print("   -", f)
