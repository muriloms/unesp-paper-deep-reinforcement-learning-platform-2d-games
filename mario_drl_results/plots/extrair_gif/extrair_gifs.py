#!/usr/bin/env python3
# =====================================================================
#  extrair_gifs.py
#  Extrai os frames dos GIFs e gera, para cada um, a linha
#  \animategraphics pronta para colar na apresentacao Beamer.
#
#  COMO USAR:
#    1) Coloque este arquivo na MESMA pasta onde estao os .gif
#       (agent_DQN_stage1-1_seed42.gif, comparison_algos_stage1-1_seed42.gif, ...)
#    2) Rode:   python3 extrair_gifs.py
#    3) Os frames vao para figuras/gifs/<NOME>/000.png, 001.png, ...
#    4) Copie a pasta figuras/ inteira para o Overleaf.
#    5) Cole no .tex as linhas \animategraphics impressas no terminal,
#       substituindo os \includegraphics estaticos do slide qualitativo.
#
#  Requisito: Pillow  (pip install Pillow)  -- ja vem no Colab/Anaconda.
#  A animacao so REPRODUZ no Adobe Acrobat Reader (no preview do
#  Overleaf/Chrome aparece o 1o frame -- que ja serve de imagem estatica).
# =====================================================================

import os
from PIL import Image, ImageSequence

# Quais GIFs extrair e com que "apelido" de pasta (chave -> arquivo .gif).
# Edite aqui se quiser outra fase/seed. seed 42 e a usada no artigo.
ALVOS = {
    # tres agentes separados, fase 1-1, lado a lado no slide
    "DQN_1-1": "agent_DQN_stage1-1_seed42.gif",
    "A2C_1-1": "agent_A2C_stage1-1_seed42.gif",
    "PPO_1-1": "agent_PPO_stage1-1_seed42.gif",

    # OPCAO ALTERNATIVA: um unico GIF ja comparando os 3 algoritmos.
    # Descomente para extrair tambem (e use uma unica \animategraphics no slide).
    # "comparison_1-1": "comparison_algos_stage1-1_seed42.gif",
}

FPS = 15          # taxa de reproducao no PDF (ajuste se quiser mais lento/rapido)
SAIDA = "figuras/gifs"

print("=" * 70)
linhas_tex = []
for apelido, arquivo in ALVOS.items():
    if not os.path.isfile(arquivo):
        print(f"[AVISO] nao encontrei: {arquivo}  -> pulando")
        continue

    destino = os.path.join(SAIDA, apelido)
    os.makedirs(destino, exist_ok=True)

    im = Image.open(arquivo)
    n = 0
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        frame.convert("RGB").save(os.path.join(destino, f"{i:03d}.png"))
        n = i
    ultimo = n  # ultimo indice (0-based)

    # caminho usado no \animategraphics (relativo ao .tex no Overleaf)
    prefixo = f"figuras/gifs/{apelido}/"
    linha = (f"\\animategraphics[autoplay,loop,width=\\textwidth]"
             f"{{{FPS}}}{{{prefixo}}}{{000}}{{{ultimo:03d}}}")
    linhas_tex.append((apelido, n + 1, linha))
    print(f"[OK] {arquivo}: {n+1} frames -> {destino}/")

print("=" * 70)
print("\nCole estas linhas no slide qualitativo (no lugar dos \\includegraphics):\n")
for apelido, qtd, linha in linhas_tex:
    print(f"% {apelido}  ({qtd} frames)")
    print(f"      {linha}\n")
print("Lembrete: abra o PDF no Adobe Acrobat Reader para ver a animacao.")
