#!/usr/bin/env python3
"""
Gera um relatório do estado de cada experimento: quantos timesteps já foram
treinados, se o modelo final existe, e o que falta para um alvo.

Lê o manifesto (progress.json) + verifica os arquivos no disco. Escreve um
arquivo de texto legível (status.txt) e também imprime na tela.

Uso:
    # Status contra um alvo de 2M
    python scripts/status.py --target 2000000

    # Salva em arquivo específico
    python scripts/status.py --target 2000000 --out status.txt

    # Só imprime, não salva
    python scripts/status.py --target 2000000 --no-save
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.training import load_progress, _find_latest_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Relatório de status dos treinos.")
    p.add_argument("--target", type=int, default=2_000_000,
                   help="Alvo de timesteps p/ calcular o que falta (default 2M)")
    p.add_argument("--out", default="status.txt", help="Arquivo de saída")
    p.add_argument("--no-save", action="store_true", help="Só imprime, não salva")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_progress()

    # Coleta todos os exp_ids conhecidos (manifesto ∪ modelos no disco)
    exp_ids = set(manifest.keys())
    if config.MODELS_DIR.exists():
        for m in config.MODELS_DIR.glob("*.zip"):
            exp_ids.add(m.stem)

    if not exp_ids:
        print("Nenhum experimento encontrado (manifesto vazio e sem modelos).")
        return 1

    lines = []
    lines.append("=" * 78)
    lines.append(f"STATUS DOS TREINOS — alvo: {args.target:,} timesteps")
    lines.append(f"Diretório: {config.ROOT_DIR}")
    lines.append("=" * 78)
    lines.append("")
    header = f"{'experimento':<34} {'treinado':>11} {'modelo':>7} {'falta':>11}  estado"
    lines.append(header)
    lines.append("-" * 78)

    n_done = 0
    n_partial = 0
    n_missing_model = 0

    for exp_id in sorted(exp_ids):
        manifest_steps = 0
        entry = manifest.get(exp_id)
        if isinstance(entry, dict):
            manifest_steps = int(entry.get("timesteps", 0))
        elif isinstance(entry, int):
            manifest_steps = entry

        _, ckpt_steps = _find_latest_checkpoint(exp_id)
        trained = max(manifest_steps, ckpt_steps)

        model_path = config.MODELS_DIR / f"{exp_id}.zip"
        has_model = model_path.exists()

        remaining = max(args.target - trained, 0)

        if trained >= args.target and has_model:
            estado = "✓ completo"
            n_done += 1
        elif not has_model:
            estado = "⚠ sem modelo final"
            n_missing_model += 1
        else:
            estado = f"↻ retomável (de {trained:,})"
            n_partial += 1

        model_tag = "sim" if has_model else "NÃO"
        lines.append(
            f"{exp_id:<34} {trained:>11,} {model_tag:>7} {remaining:>11,}  {estado}"
        )

    lines.append("-" * 78)
    lines.append(
        f"Total: {len(exp_ids)} | "
        f"completos: {n_done} | "
        f"retomáveis: {n_partial} | "
        f"sem modelo: {n_missing_model}"
    )
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)

    if not args.no_save:
        out_path = config.ROOT_DIR / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nRelatório salvo em: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
