#!/usr/bin/env python3
"""
CLI de análise: carrega logs, calcula métricas e gera plots.

Idempotente: pode rodar mesmo com dados parciais — gera o que for possível
com o que estiver disponível.

Exemplos:
  # Análise completa (CSVs + PNGs)
  python scripts/analyze.py

  # Só métricas, sem plots
  python scripts/analyze.py --no-plots
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.analysis import run_full_analysis
from src.plots import generate_all_plots
from src.utils import setup_warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calcula métricas (Grupos I e II), Spearman, Mann-Whitney + plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-plots", action="store_true",
                   help="Calcula CSVs mas pula geração dos PNGs")
    p.add_argument("--quiet", action="store_true",
                   help="Reduz verbosidade")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_warnings()

    verbose = not args.quiet

    if verbose:
        print(f"Lendo logs de   : {config.LOGS_DIR}")
        print(f"Métricas em     : {config.METRICS_DIR}")
        print(f"Plots em        : {config.PLOTS_DIR}\n")

    config.ensure_dirs()

    results = run_full_analysis(verbose=verbose)

    if "df_all" not in results or results["df_all"].empty:
        print("⚠ Sem dados. Rode `scripts/train.py` antes.")
        return 1

    if not args.no_plots:
        if verbose:
            print()
        generate_all_plots(results, savedir=config.PLOTS_DIR)

    if verbose:
        print(f"\n✓ Análise completa em {config.ROOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
