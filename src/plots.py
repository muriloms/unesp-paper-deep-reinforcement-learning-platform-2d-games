"""
Plots para o artigo:
- learning_curves.png        — curvas de aprendizado (mediana + IQR) por fase
- group1_comparison.png      — R_final, AUC, t80 por algoritmo × fase
- group2_difficulty.png      — τ, d̄, mortes vs ordem canônica de dificuldade
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import STAGES, PLOTS_DIR


PALETTE = {"DQN": "#1f77b4", "PPO": "#ff7f0e", "A2C": "#2ca02c"}


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper")


def plot_learning_curves(df_all: pd.DataFrame, savedir: Path = PLOTS_DIR) -> Path | None:
    """Curva de aprendizado por fase, com mediana + IQR sobre seeds."""
    _setup_style()
    if df_all.empty:
        print("Sem dados.")
        return None

    stages_present = sorted(df_all["stage"].unique())
    n = len(stages_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages_present):
        sub = df_all[df_all["stage"] == stage]
        for algo, g in sub.groupby("algo"):
            agg = (
                g.groupby("timestep")["reward"]
                .agg(["median",
                      lambda x: x.quantile(0.25),
                      lambda x: x.quantile(0.75)])
                .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
                .reset_index()
            )
            color = PALETTE.get(algo)
            ax.plot(agg["timestep"], agg["median"], label=algo, color=color, linewidth=2)
            ax.fill_between(agg["timestep"], agg["q25"], agg["q75"], alpha=0.2, color=color)
        ax.set_title(f"Fase {stage}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Recompensa (mediana ± IQR)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def plot_group1_comparison(metrics_g1: pd.DataFrame, savedir: Path = PLOTS_DIR) -> Path | None:
    """Bar chart de R_final, AUC, t80 por algoritmo × fase."""
    _setup_style()
    if metrics_g1.empty:
        print("Sem dados.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, title in zip(
        axes,
        ["R_final", "AUC", "t80"],
        ["Recompensa final média", "AUC normalizada", "Timesteps até 80% do máximo"],
    ):
        sns.barplot(
            data=metrics_g1, x="stage", y=metric, hue="algo",
            palette=PALETTE, ax=ax, errorbar=("ci", 95),
            order=sorted(metrics_g1["stage"].unique()),
        )
        ax.set_title(title)
        ax.set_xlabel("Fase")
        ax.set_ylabel(metric)
        ax.legend(title="Algoritmo")

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / "group1_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def plot_group2_difficulty(metrics_g2: pd.DataFrame, savedir: Path = PLOTS_DIR) -> Path | None:
    """Métricas Grupo II vs ordem canônica das fases."""
    _setup_style()
    if metrics_g2.empty:
        print("Sem dados.")
        return None

    metric_titles = {
        "tau":    "Taxa de conclusão (τ)",
        "d_bar":  "Distância normalizada (d̄)",
        "deaths": "Mortes por episódio",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    stage_order = [s for s in STAGES if s in metrics_g2["stage"].unique()]

    for ax, (metric, title) in zip(axes, metric_titles.items()):
        for algo, g in metrics_g2.groupby("algo"):
            agg = g.groupby("stage")[metric].agg([
                "median",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
            ])
            agg.columns = ["median", "q25", "q75"]
            agg = agg.reindex(stage_order)
            x = np.arange(len(stage_order))
            ax.plot(x, agg["median"], "o-", label=algo, linewidth=2,
                    markersize=8, color=PALETTE.get(algo))
            ax.fill_between(x, agg["q25"], agg["q75"], alpha=0.2,
                            color=PALETTE.get(algo))
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_xlabel("Fase (ordem canônica de dificuldade →)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(title="Algoritmo")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / "group2_difficulty.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def generate_all_plots(results: dict, savedir: Path = PLOTS_DIR):
    """
    Recebe o dict retornado por `run_full_analysis()` e gera todos os PNGs.
    """
    out_paths = {}
    out_paths["learning_curves"] = plot_learning_curves(results["df_all"], savedir)
    out_paths["group1_comparison"] = plot_group1_comparison(results["metrics_g1"], savedir)
    out_paths["group2_difficulty"] = plot_group2_difficulty(results["metrics_g2"], savedir)
    return out_paths
