"""
Plots para o artigo — um arquivo PNG por gráfico.

Saídas (em PLOTS_DIR):

  Curvas de aprendizado (1 PNG por fase):
    learning_curve_stage1-1.png
    learning_curve_stage1-2.png
    learning_curve_stage4-1.png
    learning_curve_stage8-1.png

  Grupo I — comparação entre arquiteturas (1 PNG por métrica):
    group1_R_final.png        — Recompensa final média
    group1_AUC.png            — AUC normalizada
    group1_t80.png            — Timesteps até 80% do máximo

  Grupo II — avaliação de dificuldade (1 PNG por métrica):
    group2_tau.png            — Taxa de conclusão
    group2_d_bar.png          — Distância normalizada
    group2_deaths.png         — Mortes por episódio
    group2_time.png           — Tempo até conclusão (frames)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import STAGES, PLOTS_DIR


PALETTE = {"DQN": "#1f77b4", "PPO": "#ff7f0e", "A2C": "#2ca02c"}

# Tamanho padrão de figura individual (largura, altura em polegadas)
FIGSIZE_INDIVIDUAL = (6, 4.5)


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper")


# ============================================================================
# CURVAS DE APRENDIZADO — 1 PNG por fase
# ============================================================================
def plot_learning_curve_single_stage(
    df_all: pd.DataFrame,
    stage: str,
    savedir: Path = PLOTS_DIR,
    figsize: tuple = FIGSIZE_INDIVIDUAL,
) -> Path | None:
    """Curva de aprendizado de uma fase específica (mediana + IQR sobre seeds)."""
    _setup_style()
    sub = df_all[df_all["stage"] == stage]
    if sub.empty:
        print(f"  ⚠ sem dados para fase {stage}")
        return None

    fig, ax = plt.subplots(figsize=figsize)
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

    ax.set_title(f"Curva de aprendizado — Fase {stage}")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Recompensa (mediana ± IQR)")
    ax.legend(title="Algoritmo", loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / f"learning_curve_stage{stage}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def plot_learning_curves(df_all: pd.DataFrame, savedir: Path = PLOTS_DIR) -> list[Path]:
    """Gera 1 PNG por fase. Retorna lista de paths gerados."""
    if df_all.empty:
        print("Sem dados.")
        return []
    out_paths = []
    for stage in sorted(df_all["stage"].unique()):
        p = plot_learning_curve_single_stage(df_all, stage, savedir)
        if p:
            out_paths.append(p)
    return out_paths


# ============================================================================
# GRUPO I — 1 PNG por métrica (bar chart algo × fase)
# ============================================================================
GROUP1_METRICS = {
    "R_final": "Recompensa final média",
    "AUC":     "AUC normalizada",
    "t80":     "Timesteps até 80% do máximo",
}


def plot_group1_single_metric(
    metrics_g1: pd.DataFrame,
    metric: str,
    savedir: Path = PLOTS_DIR,
    figsize: tuple = FIGSIZE_INDIVIDUAL,
) -> Path | None:
    """Bar chart de UMA métrica do Grupo I (R_final, AUC ou t80) por fase."""
    _setup_style()
    if metrics_g1.empty or metric not in metrics_g1.columns:
        print(f"  ⚠ sem dados ou métrica '{metric}' não encontrada")
        return None

    title = GROUP1_METRICS.get(metric, metric)

    fig, ax = plt.subplots(figsize=figsize)
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
    out = savedir / f"group1_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def plot_group1_comparison(metrics_g1: pd.DataFrame, savedir: Path = PLOTS_DIR) -> list[Path]:
    """Gera 1 PNG por métrica do Grupo I."""
    if metrics_g1.empty:
        print("Sem dados.")
        return []
    out_paths = []
    for metric in GROUP1_METRICS:
        p = plot_group1_single_metric(metrics_g1, metric, savedir)
        if p:
            out_paths.append(p)
    return out_paths


# ============================================================================
# GRUPO II — 1 PNG por métrica (linha vs ordem canônica de dificuldade)
# ============================================================================
GROUP2_METRICS = {
    "tau":         "Taxa de conclusão (τ)",
    "d_bar":       "Distância normalizada (d̄)",
    "deaths":      "Mortes por episódio",
    "time_frames": "Tempo até conclusão (frames)",
}


def plot_group2_single_metric(
    metrics_g2: pd.DataFrame,
    metric: str,
    savedir: Path = PLOTS_DIR,
    figsize: tuple = FIGSIZE_INDIVIDUAL,
) -> Path | None:
    """Plot de UMA métrica do Grupo II vs ordem canônica de dificuldade."""
    _setup_style()
    if metrics_g2.empty or metric not in metrics_g2.columns:
        print(f"  ⚠ sem dados ou métrica '{metric}' não encontrada")
        return None

    title = GROUP2_METRICS.get(metric, metric)
    stage_order = [s for s in STAGES if s in metrics_g2["stage"].unique()]
    if not stage_order:
        print(f"  ⚠ nenhuma fase válida")
        return None

    fig, ax = plt.subplots(figsize=figsize)

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

    ax.set_xticks(np.arange(len(stage_order)))
    ax.set_xticklabels(stage_order)
    ax.set_xlabel("Fase (ordem canônica de dificuldade →)")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(title="Algoritmo")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    # Para evitar nomes de arquivo confusos, mantemos `time` para `time_frames`
    metric_for_filename = "time" if metric == "time_frames" else metric
    out = savedir / f"group2_{metric_for_filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")
    return out


def plot_group2_difficulty(metrics_g2: pd.DataFrame, savedir: Path = PLOTS_DIR) -> list[Path]:
    """Gera 1 PNG por métrica do Grupo II."""
    if metrics_g2.empty:
        print("Sem dados.")
        return []
    out_paths = []
    for metric in GROUP2_METRICS:
        p = plot_group2_single_metric(metrics_g2, metric, savedir)
        if p:
            out_paths.append(p)
    return out_paths


# ============================================================================
# PIPELINE — gera todos os PNGs (individuais)
# ============================================================================
def generate_all_plots(results: dict, savedir: Path = PLOTS_DIR) -> dict:
    """
    Recebe o dict retornado por `run_full_analysis()` e gera todos os PNGs
    individuais. Retorna dict com listas de paths por categoria.
    """
    out = {}
    print("\n--- Curvas de aprendizado (1 por fase) ---")
    out["learning_curves"] = plot_learning_curves(results["df_all"], savedir)

    print("\n--- Grupo I — comparação (1 por métrica) ---")
    out["group1"] = plot_group1_comparison(results["metrics_g1"], savedir)

    print("\n--- Grupo II — dificuldade (1 por métrica) ---")
    out["group2"] = plot_group2_difficulty(results["metrics_g2"], savedir)

    n_total = sum(len(v) for v in out.values())
    print(f"\n✓ Total de {n_total} PNG(s) gerado(s) em {savedir}")
    return out
