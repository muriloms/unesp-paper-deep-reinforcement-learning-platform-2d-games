"""
Cálculo das métricas Grupo I e Grupo II + análise estatística.

- Grupo I: comparação entre arquiteturas (R_final, AUC, t80, sigma_final)
- Grupo II: avaliação automática de dificuldade (tau, d_bar, deaths, time)
- Spearman: ranking do agente vs dificuldade canônica das fases
- Mann-Whitney U: pareado entre algoritmos por fase
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from .config import STAGE_LENGTH, STAGES, LOGS_DIR, METRICS_DIR


# ============================================================================
# CARREGAMENTO DOS LOGS
# ============================================================================
def _parse_log_name(stem: str):
    """
    Parseia stem do CSV. Aceita:
        DQN_stage1-1_seed42             → ("DQN", "1-1", 42, None)
        PPO_stage4-1_seed42_shape       → ("PPO", "4-1", 42, "shape")
        A2C_stage1-1_seed42_eval        → ("A2C", "1-1", 42, "eval")
        PPO_stage1-1_seed42_shape_eval  → ("PPO", "1-1", 42, "shape_eval")
    Retorna (algo, stage, seed, suffix_str_or_None) ou None se não bate.
    """
    import re
    m = re.match(r"^(DQN|PPO|A2C)_stage([^_]+)_seed(\d+)(?:_(.+))?$", stem)
    if not m:
        return None
    algo, stage, seed_str, suffix = m.groups()
    return algo, stage, int(seed_str), suffix


def load_all_logs(suffix_filter: str | None = "exclude_eval") -> pd.DataFrame:
    """
    Concatena todos os CSVs de LOGS_DIR num DataFrame longo com [algo, stage, seed].

    Args:
        suffix_filter: filtro de sufixo no nome do CSV.
            None              → carrega tudo (incluindo _shape, _eval, etc).
            "exclude_eval"    → ignora *_eval.csv (default).
            "eval_only"       → carrega APENAS *_eval.csv (CSVs com deaths corrigidos).
            "shape_only"      → carrega APENAS modelos com sufixo "shape" (qualquer variante).
            "no_suffix"       → carrega APENAS os baseline (sem nenhum sufixo).
            "<custom>"        → carrega APENAS sufixo exato (ex: "shape", "eval").
    """
    rows = []
    for csv_path in sorted(LOGS_DIR.glob("*.csv")):
        parsed = _parse_log_name(csv_path.stem)
        if parsed is None:
            print(f"  ⚠ ignorando {csv_path.name} (nome fora do padrão)")
            continue
        algo, stage, seed, suffix = parsed

        # Filtra conforme suffix_filter
        if suffix_filter == "exclude_eval":
            if suffix and "eval" in suffix:
                continue
        elif suffix_filter == "eval_only":
            if not suffix or "eval" not in suffix:
                continue
        elif suffix_filter == "shape_only":
            if not suffix or "shape" not in suffix:
                continue
        elif suffix_filter == "no_suffix":
            if suffix is not None:
                continue
        elif suffix_filter is not None and suffix_filter not in ("exclude_eval",):
            # Filtro exato (string específica)
            if suffix != suffix_filter:
                continue

        df = pd.read_csv(csv_path)
        df["algo"] = algo
        df["stage"] = stage
        df["seed"] = seed
        df["suffix"] = suffix or ""
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# GRUPO I — comparação entre arquiteturas
# ============================================================================
def compute_group1_metrics(df_all: pd.DataFrame, final_window: int = 50_000) -> pd.DataFrame:
    """
    Por (algo, stage, seed):
      - R_final     : média de recompensa nos últimos `final_window` timesteps
      - AUC         : integral normalizada da curva de aprendizado
      - t80         : timesteps até atingir 80% do máximo
      - sigma_final : desvio-padrão da recompensa nos últimos `final_window`
    """
    if df_all.empty:
        return pd.DataFrame()
    rows = []
    for (algo, stage, seed), g in df_all.groupby(["algo", "stage", "seed"]):
        g = g.sort_values("timestep")
        per_t = g.groupby("timestep")["reward"].mean().reset_index()
        per_t = per_t.sort_values("timestep")

        max_t = per_t["timestep"].max()
        max_r = per_t["reward"].max()

        final_mask = per_t["timestep"] >= (max_t - final_window)
        R_final = per_t.loc[final_mask, "reward"].mean()
        sigma_final = per_t.loc[final_mask, "reward"].std()

        auc = np.trapz(per_t["reward"], per_t["timestep"]) / max(max_t, 1)

        threshold = 0.8 * max_r
        above = per_t[per_t["reward"] >= threshold]
        t80 = int(above["timestep"].iloc[0]) if len(above) > 0 else np.nan

        rows.append(dict(
            algo=algo, stage=stage, seed=seed,
            R_final=R_final, AUC=auc, t80=t80, sigma_final=sigma_final,
        ))
    return pd.DataFrame(rows)


def aggregate_group1(metrics_g1: pd.DataFrame) -> pd.DataFrame:
    """Agrega Grupo I por (algo, stage): mediana, IQR."""
    return (metrics_g1
        .groupby(["algo", "stage"])
        .agg(
            R_final_median=("R_final", "median"),
            R_final_iqr=("R_final", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            AUC_median=("AUC", "median"),
            AUC_iqr=("AUC", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            t80_median=("t80", "median"),
            sigma_final_median=("sigma_final", "median"),
        )
        .reset_index())


# ============================================================================
# GRUPO II — avaliação de dificuldade
# ============================================================================
def compute_group2_metrics(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Por (algo, stage, seed), sobre os episódios da ÚLTIMA avaliação:
      - tau         : taxa de conclusão (média de flag_get)
      - d_bar       : distância normalizada (max_x_pos / STAGE_LENGTH)
      - deaths      : mortes médias
      - time_frames : tempo médio até conclusão (só ep. bem-sucedidos)
    """
    if df_all.empty:
        return pd.DataFrame()
    rows = []
    for (algo, stage, seed), g in df_all.groupby(["algo", "stage", "seed"]):
        last_t = g["timestep"].max()
        last_eval = g[g["timestep"] == last_t]

        tau = last_eval["flag_get"].mean()
        d_bar = (last_eval["max_x_pos"] / STAGE_LENGTH.get(stage, 3266)).mean()
        deaths_mean = last_eval["deaths"].mean()

        successful = last_eval[last_eval["flag_get"] == 1]
        time_mean = successful["frames"].mean() if len(successful) > 0 else np.nan

        rows.append(dict(
            algo=algo, stage=stage, seed=seed,
            tau=tau, d_bar=d_bar, deaths=deaths_mean, time_frames=time_mean,
        ))
    return pd.DataFrame(rows)


def aggregate_group2(metrics_g2: pd.DataFrame) -> pd.DataFrame:
    return (metrics_g2
        .groupby(["algo", "stage"])
        .agg(
            tau_median=("tau", "median"),
            tau_iqr=("tau", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            d_bar_median=("d_bar", "median"),
            deaths_median=("deaths", "median"),
            time_median=("time_frames", "median"),
        )
        .reset_index())


# ============================================================================
# ESTATÍSTICA — Spearman e Mann-Whitney
# ============================================================================
def spearman_difficulty(metrics_g2_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Correlação de Spearman entre o ranking de cada métrica do Grupo II
    e o ranking canônico das fases (1-1 → 1-2 → 4-1 → 8-1 = 1..4).

    Hipóteses:
        - tau, d_bar:   ρ → -1 (quanto mais difícil, menor a métrica)
        - deaths, time: ρ → +1 (quanto mais difícil, maior a métrica)
    """
    if metrics_g2_agg.empty:
        return pd.DataFrame()
    rank_canonical = {s: i for i, s in enumerate(STAGES, start=1)}
    rows = []
    for algo in metrics_g2_agg["algo"].unique():
        sub = metrics_g2_agg[metrics_g2_agg["algo"] == algo].copy()
        sub["rank_canonical"] = sub["stage"].map(rank_canonical)
        sub = sub.sort_values("rank_canonical")

        if len(sub) < 3:
            continue

        for metric, expected_sign in [
            ("tau_median",    "-"),
            ("d_bar_median",  "-"),
            ("deaths_median", "+"),
            ("time_median",   "+"),
        ]:
            x = sub["rank_canonical"].values
            y = sub[metric].values
            if np.all(np.isnan(y)):
                continue
            rho, p = spearmanr(x, y, nan_policy="omit")
            rows.append(dict(
                algo=algo, metric=metric, expected_sign=expected_sign,
                rho=rho, p_value=p, n=len(sub),
            ))
    return pd.DataFrame(rows)


def mannwhitney_between_algos(metrics_g1: pd.DataFrame, metric: str = "R_final") -> pd.DataFrame:
    """
    Teste pareado por fase entre todos os pares de algoritmos.
    """
    if metrics_g1.empty:
        return pd.DataFrame()
    rows = []
    for stage in metrics_g1["stage"].unique():
        sub = metrics_g1[metrics_g1["stage"] == stage]
        for a, b in combinations(sorted(sub["algo"].unique()), 2):
            xa = sub[sub["algo"] == a][metric].dropna().values
            xb = sub[sub["algo"] == b][metric].dropna().values
            if len(xa) < 2 or len(xb) < 2:
                continue
            stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
            rows.append(dict(
                stage=stage,
                comparison=f"{a} vs {b}",
                median_diff=float(np.median(xa) - np.median(xb)),
                U=stat, p_value=p, significant=bool(p < 0.05),
            ))
    return pd.DataFrame(rows)


# ============================================================================
# PIPELINE COMPLETO — gera todos os CSVs
# ============================================================================
def run_full_analysis(
    verbose: bool = True,
    suffix_filter: str | None = "exclude_eval",
    use_eval_for_group2: bool = False,
) -> dict:
    """
    Carrega logs, calcula todas as métricas e salva CSVs em METRICS_DIR.

    Args:
        suffix_filter: filtro principal (ver `load_all_logs()`).
            "exclude_eval" (default) → CSVs do treino, com trajetória completa.
            "eval_only"              → APENAS CSVs *_eval.csv (1 timestep só).
            "shape_only"             → APENAS modelos com sufixo "shape".

        use_eval_for_group2: se True, Grupo II e estatísticas relacionadas usam
            CSVs *_eval.csv (com deaths corrigidos), enquanto Grupo I e curvas
            de aprendizado continuam usando os CSVs originais do treino
            (que têm trajetória completa). Solução correta para análise final:
              - Grupo I e curvas precisam de trajetória (50+ pontos)
              - Grupo II só usa a última avaliação, então faz sentido pegar
                a versão com deaths corrigido.

    Returns:
        dict com os DataFrames principais.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # DataFrame principal — trajetória do treino (para curvas e Grupo I)
    df_all = load_all_logs(suffix_filter=suffix_filter)
    if df_all.empty:
        if verbose:
            print("Nenhum log encontrado em", LOGS_DIR)
            print(f"  (filtro de sufixo: {suffix_filter!r})")
        return {"df_all": df_all}

    if verbose:
        n_cfg = df_all.groupby(["algo", "stage", "seed"]).ngroups
        suf_info = ""
        if "suffix" in df_all.columns:
            suffixes = sorted(set(s for s in df_all["suffix"].unique() if s))
            if suffixes:
                suf_info = f"  [sufixos: {suffixes}]"
        print(f"Avaliações carregadas (trajetória): {len(df_all):,}  "
              f"({n_cfg} configurações){suf_info}")

    # Grupo I — sempre usa trajetória completa
    metrics_g1 = compute_group1_metrics(df_all)
    metrics_g1.to_csv(METRICS_DIR / "group1_per_seed.csv", index=False)
    metrics_g1_agg = aggregate_group1(metrics_g1)
    metrics_g1_agg.to_csv(METRICS_DIR / "group1_aggregated.csv", index=False)

    # Grupo II — opcionalmente usa CSVs *_eval.csv (deaths corrigidos)
    if use_eval_for_group2:
        df_eval = load_all_logs(suffix_filter="eval_only")
        if df_eval.empty:
            if verbose:
                print("⚠ use_eval_for_group2=True mas nenhum *_eval.csv encontrado.")
                print("  Rode `python scripts/recompute_metrics.py` antes.")
                print("  Caindo no fallback: usando trajetória para Grupo II também.")
            df_for_g2 = df_all
        else:
            if verbose:
                n_cfg_eval = df_eval.groupby(["algo", "stage", "seed"]).ngroups
                print(f"Avaliações carregadas (eval corrigido): {len(df_eval):,}  "
                      f"({n_cfg_eval} configurações)")
            df_for_g2 = df_eval
    else:
        df_for_g2 = df_all

    metrics_g2 = compute_group2_metrics(df_for_g2)
    metrics_g2.to_csv(METRICS_DIR / "group2_per_seed.csv", index=False)
    metrics_g2_agg = aggregate_group2(metrics_g2)
    metrics_g2_agg.to_csv(METRICS_DIR / "group2_aggregated.csv", index=False)

    # Estatísticas
    spearman_df = spearman_difficulty(metrics_g2_agg)
    spearman_df.to_csv(METRICS_DIR / "spearman_difficulty.csv", index=False)

    mw_df = mannwhitney_between_algos(metrics_g1, metric="R_final")
    mw_df.to_csv(METRICS_DIR / "mannwhitney_algos.csv", index=False)

    if verbose:
        print(f"✓ Métricas salvas em {METRICS_DIR}")

    return dict(
        df_all=df_all,
        metrics_g1=metrics_g1,
        metrics_g1_agg=metrics_g1_agg,
        metrics_g2=metrics_g2,
        metrics_g2_agg=metrics_g2_agg,
        spearman=spearman_df,
        mannwhitney=mw_df,
    )
