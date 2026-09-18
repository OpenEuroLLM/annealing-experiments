"""Plot multilingual benchmark averages across annealing token budgets.

Adapted directly from https://github.com/OpenEuroLLM/notebooks/blob/main/prelude-evals-wip/analyse_multilingual.py
"""
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

CATEGORICAL_COLORS = [
    "#2B7EEA",
    "#E83F3F",
    "#0EC12C",
    "#656E7C",
]
GRIDLINE_COLOR = "#dbdee4"

# bleu/chrf++ live on a different (0-100) scale and would distort the
# average; acc/acc_norm/exact_match are all in [0, 1] so they mix fine
ACCURACY_METRICS = {"acc", "acc_norm", "exact_match"}

# PolyMath is excluded since it's not suited for base pretrained models
EXCLUDED_TASKS = ["PolyMath"]
# XCOPA and FLORES are only partially evaluated across these checkpoints.
EXCLUDED_TASKS.extend(["XCOPA", "Flores-200"])

# tokens per training iteration = seq_len * global_batch_size
TOKENS_PER_ITER = {
    "openeurollm/datamix-9b-80-20": 2048 * 2048,
    "openeurollm/prelude-checkpoints": 4096 * 2048,
    "allenai/Olmo-3-1025-7B": 4_194_304,
}

APERTUS_TOKENS_RE = re.compile(r"tokens(\d+(?:\.\d+)?)([BT])")
ITER_RE = re.compile(r"iter_(\d+)")
OLMO_STEP_RE = re.compile(r"stage1-step(\d+)")

def compute_tokens_b(row):
    data, iter_ = row["data"], row["iter"]
    if data == "allenai/Olmo-3-1025-7B":
        m = OLMO_STEP_RE.search(iter_)
        if not m:
            return None
        return int(m.group(1)) * TOKENS_PER_ITER[data] / 1e9
    if data in TOKENS_PER_ITER:
        m = ITER_RE.search(iter_)
        if not m:
            return None
        return int(m.group(1)) * TOKENS_PER_ITER[data] / 1e9
    if data == "swiss-ai/Apertus-8B-2509":
        m = APERTUS_TOKENS_RE.search(iter_)
        if not m:
            return None
        value = float(m.group(1))
        return value * 1000 if m.group(2) == "T" else value
    return None


def average_downstream_performance(df, value_col="score"):
    """Average scores over languages within each benchmark first (e.g. all
    arc_challenge_mt_* tasks -> one ARC Challenge_mt score), then average
    over benchmarks to get a single downstream performance number per
    (data, iter, tokens_B)."""
    per_benchmark = df.groupby(["data", "iter", "tokens_B", "benchmark"], dropna=False)[value_col].mean()
    return (
        per_benchmark.groupby(["data", "iter", "tokens_B"], dropna=False)
        .mean()
        .rename("avg_score")
        .reset_index()
    )


def minmax_normalize_scores(df):
    """Min-max normalize each (benchmark, metric) pair to [0, 1] across the
    whole dataset (all current metrics -- acc, acc_norm, exact_match, bleu,
    chrf++ -- are higher-is-better). This puts every metric on a comparable
    scale before averaging across benchmarks."""
    df = df.copy()
    stats = df.groupby(["benchmark", "metric"])["score"].agg(lo="min", hi="max")
    df = df.join(stats, on=["benchmark", "metric"])
    span = df["hi"] - df["lo"]
    df["score_norm"] = ((df["score"] - df["lo"]) / span.replace(0, pd.NA)).fillna(0.5)
    return df.drop(columns=["lo", "hi"])


FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelude", "evals")


def plot_performance_vs_tokens(
    avg_df,
    benchmark_df,
    output_name,
    ylabel,
    title,
    value_col,
):
    benchmarks = [(name, average_downstream_performance(group, value_col=value_col))
                  for name, group in benchmark_df.groupby("benchmark")] if benchmark_df is not None else []
    models = avg_df["data"].str.extract(r"_anneal\d+b_(.+)$")[0].dropna().sort_values().unique()
    colors = {model: CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i, model in enumerate(models)}
    columns = 4
    rows = (len(benchmarks) + columns - 1) // columns
    fig = plt.figure(figsize=(3 * columns, 7 + 2 * rows))
    grid = fig.add_gridspec(rows + 1, columns, height_ratios=[2.5] + [1] * rows)
    for panel, (panel_title, scores) in enumerate([(title, avg_df)] + benchmarks):
        is_main = panel == 0
        ax = fig.add_subplot(grid[0, :] if is_main else grid[1 + (panel - 1) // columns, (panel - 1) % columns])
        with_tokens = scores.dropna(subset=["tokens_B"])
        origin = with_tokens[with_tokens["tokens_B"] == 0]
        with_tokens = with_tokens.assign(model=with_tokens["data"].str.extract(r"_anneal\d+b_(.+)$")[0])

        for data, group in with_tokens.groupby("model"):
            group = group.sort_values("tokens_B")
            # Connect the shared origin to the final checkpoint of each budget.
            group = pd.concat([origin, group.groupby("data").tail(1)])
            ax.plot(
                group["tokens_B"],
                group["avg_score"],
                marker="o",
                markevery=slice(len(origin), None),
                color=colors[data],
                label=data,
                linewidth=3.2 if is_main else 2,
                markersize=7 if is_main else 5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )

        if not origin.empty:
            ax.scatter(origin["tokens_B"], origin["avg_score"], color="0.3",
                       marker="D", label="Before annealing", zorder=4)

        ax.set_xlabel("Tokens trained during annealing (billions)" if is_main else "Tokens (billions)",
                      fontsize=12 if is_main else 10, labelpad=12)
        panel_ylabel = "Normalized score" if value_col == "score_norm" else "Average score"
        ax.set_ylabel(ylabel.replace(" (", "\n(") if is_main else panel_ylabel, fontsize=12 if is_main else 10,
                      labelpad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_title(panel_title, loc="center", fontsize=15 if is_main else 12,
                     fontweight="semibold", pad=18)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=11 if is_main else 9, pad=8)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRIDLINE_COLOR, linewidth=0.8)
        if is_main:
            ax.legend(loc="best", fontsize=10, frameon=False)
        else:
            ax.set_xticks([0, 20, 50, 100])
    fig.tight_layout(h_pad=3, w_pad=2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{output_name}.{ext}")
        fig.savefig(path, dpi=150)
        print(f"\nWrote {path}")
    plt.close(fig)


# Local task prefixes -> the benchmark names used by the original script.
BENCHMARKS = {
    "arc_challenge_mt": "ARC Challenge MT",
    "belebele": "Belebele",
    "flores200": "Flores-200",
    "global_mgsm": "Global MGSM",
    "global_mmlu_full": "Global MMLU",
    "global_piqa": "Global PIQA",
    "hellaswag": "xHellaswag",
    "include_base_44": "INCLUDE",
    "mgsm_native_cot": "MGSM",
    "multiblimp": "MultiBLiMP",
    "opensubtitles_multi40": "OpenSubtitles",
    "polymath": "PolyMath",
    "sib200": "SIB-200",
    "xcopa": "XCOPA",
    "xcsqa": "X-CSQA",
}


def main():
    dfs = [pd.read_csv(path) for path in sorted(glob(os.path.join(FIGURES_DIR, "*bt_multilingual", "eval_results.csv")))]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(["model_name", "task", "n_shot", "metric_name"])
    df = df.rename(columns={"performance": "score"})
    df["data"] = df["model_name"].str.extract(r"/hf_models/([^/]+)", expand=False)
    df["iter"] = df["model_name"].str.extract(r"(iter_\d+)", expand=False)
    df["metric"] = df["metric_name"].str.split(",").str[0]
    df["benchmark"] = df["task"].str.extract("^(" + "|".join(BENCHMARKS) + ")", expand=False).map(BENCHMARKS)
    TOKENS_PER_ITER.update(dict.fromkeys(df["data"].unique(), TOKENS_PER_ITER["openeurollm/prelude-checkpoints"]))
    df = df[~df["benchmark"].isin(EXCLUDED_TASKS)]
    df["tokens_B"] = df.apply(compute_tokens_b, axis=1)
    # These runs all start from the same checkpoint before annealing.
    df["tokens_B"] -= 953312 * TOKENS_PER_ITER["openeurollm/prelude-checkpoints"] / 1e9

    accuracy_df = df[df["metric"].isin(ACCURACY_METRICS)]
    avg_df = average_downstream_performance(accuracy_df)
    plot_performance_vs_tokens(
        avg_df,
        accuracy_df,
        output_name="multilingual_summary",
        ylabel="Average downstream performance",
        title="Multilingual annealing ablation: average multilingual performance",
        value_col="score",
    )

    normalized_df = minmax_normalize_scores(df)
    avg_norm_df = average_downstream_performance(normalized_df, value_col="score_norm")
    plot_performance_vs_tokens(
        avg_norm_df,
        normalized_df,
        output_name="normalized_overall_performance",
        ylabel="Min-max normalized average performance",
        title="Multilingual annealing ablation: min-max average multilingual performance",
        value_col="score_norm",
    )


if __name__ == "__main__":
    main()
