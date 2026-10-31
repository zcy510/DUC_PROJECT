"""
Phase 3: turn results/experiments/*.csv into figures and summary tables.

  fig_rq1_error_vs_epsilon.png   central DP error vs eps, per query and N
  fig_rq2_central_vs_local.png   central vs local DP (category shares) vs eps
  fig_rq3_bounds.png             bias vs noise vs total error for Linf and L0
  fig_rq4_runtime.png            runtime vs N: baseline, central DP, local DP
  summary_*.csv                  the numbers behind every figure (mean, 95% CI)

Every point is the mean over repetitions; shaded bands are 95% confidence
intervals (t-distribution).

Usage:
  python src/plot_results.py                                  # results/experiments
  python src/plot_results.py --results results/experiments_quick
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # no window needed; just write PNG files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# Validated categorical palette (fixed order, never cycled) + recessive ink.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
MARKERS = ["o", "s", "^"]                         # second encoding besides colour
INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#fcfcfb"
SIZE_LABEL = {"n10k": "N = 10k", "n100k": "N = 100k", "n1m": "N = 1M", "n10m": "N = 10M"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def ci_summary(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    """Mean and 95% CI half-width of `col` per group."""
    def agg(x):
        x = x.dropna()
        n = len(x)
        half = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return pd.Series({"mean": x.mean(), "ci95": half, "n_reps": n})
    return df.groupby(by)[col].apply(agg).unstack().reset_index()


def line(ax, x, s, color, marker, label, linestyle="-"):
    ax.plot(x, s["mean"], color=color, marker=marker, label=label, linestyle=linestyle)
    ax.fill_between(x, s["mean"] - s["ci95"], s["mean"] + s["ci95"], color=color, alpha=0.15, lw=0)


def sizes_in(df):
    return [s for s in SIZE_LABEL if s in set(df["dataset"])]


# ------------------------------------------------------------------ RQ1

def fig_rq1(runs, out):
    g = runs[(runs.experiment == "grid") & (runs.model == "central")]
    if g.empty:
        return
    queries = [("q1_MRE", "Q1 · events per category"), ("q2_MRE", "Q2 · users per category"),
               ("q3_MRE", "Q3 · events per category × hour")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    rows = []
    for ax, (col, title) in zip(axes, queries):
        s = ci_summary(g, ["dataset", "epsilon"], col)
        s["metric"] = col
        rows.append(s)
        for i, d in enumerate(sizes_in(g)):
            si = s[s.dataset == d].sort_values("epsilon")
            line(ax, si["epsilon"], si, SLOT[i], MARKERS[i], SIZE_LABEL[d])
        ax.set(xscale="log", yscale="log", title=title, xlabel="privacy budget ε (log)")
        ax.set_xticks(sorted(g.epsilon.unique()), [f"{e:g}" for e in sorted(g.epsilon.unique())])
    axes[0].set_ylabel("mean relative error (log)")
    axes[0].legend(loc="lower left")
    fig.suptitle("RQ1 · Central DP: error vs privacy budget", color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_rq1_error_vs_epsilon.png", dpi=200)
    pd.concat(rows).to_csv(out / "summary_rq1.csv", index=False)
    plt.close(fig)


# ------------------------------------------------------------------ RQ2

def fig_rq2(runs, out):
    g = runs[runs.experiment == "grid"]
    if g.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for ax, (col, title, logy) in zip(axes, [
            ("q1share_MRE", "Mean relative error of category shares", True),
            ("q1share_spearman", "Ranking quality (Spearman ρ)", False)]):
        s = ci_summary(g, ["model", "dataset", "epsilon"], col)
        s["metric"] = col
        rows.append(s)
        for i, d in enumerate(sizes_in(g)):
            for model, ls in (("central", "-"), ("local", "--")):
                si = s[(s.dataset == d) & (s.model == model)].sort_values("epsilon")
                if not si.empty:
                    line(ax, si["epsilon"], si, SLOT[i], MARKERS[i],
                         f"{model} · {SIZE_LABEL[d]}", linestyle=ls)
        ax.set(xscale="log", yscale="log" if logy else "linear", title=title,
               xlabel="privacy budget ε (log)")
        ax.set_xticks(sorted(g.epsilon.unique()), [f"{e:g}" for e in sorted(g.epsilon.unique())])
    axes[0].set_ylabel("MRE (log)")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_ylim(top=1.02)
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("RQ2 · Central (solid) vs local (dashed) DP at the same ε and N",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_rq2_central_vs_local.png", dpi=200)
    pd.concat(rows).to_csv(out / "summary_rq2.csv", index=False)
    plt.close(fig)


# ------------------------------------------------------------------ RQ3

def expected_bounded_counts(events: pd.DataFrame, l0: int, linf: int) -> pd.Series:
    """EXACT expected Q1 count per category after PipelineDP's contribution
    bounding (no noise), computed from the raw events:

      - per category, a user keeps min(events, Linf) events;
      - a user active in k categories keeps L0 of them, chosen uniformly at
        random, so each category survives with probability min(1, L0 / k).

    expected count(c) = sum over users of min(n_uc, Linf) * min(1, L0 / k_u)
    """
    uc = events.groupby(["user_id", "category"]).size().rename("n").reset_index()
    k = uc.groupby("user_id")["category"].transform("size")
    kept = np.minimum(uc["n"], linf) * np.minimum(1.0, l0 / k)
    return kept.groupby(uc["category"]).sum()


def bias_noise(cells: pd.DataFrame, runs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Decompose the Q1 error of each bound setting, as mean relative error per category:

      bias  = |expected bounded count - true| / true      (exact, from the data)
      noise = E|Laplace| / true = (noise_std / sqrt 2) / true   (exact, from Delta/eps)
      total = measured mean |DP - true| / true over all reps   (= MRE, empirical)
    """
    rows = []
    for (l0, linf), d in cells.groupby(["l0", "linf"]):
        truth = d.groupby("category")["true"].first()
        truth = truth[truth > 0]
        bounded = expected_bounded_counts(events, l0, linf).reindex(truth.index, fill_value=0)
        noise_std = runs[(runs.l0 == l0) & (runs.linf == linf)]["q1_noise_std"].iloc[0]
        total = (d.assign(rel=d.error.abs() / d.true)[d.true > 0]
                 .groupby("rep")["rel"].mean())  # MRE per run
        rows.append({"l0": l0, "linf": linf,
                     "bias": float(((bounded - truth).abs() / truth).mean()),
                     "noise": float(((noise_std / np.sqrt(2)) / truth).mean()),
                     "total": float(total.mean()),
                     "total_ci95": float(stats.t.ppf(0.975, len(total) - 1) * total.std(ddof=1)
                                         / np.sqrt(len(total))) if len(total) > 1 else 0.0,
                     "n_reps": int(len(total))})
    return pd.DataFrame(rows)


def fig_rq3(cells, runs, data_root, out):
    b = cells[cells.experiment == "bounds"] if not cells.empty else cells
    if b.empty:
        return
    ds = b.dataset.iloc[0]
    events = pd.read_parquet(data_root / ds / "events", columns=["user_id", "category"])
    s_all = bias_noise(b, runs[runs.experiment == "bounds"], events)
    s_all.to_csv(out / "summary_rq3.csv", index=False)
    panels = [("linf", s_all[s_all.l0 == 4], "L∞", "L∞ = max events per user per category  (L₀ = 4)"),
              ("l0", s_all[s_all.linf == 10], "L₀", "L₀ = max categories per user  (L∞ = 10)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (param, s, sym, title) in zip(axes, panels):
        if s[param].nunique() < 2:
            ax.set_visible(False)
            continue
        s = s.sort_values(param)
        for i, (col, label) in enumerate([("bias", "bias: data dropped by bounding"),
                                          ("noise", "noise: expected |Laplace|"),
                                          ("total", "total: measured error (MRE)")]):
            ax.plot(s[param], s[col], color=SLOT[i], marker=MARKERS[i], label=label)
        ax.fill_between(s[param], s.total - s.total_ci95, s.total + s.total_ci95,
                        color=SLOT[2], alpha=0.15, lw=0)
        best = s.loc[s["total"].idxmin()]
        ax.annotate(f"lowest total error\n{sym} = {best[param]:g}", (best[param], best["total"]),
                    textcoords="offset points", xytext=(0, 28), ha="center", color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
        ax.set(xscale="log", title=title, xlabel=f"{sym} (log scale)")
        ax.set_xticks(s[param], [f"{v:g}" for v in s[param]])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("mean relative error per category")
    axes[0].legend(loc="upper center")
    fig.suptitle(f"RQ3 · Contribution bounds: bias vs noise  (central DP, ε = 1, {SIZE_LABEL.get(ds, ds)})",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_rq3_bounds.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ RQ4

def fig_rq4(runs, out):
    base = runs[runs.experiment == "baseline"]
    grid = runs[(runs.experiment == "grid") & (runs.epsilon == 1)]
    series = [("baseline", base, "baseline (no privacy)"),
              ("central", grid[grid.model == "central"], "central DP (PipelineDP)"),
              ("local", grid[grid.model == "local"], "local DP (devices + server)")]
    if base.empty and grid.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = []
    for i, (name, d, label) in enumerate(series):
        if d.empty:
            continue
        s = ci_summary(d, ["n_users"], "seconds").sort_values("n_users")
        s["series"] = name
        rows.append(s)
        line(ax, s["n_users"], s, SLOT[i], MARKERS[i], label)
    ax.set(xscale="log", yscale="log", xlabel="number of users N (log)",
           ylabel="seconds per run (log)", title="RQ4 · Runtime vs dataset size (ε = 1)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_rq4_runtime.png", dpi=200)
    summary = pd.concat(rows)
    # Overhead factor = DP time / baseline time at the same N.
    if not base.empty:
        b = summary[summary.series == "baseline"].set_index("n_users")["mean"]
        summary["overhead_vs_baseline"] = summary.apply(
            lambda r: r["mean"] / b.get(r["n_users"], np.nan), axis=1)
    summary.to_csv(out / "summary_rq4.csv", index=False)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/experiments"))
    ap.add_argument("--data", type=Path, default=Path("data"), help="datasets (RQ3 needs raw events)")
    args = ap.parse_args()
    runs = pd.read_json(args.results / "runs.jsonl", lines=True)
    cells_path = args.results / "cells_q1.csv"
    cells = pd.read_csv(cells_path) if cells_path.exists() else pd.DataFrame()
    # A run interrupted between writing its cells and its run line is redone,
    # which can leave duplicate cells: keep the last copy only.
    keys = ["experiment", "model", "dataset", "epsilon", "l0", "linf", "rep"]
    runs = runs.drop_duplicates(subset=keys, keep="last")
    if not cells.empty:
        cells = cells.drop_duplicates(subset=keys + ["category"], keep="last")
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_rq1(runs, out)
    fig_rq2(runs, out)
    fig_rq3(cells, runs, args.data, out)
    fig_rq4(runs, out)
    print(f"Figures and summary tables written to {out}/")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
