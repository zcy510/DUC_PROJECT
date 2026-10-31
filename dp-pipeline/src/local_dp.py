"""
Local differential privacy with k-ary randomised response (Phase 2, Option A).

Here the server is NOT trusted: noise is added on each device before any
data leaves it. Pipeline:

  device (simulated, NumPy)          ingestion        server (Spark)
  ----------------------------       ---------        -----------------------------
  pick <= m of my events        ->   Parquet     ->   count reports per category c_i
  randomise each category                              unbiased estimate:
  with k-RR at eps/m                                   c_hat_i = (c_i - n*q) / (p - q)

k-ary randomised response (project plan, section 4.6), with k categories:
  report the TRUE category       with probability  p = e^eps / (e^eps + k - 1)
  report one specific OTHER one  with probability  q = 1     / (e^eps + k - 1)

USER-LEVEL privacy: a user may have hundreds of events. Randomising each one
with eps would leak eps * (#events) in total. So each user sends at most
m reports, each with eps/m  ->  total eps per user (sequential composition),
the same guarantee as the central model.

What it estimates: the number of REPORTS per category, i.e. the category
distribution of users' (sampled) events. Its share per category is
comparable to the share from central DP (Phase 3, RQ2).

Usage:
  python src/local_dp.py --data data/n100k --out results/local/n100k --epsilon 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import generate_data as gen
from metrics import compare


def krr_probabilities(eps: float, k: int) -> tuple[float, float]:
    """(p, q) for k-ary randomised response."""
    denom = np.exp(eps) + k - 1
    return float(np.exp(eps) / denom), float(1.0 / denom)


def krr_perturb(values: np.ndarray, eps: float, k: int, rng) -> np.ndarray:
    """Randomise category indices 0..k-1 (vectorised: one entry per device report)."""
    p, _ = krr_probabilities(eps, k)
    keep = rng.random(values.size) < p
    # A uniformly random OTHER value: draw from 0..k-2 and skip over the true one.
    other = rng.integers(0, k - 1, values.size)
    other = other + (other >= values)
    return np.where(keep, values, other)


def krr_estimate(observed: np.ndarray, eps: float, k: int) -> np.ndarray:
    """Unbiased estimate of the true counts from the observed (noisy) counts."""
    p, q = krr_probabilities(eps, k)
    n = observed.sum()
    return (observed - n * q) / (p - q)


def simulate_devices(events: pd.DataFrame, eps: float, m: int, seed: int,
                     categories=None) -> tuple[pd.DataFrame, np.ndarray]:
    """What every phone does before uploading.

    Returns (reports, true_report_counts): the uploaded noisy reports, and,
    for testing only, how many reports per category there were BEFORE noise
    (a real server can never see this).
    """
    categories = list(categories or gen.CATEGORIES)
    k = len(categories)
    rng = np.random.default_rng(seed)

    # 1) Each user picks at most m of their events uniformly at random.
    shuffled = events.iloc[rng.permutation(len(events))]
    picked = shuffled[shuffled.groupby("user_id").cumcount() < m]

    # 2) Randomise each picked category with eps/m.
    true_idx = pd.Categorical(picked["category"], categories=categories).codes.astype(np.int64)
    noisy_idx = krr_perturb(true_idx, eps / m, k, rng)

    reports = pd.DataFrame({"category": pd.Categorical.from_codes(noisy_idx, categories).astype(str)})
    true_counts = np.bincount(true_idx, minlength=k)
    return reports, true_counts


def spark_estimate(spark, reports_dir: Path, eps_per_report: float, categories=None) -> pd.DataFrame:
    """Server side, on Spark: count reports per category, then de-bias."""
    from pyspark.sql import functions as F

    categories = list(categories or gen.CATEGORIES)
    observed = (spark.read.parquet(str(reports_dir))
                .groupBy("category").agg(F.count("*").alias("observed"))
                .toPandas().set_index("category")["observed"]
                .reindex(categories, fill_value=0))  # categories nobody reported -> 0
    est = krr_estimate(observed.to_numpy(float), eps_per_report, len(categories))
    return pd.DataFrame({"category": categories, "observed": observed.to_numpy(),
                         "est_reports": est, "est_share": est / est.sum()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=1.0, help="TOTAL budget per user")
    ap.add_argument("--reports-per-user", type=int, default=1, help="m: each report gets eps/m")
    ap.add_argument("--seed", type=int, default=0, help="device randomness (vary it across runs)")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--shuffle-partitions", type=int, default=8)
    args = ap.parse_args()
    m, eps = args.reports_per_user, args.epsilon

    # --- Devices -------------------------------------------------------
    t0 = time.perf_counter()
    events = pd.read_parquet(args.data / "events", columns=["user_id", "category"])
    reports, true_report_counts = simulate_devices(events, eps, m, args.seed)
    reports_dir = args.out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports.to_parquet(reports_dir / "part-00000.parquet", index=False)
    device_seconds = round(time.perf_counter() - t0, 3)

    # --- Server (Spark) ------------------------------------------------
    from baseline_job import make_spark
    spark = make_spark(args.master, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("ERROR")
    spark.range(1).count()  # warm-up
    t0 = time.perf_counter()
    result = spark_estimate(spark, reports_dir, eps / m)
    server_seconds = round(time.perf_counter() - t0, 3)
    spark.stop()

    # --- Evaluation (uses the ground truth; not part of the mechanism) --
    truth = pd.read_csv(args.data / "ground_truth" / "events_per_category.csv")
    truth = truth.set_index("category").reindex(gen.CATEGORIES, fill_value=0).reset_index()
    truth["true_share"] = truth["n_events"] / truth["n_events"].sum()
    result["true_reports"] = true_report_counts
    result.to_csv(args.out / "events_per_category.csv", index=False)

    p, q = krr_probabilities(eps / m, len(gen.CATEGORIES))
    summary = {
        "epsilon": eps, "reports_per_user": m, "eps_per_report": eps / m,
        "p_true": round(p, 5), "q_other": round(q, 5), "n_reports": int(len(reports)),
        "device_seconds": device_seconds, "server_seconds": server_seconds,
        "accuracy": {
            # did the estimator recover what devices really had? (noise only)
            "reports_vs_true_reports": compare(result, result, ["category"], "est_reports", "true_reports"),
            # how well does the LDP share match the true event share? (noise + sampling)
            "share_vs_true_event_share": compare(result, truth, ["category"], "est_share", "true_share"),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
