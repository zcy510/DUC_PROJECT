"""
Central differential privacy with PipelineDP on Spark (Phase 2, Option B).

The server sees the raw events (it is trusted), and releases the three
baseline statistics with an (epsilon, 0)-DP guarantee PER USER:

  Q1  events per category          (COUNT)
  Q2  distinct users per category  (PRIVACY_ID_COUNT)
  Q3  events per category and hour (COUNT)

What PipelineDP does for us (project plan, section 4.5):
  1. Contribution bounding: each user keeps at most L0 partitions
     (categories) and at most Linf events per partition; the rest is
     randomly sampled away. -> sensitivity  Delta = L0 * Linf.
  2. Laplace noise with scale Delta / epsilon_i added to every count,
     using Google's secure noise generation (no floating-point leaks).
  3. Public partitions: the 50 categories (and 24 hours) are known in
     advance, so no private partition selection is needed, and every
     category appears in the output even if nobody used it.
  4. Budget accounting: the total epsilon is split over all released
     statistics (sequential composition), so the WHOLE release is eps-DP.

Usage:
  python src/central_dp.py --data data/n10k --out results/central/n10k --epsilon 1
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pipeline_dp
from pipeline_dp.report_generator import ExplainComputationReport

import generate_data as gen
from metrics import compare

# Rows are plain tuples: (user_id, category, hour)
USER, CATEGORY, HOUR = 0, 1, 2


def dp_aggregate(rows, backend, epsilon: float,
                 l0: int = 4, linf: int = 10,
                 l0_hour: int = 20, linf_hour: int = 2,
                 categories=None, noise=pipeline_dp.NoiseKind.LAPLACE):
    """Build the DP computation. Works with any PipelineDP backend:
    SparkRDDBackend (rows = RDD) or LocalBackend (rows = Python list).

    Returns (lazy results per query, explain reports, finalize function).
    Nothing runs until `finalize()` computes the budgets and you collect.
    """
    categories = list(categories or gen.CATEGORIES)

    # One accountant for the whole release -> total epsilon is respected.
    # delta = 0 because Laplace noise + public partitions give pure eps-DP.
    accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=0)
    engine = pipeline_dp.DPEngine(accountant, backend)
    reports = {"per_category": ExplainComputationReport(),
               "per_category_hour": ExplainComputationReport()}

    # --- Aggregation 1: Q1 + Q2, partition = category -------------------
    params_cat = pipeline_dp.AggregateParams(
        metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT],
        noise_kind=noise,
        max_partitions_contributed=l0,        # L0
        max_contributions_per_partition=linf,  # Linf
        output_noise_stddev=True)
    extract_cat = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda r: r[USER],   # WHO is protected
        partition_extractor=lambda r: r[CATEGORY],
        value_extractor=lambda r: 0)              # unused for counts
    per_cat = engine.aggregate(rows, params_cat, extract_cat,
                               public_partitions=categories,
                               out_explain_computation_report=reports["per_category"])

    # --- Aggregation 2: Q3, partition = (category, hour) ----------------
    params_hour = pipeline_dp.AggregateParams(
        metrics=[pipeline_dp.Metrics.COUNT],
        noise_kind=noise,
        max_partitions_contributed=l0_hour,
        max_contributions_per_partition=linf_hour,
        output_noise_stddev=True)
    extract_hour = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda r: r[USER],
        partition_extractor=lambda r: (r[CATEGORY], r[HOUR]),
        value_extractor=lambda r: 0)
    per_hour = engine.aggregate(rows, params_hour, extract_hour,
                                public_partitions=[(c, h) for c in categories for h in range(24)],
                                out_explain_computation_report=reports["per_category_hour"])

    # The budget is split only once all aggregations are declared.
    def finalize():
        accountant.compute_budgets()

    return {"per_category": per_cat, "per_category_hour": per_hour}, reports, finalize


def to_frames(per_cat_rows, per_hour_rows) -> dict:
    """Turn PipelineDP output [(partition, MetricsTuple), ...] into tables
    shaped like the baseline results."""
    cat = pd.DataFrame([{"category": k, "n_events": m.count, "n_users": m.privacy_id_count,
                         "events_noise_std": m.count_noise_stddev,
                         "users_noise_std": m.privacy_id_count_noise_stddev}
                        for k, m in per_cat_rows])
    hour = pd.DataFrame([{"category": k[0], "hour": k[1], "n_events": m.count,
                          "noise_std": m.count_noise_stddev}
                         for k, m in per_hour_rows])
    return {
        "events_per_category": cat[["category", "n_events", "events_noise_std"]],
        "users_per_category": cat[["category", "n_users", "users_noise_std"]],
        "events_per_category_hour": hour,
    }


def run_spark(spark, data_dir: Path, epsilon: float, **bounds):
    """Run central DP on Spark. Returns (tables, explain text, seconds)."""
    from pyspark.sql import functions as F

    events = spark.read.parquet(str(data_dir / "events"))
    events.count()  # warm-up, same as the baseline job, for fair timing
    t0 = time.perf_counter()
    # PipelineDP works on RDDs of Python objects -> convert rows to tuples.
    rows = (events.select("user_id", "category", F.hour("ts").alias("hour"))
            .rdd.map(tuple))
    backend = pipeline_dp.SparkRDDBackend(spark.sparkContext)
    lazy, reports, finalize = dp_aggregate(rows, backend, epsilon, **bounds)
    finalize()
    tables = to_frames(lazy["per_category"].collect(), lazy["per_category_hour"].collect())
    seconds = round(time.perf_counter() - t0, 3)
    explain = "\n\n".join(r.text() for r in reports.values())
    return tables, explain, seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=1.0, help="TOTAL budget for all 3 statistics")
    ap.add_argument("--l0", type=int, default=4, help="max categories per user")
    ap.add_argument("--linf", type=int, default=10, help="max events per user per category")
    ap.add_argument("--l0-hour", type=int, default=20, help="max (category,hour) cells per user")
    ap.add_argument("--linf-hour", type=int, default=2, help="max events per user per cell")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--shuffle-partitions", type=int, default=8)
    args = ap.parse_args()

    from baseline_job import make_spark
    spark = make_spark(args.master, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("ERROR")
    bounds = dict(l0=args.l0, linf=args.linf, l0_hour=args.l0_hour, linf_hour=args.linf_hour)
    tables, explain, seconds = run_spark(spark, args.data, args.epsilon, **bounds)
    spark.stop()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "explain.txt").write_text(explain)

    # Quick accuracy summary against the generator's ground truth.
    gt = args.data / "ground_truth"
    summary = {"epsilon": args.epsilon, **bounds, "dp_seconds": seconds, "accuracy": {
        "events_per_category": compare(tables["events_per_category"],
                                       pd.read_csv(gt / "events_per_category.csv"),
                                       ["category"], "n_events", "n_events"),
        "users_per_category": compare(tables["users_per_category"],
                                      pd.read_csv(gt / "users_per_category.csv"),
                                      ["category"], "n_users", "n_users"),
        "events_per_category_hour": compare(tables["events_per_category_hour"],
                                            pd.read_csv(gt / "events_per_category_hour.csv"),
                                            ["category", "hour"], "n_events", "n_events"),
    }}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(explain)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
