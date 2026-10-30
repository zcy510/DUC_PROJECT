"""
Baseline (non-private) Spark aggregation job (Phase 1, step 2).

Reads the event Parquet files and computes three exact statistics
(project plan, section 4.4):
  Q1  number of events per category
  Q2  number of distinct users per category
  Q3  number of events per category and hour of day

These exact results are (a) the accuracy reference for the DP versions and
(b) the runtime baseline used to measure DP overhead (RQ4).

Usage:
  python src/baseline_job.py --data data/n10k --out results/baseline/n10k
  python src/baseline_job.py --data data/n10k --out results/baseline/n10k --explain
"""

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def make_spark(master: str = "local[*]", shuffle_partitions: int = 8) -> SparkSession:
    """Create a Spark session.

    - local[*]: run Spark on this machine, one worker thread per CPU core.
      On a real cluster you would pass e.g. spark://host:7077 instead.
    - spark.sql.shuffle.partitions: how many partitions a shuffle produces.
      The default (200) is meant for clusters; for a laptop, ~2x cores is better.
    - session time zone UTC: the generator writes time-zone-free timestamps
      (Spark reads them as TIMESTAMP_NTZ, so hour(ts) is taken literally).
      Fixing the session to UTC is a safety net: if timestamps ever carry a
      time zone, hour(ts) still matches the generator instead of shifting
      to your laptop's local time.
    """
    return (SparkSession.builder
            .master(master)
            .appName("dp-baseline")
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def events_per_category(events: DataFrame) -> DataFrame:
    """Q1: SELECT category, COUNT(*) FROM events GROUP BY category"""
    return events.groupBy("category").agg(F.count("*").alias("n_events"))


def users_per_category(events: DataFrame) -> DataFrame:
    """Q2: SELECT category, COUNT(DISTINCT user_id) ... GROUP BY category"""
    return events.groupBy("category").agg(F.countDistinct("user_id").alias("n_users"))


def events_per_category_hour(events: DataFrame) -> DataFrame:
    """Q3: events per (category, hour of day)"""
    return (events.withColumn("hour", F.hour("ts"))
            .groupBy("category", "hour").agg(F.count("*").alias("n_events")))


QUERIES = {
    "events_per_category": events_per_category,
    "users_per_category": users_per_category,
    "events_per_category_hour": events_per_category_hour,
}


def run(spark: SparkSession, data_dir: Path, explain: bool = False) -> tuple[dict, dict]:
    """Run all three queries. Returns (results as pandas, timings in seconds)."""
    events = spark.read.parquet(str(data_dir / "events"))

    # Warm-up: the first Spark action pays JVM / code-generation start-up
    # costs. Doing one cheap action first keeps the timings below fair.
    n_events = events.count()

    results, timings = {}, {"n_events": n_events}
    for name, query in QUERIES.items():
        df = query(events)
        if explain:
            print(f"\n===== Physical plan: {name} =====")
            df.explain()  # look for 'Exchange' = a shuffle across the cluster
        t0 = time.perf_counter()
        # Spark is lazy: nothing runs until an action. toPandas() is the action.
        # Results are tiny (<= 50 x 24 rows), so collecting them is fine.
        results[name] = df.toPandas()
        timings[name] = round(time.perf_counter() - t0, 3)
    return results, timings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder made by generate_data.py")
    ap.add_argument("--out", type=Path, required=True, help="where to write results")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--shuffle-partitions", type=int, default=8)
    ap.add_argument("--explain", action="store_true", help="print Spark physical plans")
    args = ap.parse_args()

    spark = make_spark(args.master, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.perf_counter()
    results, timings = run(spark, args.data, args.explain)
    timings["total_job"] = round(time.perf_counter() - t0, 3)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, pdf in results.items():
        keys = ["category", "hour"] if "hour" in pdf.columns else ["category"]
        pdf.sort_values(keys).to_csv(args.out / f"{name}.csv", index=False)
    timings.update(master=args.master, shuffle_partitions=args.shuffle_partitions,
                   spark_version=spark.version)
    (args.out / "runtime.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
