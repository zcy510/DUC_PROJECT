"""
Phase 3 experiment runner.

Three experiments (they cover all four scenarios of the plan):

  grid      S1 + S2 + S4   eps x N x {central, local} x reps
  bounds    S3             Linf sweep and L0 sweep at N=100k, eps=1
  baseline  S4 reference   non-private Spark job x N x reps (runtime only)

Every finished run is appended as ONE line to results/experiments/runs.jsonl
(JSON Lines: one JSON object per line; read with pd.read_json(path, lines=True)),
and its 50 per-category Q1 errors to cells_q1.csv. If the script stops
(crash, laptop closed, Ctrl-C), just start it again: runs already in the
CSV are skipped.

Usage:
  python src/experiments.py --experiment all                 # the real thing (hours)
  python src/experiments.py --experiment grid --sizes n10k n100k
  python src/experiments.py --quick                          # 2-minute smoke test
"""

import argparse
import csv
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_job
import central_dp
import generate_data as gen
import local_dp
from metrics import compare

EPSILONS = [0.1, 0.25, 0.5, 1, 2, 4, 8]                 # plan section 4.7
SIZES = ["n10k", "n100k", "n1m"]                        # N = 10^4, 10^5, 10^6
DEFAULT_BOUNDS = dict(l0=4, linf=10, l0_hour=20, linf_hour=2)
LINF_SWEEP = [1, 2, 5, 10, 20, 50, 100]                 # S3, L0 fixed at 4
L0_SWEEP = [1, 2, 4, 8, 16]                             # S3, Linf fixed at 10

RUN_KEYS = ["experiment", "model", "dataset", "epsilon", "l0", "linf", "rep"]


# ------------------------------------------------------------------ helpers

class Store:
    """Append-only result files + skip runs that are already done.
    runs.jsonl: rows from different experiments have different columns, so each
    run is one JSON object per line. cells_q1.csv: fixed columns, plain CSV."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.runs_path = out_dir / "runs.jsonl"
        self.cells_path = out_dir / "cells_q1.csv"
        self.done = set()
        if self.runs_path.exists():
            with self.runs_path.open() as f:
                self.done = {self._key(json.loads(line)) for line in f if line.strip()}

    @staticmethod
    def _key(r: dict) -> tuple:
        return (r["experiment"], r["model"], r["dataset"], float(r["epsilon"]),
                int(r["l0"]), int(r["linf"]), int(r["rep"]))

    def is_done(self, **key) -> bool:
        return self._key(key) in self.done

    @staticmethod
    def _append(path: Path, rows: list[dict]):
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    def save(self, run: dict, cells: list[dict] | None = None):
        # Cells first: if the script dies in between, the run is simply redone.
        if cells:
            self._append(self.cells_path, cells)
        with self.runs_path.open("a") as f:
            f.write(json.dumps(run) + "\n")
        self.done.add(self._key(run))


class Dataset:
    """Ground truth (and, for local DP, the raw events) of one dataset, loaded once."""

    def __init__(self, data_root: Path, name: str):
        self.name, self.dir = name, data_root / name
        gt = self.dir / "ground_truth"
        cats = pd.DataFrame({"category": gen.CATEGORIES})
        self.q1 = cats.merge(pd.read_csv(gt / "events_per_category.csv"), how="left").fillna(0)
        self.q1["share"] = self.q1["n_events"] / self.q1["n_events"].sum()
        self.q2 = pd.read_csv(gt / "users_per_category.csv")
        self.q3 = pd.read_csv(gt / "events_per_category_hour.csv")
        self.n_users = int(pd.read_json(self.dir / "metadata.json", typ="series")["users"])
        self._events = None

    @property
    def events(self) -> pd.DataFrame:  # only needed for local DP, so load lazily
        if self._events is None:
            self._events = pd.read_parquet(self.dir / "events", columns=["user_id", "category"])
        return self._events


def flat(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k.replace('@', '_at_')}": v for k, v in metrics.items()}


def q1_cells(base: dict, est: pd.DataFrame, ds: Dataset) -> list[dict]:
    """Per-category error of Q1 for this run (needed to split bias vs noise)."""
    m = ds.q1.merge(est[["category", "n_events"]].rename(columns={"n_events": "est"}), on="category")
    return [{**base, "category": r.category, "true": r.n_events, "est": r.est, "error": r.est - r.n_events}
            for r in m.itertuples()]


# ------------------------------------------------------------------ one run of each model

def run_central(spark, ds: Dataset, eps: float, bounds: dict) -> tuple[dict, list[dict]]:
    tables, _, seconds = central_dp.run_spark(spark, ds.dir, eps, **bounds)
    q1 = tables["events_per_category"].copy()
    q1["share"] = q1["n_events"] / q1["n_events"].sum()
    row = {"seconds": seconds,
           "q1_noise_std": float(q1["events_noise_std"].iloc[0]),
           **flat("q1", compare(q1, ds.q1, ["category"], "n_events", "n_events")),
           **flat("q1share", compare(q1, ds.q1, ["category"], "share", "share")),
           **flat("q2", compare(tables["users_per_category"], ds.q2, ["category"], "n_users", "n_users")),
           **flat("q3", compare(tables["events_per_category_hour"], ds.q3,
                                ["category", "hour"], "n_events", "n_events"))}
    return row, q1


def run_local(spark, ds: Dataset, eps: float, seed: int, m: int = 1) -> tuple[dict, pd.DataFrame]:
    t0 = time.perf_counter()
    reports, _ = local_dp.simulate_devices(ds.events, eps, m, seed)
    tmp = Path(tempfile.mkdtemp(prefix="ldp_reports_"))
    reports.to_parquet(tmp / "part-00000.parquet", index=False)
    device_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    est = local_dp.spark_estimate(spark, tmp, eps / m)
    server_s = time.perf_counter() - t0
    shutil.rmtree(tmp, ignore_errors=True)
    # Scale shares to event counts with the (public-assumed) total, so per-category
    # errors are in the same units as central DP. Main LDP metric = shares.
    est["share"] = est["est_share"]
    est["n_events"] = est["share"] * ds.q1["n_events"].sum()
    row = {"seconds": round(device_s + server_s, 3), "device_seconds": round(device_s, 3),
           "server_seconds": round(server_s, 3),
           **flat("q1share", compare(est, ds.q1, ["category"], "share", "share")),
           "q1_negative_estimates": int((est["est_reports"] < 0).sum())}
    return row, est


# ------------------------------------------------------------------ experiments

def experiment_grid(spark, store, data_root, sizes, epsilons, reps):
    """S1 + S2 + S4: every eps x N, central and local, `reps` times."""
    for name in sizes:
        ds = Dataset(data_root, name)
        for eps in epsilons:
            for rep in range(reps):
                for model in ("central", "local"):
                    key = dict(experiment="grid", model=model, dataset=name, epsilon=eps,
                               l0=DEFAULT_BOUNDS["l0"], linf=DEFAULT_BOUNDS["linf"], rep=rep)
                    if store.is_done(**key):
                        continue
                    if model == "central":
                        row, est = run_central(spark, ds, eps, DEFAULT_BOUNDS)
                    else:
                        row, est = run_local(spark, ds, eps, seed=rep)
                    base = {**key, "n_users": ds.n_users}
                    store.save({**base, **row}, q1_cells(base, est, ds))
                    print(f"[grid] {model:7s} {name:6s} eps={eps:<5} rep={rep:2d} "
                          f"{row['seconds']:7.1f}s  share-MRE={row['q1share_MRE']:.3f}", flush=True)


def experiment_bounds(spark, store, data_root, dataset, reps, linfs, l0s, eps=1.0):
    """S3: vary one bound at a time, central DP only (bounds don't exist in LDP)."""
    ds = Dataset(data_root, dataset)
    settings = [(4, x) for x in linfs] + [(x, 10) for x in l0s if x != 4]
    for l0, linf in settings:
        for rep in range(reps):
            key = dict(experiment="bounds", model="central", dataset=dataset, epsilon=eps,
                       l0=l0, linf=linf, rep=rep)
            if store.is_done(**key):
                continue
            bounds = {**DEFAULT_BOUNDS, "l0": l0, "linf": linf}
            row, est = run_central(spark, ds, eps, bounds)
            base = {**key, "n_users": ds.n_users}
            store.save({**base, **row}, q1_cells(base, est, ds))
            print(f"[bounds] L0={l0:2d} Linf={linf:3d} rep={rep:2d} {row['seconds']:6.1f}s "
                  f"MRE={row['q1_MRE']:.3f}", flush=True)


def experiment_baseline(spark, store, data_root, sizes, reps):
    """S4 reference: runtime of the non-private job (same Spark session, same settings)."""
    for name in sizes:
        ds = Dataset(data_root, name)
        for rep in range(reps):
            key = dict(experiment="baseline", model="none", dataset=name, epsilon=0,
                       l0=0, linf=0, rep=rep)
            if store.is_done(**key):
                continue
            t0 = time.perf_counter()
            _, timings = baseline_job.run(spark, ds.dir)
            seconds = sum(timings[q] for q in baseline_job.QUERIES)
            store.save({**key, "n_users": ds.n_users, "seconds": round(seconds, 3),
                        "wall_seconds": round(time.perf_counter() - t0, 3)})
            print(f"[baseline] {name:6s} rep={rep} {seconds:6.2f}s", flush=True)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["grid", "bounds", "baseline", "all"], default="all")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/experiments"))
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--baseline-reps", type=int, default=5)
    ap.add_argument("--bounds-dataset", default="n100k")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: n10k, 3 eps, 2 reps, 3 bounds -> results/experiments_quick")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--shuffle-partitions", type=int, default=8)
    args = ap.parse_args()

    epsilons, linfs, l0s = EPSILONS, LINF_SWEEP, L0_SWEEP
    if args.quick:
        args.sizes, args.reps, args.baseline_reps = ["n10k"], 2, 2
        args.bounds_dataset, args.out = "n10k", Path("results/experiments_quick")
        epsilons, linfs, l0s = [0.1, 1, 8], [1, 10, 100], [1, 4]

    missing = [s for s in set(args.sizes + [args.bounds_dataset]) if not (args.data / s).exists()]
    if missing:
        raise SystemExit(f"Missing datasets {missing}: create them with generate_data.py first.")

    spark = baseline_job.make_spark(args.master, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("ERROR")
    store = Store(args.out)
    t0 = time.perf_counter()
    try:
        # Warm-up (not recorded): the first Spark jobs pay JVM / Python-worker
        # start-up costs, which would make the first measured run look slower.
        warm = Dataset(args.data, min(args.sizes, key=lambda s: (args.data / s / "events").stat().st_size
                                      if (args.data / s / "events").exists() else 0))
        print(f"Warm-up on {warm.name} ...", flush=True)
        baseline_job.run(spark, warm.dir)
        central_dp.run_spark(spark, warm.dir, 1.0, **DEFAULT_BOUNDS)
        if args.experiment in ("baseline", "all"):
            experiment_baseline(spark, store, args.data, args.sizes, args.baseline_reps)
        if args.experiment in ("grid", "all"):
            experiment_grid(spark, store, args.data, args.sizes, epsilons, args.reps)
        if args.experiment in ("bounds", "all"):
            experiment_bounds(spark, store, args.data, args.bounds_dataset, args.reps, linfs, l0s)
    finally:
        spark.stop()
    print(f"\nDone in {time.perf_counter() - t0:.0f}s. Results: {store.runs_path}")


if __name__ == "__main__":
    main()
