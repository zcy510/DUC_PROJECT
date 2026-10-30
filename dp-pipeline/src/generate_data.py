"""
Synthetic generator for sensitive app-usage events (Phase 1, step 1).

Each row is one event:  user_id | category | ts
  - user_id  : pseudonymous integer ID (no real identity)
  - category : app category, e.g. "messaging", "banking", "dating"
  - ts       : UTC timestamp of the event

Design (from the project plan, section 4.3):
  - N users
  - ~50 categories whose popularity follows a Zipf distribution
    (a few very popular categories, a long tail of rare ones)
  - a heavy-tailed (log-normal) number of events per user, so a few
    "power users" generate very many events -> makes contribution
    bounding relevant later
  - timestamps with a daily pattern (quiet at night, busy in the evening)

Because we generate the data ourselves, we also know the exact answers.
The generator writes them as "ground truth" so every later result
(baseline and DP) can be scored precisely.

Usage:
  python src/generate_data.py --users 10000 --out data/n10k
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 50 categories, ordered from most to least popular (rank 1 = most popular).
# Sensitive categories (health, dating, religion, politics ...) are mostly
# in the tail -- which is exactly where DP noise hurts accuracy the most.
CATEGORIES = [
    "messaging", "social_media", "web_browser", "video_streaming", "email",
    "music", "maps_navigation", "camera", "news", "shopping",
    "weather", "banking", "payments", "food_delivery", "calendar",
    "public_transport", "photo_editing", "gaming", "notes", "podcasts",
    "ride_hailing", "fitness", "cloud_storage", "productivity", "health_tracking",
    "ebooks", "language_learning", "travel_booking", "job_search", "education",
    "smart_home", "investing", "vpn", "password_manager", "dating",
    "political_news", "parking", "insurance", "government_services", "telehealth",
    "mental_health", "religion_prayer", "period_tracker", "medication_reminder", "crypto",
    "pregnancy", "lgbtq_community", "gambling", "lottery", "addiction_support",
]

# Relative activity per hour of day (0..23): low at night, a morning bump,
# a lunch bump and an evening peak. Normalised to probabilities below.
HOURLY_WEIGHTS = np.array([
    2, 1, 1, 1, 1, 2,      # 00-05  night
    4, 7, 8, 7, 6, 7,      # 06-11  morning
    9, 8, 6, 6, 7, 8,      # 12-17  lunch / afternoon
    10, 11, 12, 11, 8, 4,  # 18-23  evening peak
], dtype=float)


def zipf_probabilities(k: int, s: float) -> np.ndarray:
    """P(rank r) proportional to 1 / r^s for r = 1..k."""
    ranks = np.arange(1, k + 1)
    w = 1.0 / ranks ** s
    return w / w.sum()


def generate(n_users: int, n_categories: int = 50, zipf_s: float = 1.1,
             lognorm_mu: float = 2.0, lognorm_sigma: float = 1.0,
             max_favourites: int = 12, days: int = 7,
             start: str = "2026-10-01", seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of synthetic events (fully determined by `seed`)."""
    rng = np.random.default_rng(seed)
    cat_p = zipf_probabilities(n_categories, zipf_s)

    # 1) How many events does each user produce?  Log-normal -> heavy tail.
    #    mu=2, sigma=1 gives median ~7 events, mean ~12, a few users with 100s.
    events_per_user = np.maximum(
        1, np.rint(rng.lognormal(lognorm_mu, lognorm_sigma, n_users))
    ).astype(np.int64)

    # 2) Each user has a few "favourite" categories (people use a handful of
    #    apps). The favourites are drawn from the global Zipf popularity, so
    #    popular categories appear in many users' sets.
    n_fav = np.clip(1 + rng.poisson(3, n_users), 1, max_favourites)
    fav = rng.choice(n_categories, size=(n_users, max_favourites), p=cat_p)

    # 3) Expand to one row per event, and pick one of the user's favourites.
    user_idx = np.repeat(np.arange(n_users), events_per_user)
    pick = (rng.random(user_idx.size) * n_fav[user_idx]).astype(np.int64)
    cat_idx = fav[user_idx, pick]

    # 4) Timestamps: a random day, an hour from the daily pattern,
    #    and a random second within that hour.
    hour_p = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()
    day = rng.integers(0, days, user_idx.size)
    hour = rng.choice(24, size=user_idx.size, p=hour_p)
    sec = rng.integers(0, 3600, user_idx.size)
    base = np.datetime64(start, "s")
    ts = base + (day * 86400 + hour * 3600 + sec).astype("timedelta64[s]")

    return pd.DataFrame({
        "user_id": user_idx.astype(np.int64),
        "category": pd.Categorical.from_codes(cat_idx, CATEGORIES[:n_categories]).astype(str),
        "ts": ts,
    })


def ground_truth(df: pd.DataFrame) -> dict:
    """Exact answers to the three baseline queries, computed with pandas
    (independently of Spark, so the Spark job can be checked against it)."""
    hour = df["ts"].dt.hour
    return {
        "events_per_category": (df.groupby("category").size()
                                .rename("n_events").reset_index()),
        "users_per_category": (df.groupby("category")["user_id"].nunique()
                               .rename("n_users").reset_index()),
        "events_per_category_hour": (df.assign(hour=hour)
                                     .groupby(["category", "hour"]).size()
                                     .rename("n_events").reset_index()),
    }


def write_parquet(df: pd.DataFrame, out_dir: Path, rows_per_file: int) -> int:
    """Write events as several Parquet files, like many devices uploading
    batches. Several files let Spark read them in parallel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for i, start in enumerate(range(0, len(df), rows_per_file)):
        chunk = pa.Table.from_pandas(df.iloc[start:start + rows_per_file],
                                     preserve_index=False)
        pq.write_table(chunk, out_dir / f"part-{i:05d}.parquet")
        n_files += 1
    return n_files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=10_000, help="number of users N")
    ap.add_argument("--categories", type=int, default=50)
    ap.add_argument("--zipf-s", type=float, default=1.1, help="Zipf exponent (higher = more skewed)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows-per-file", type=int, default=500_000)
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. data/n10k")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = generate(args.users, args.categories, args.zipf_s,
                  days=args.days, seed=args.seed)
    n_files = write_parquet(df, args.out / "events", args.rows_per_file)

    gt_dir = args.out / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for name, table in ground_truth(df).items():
        table.to_csv(gt_dir / f"{name}.csv", index=False)

    per_user = df.groupby("user_id").size()
    meta = {
        "users": args.users, "categories": args.categories, "zipf_s": args.zipf_s,
        "days": args.days, "seed": args.seed, "events": int(len(df)),
        "parquet_files": n_files,
        "events_per_user": {"median": float(per_user.median()),
                            "mean": round(float(per_user.mean()), 2),
                            "max": int(per_user.max())},
        "categories_per_user_max": int(df.groupby("user_id")["category"].nunique().max()),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
