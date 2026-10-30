"""
T0 correctness tests for Phase 1 (run with:  pytest -v).

1. The generator produces the shape the plan asks for
   (Zipf popularity, heavy-tailed users, daily pattern, reproducible).
2. The Spark baseline gives EXACTLY the generator's ground truth.
   If this fails, the bug is in our Spark code -- not in DP.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baseline_job  # noqa: E402
import generate_data as gen  # noqa: E402

N_USERS = 3_000


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Generate a small dataset once and share it between the tests."""
    out = tmp_path_factory.mktemp("data")
    df = gen.generate(N_USERS, seed=7)
    gen.write_parquet(df, out / "events", rows_per_file=10_000)  # several files
    return out, df, gen.ground_truth(df)


@pytest.fixture(scope="module")
def spark():
    s = baseline_job.make_spark("local[2]", shuffle_partitions=4)
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


# ---------- 1. generator properties ----------

def test_generator_is_reproducible():
    a = gen.generate(500, seed=1)
    b = gen.generate(500, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_every_user_has_at_least_one_event(dataset):
    _, df, _ = dataset
    assert df["user_id"].nunique() == N_USERS


def test_popularity_is_skewed_like_zipf(dataset):
    _, df, _ = dataset
    counts = df["category"].value_counts()
    # The most popular category should be the rank-1 one, and much more
    # frequent than the median category.
    assert counts.index[0] == gen.CATEGORIES[0]
    assert counts.iloc[0] > 5 * counts.median()


def test_events_per_user_is_heavy_tailed(dataset):
    _, df, _ = dataset
    per_user = df.groupby("user_id").size()
    assert per_user.max() > 10 * per_user.median()  # a few power users


def test_daily_pattern(dataset):
    _, df, _ = dataset
    by_hour = df["ts"].dt.hour.value_counts()
    assert by_hour[20] > 3 * by_hour[3]  # evening much busier than 3 a.m.


# ---------- 2. Spark baseline == ground truth ----------

@pytest.mark.parametrize("name,keys", [
    ("events_per_category", ["category"]),
    ("users_per_category", ["category"]),
    ("events_per_category_hour", ["category", "hour"]),
])
def test_spark_matches_ground_truth(dataset, spark, name, keys):
    data_dir, _, truth = dataset
    results, _ = baseline_job.run(spark, data_dir)

    got = results[name].sort_values(keys).reset_index(drop=True)
    exp = truth[name].sort_values(keys).reset_index(drop=True)
    value = got.columns.difference(keys)[0]

    assert list(got[keys].itertuples(index=False)) == list(exp[keys].itertuples(index=False))
    np.testing.assert_array_equal(got[value].to_numpy(), exp[value].to_numpy())


def test_total_events_match(dataset, spark):
    data_dir, df, _ = dataset
    _, timings = baseline_job.run(spark, data_dir)
    assert timings["n_events"] == len(df)
