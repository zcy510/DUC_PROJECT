# Phase 2: DP Integration (Milestone M2)

Phase 2 (weeks 3–5 of the plan, §4.5–4.6) adds the two places noise can go:

| | **Option B: central DP** | **Option A: local DP** |
|---|---|---|
| Who adds noise | the server, during the Spark aggregation | each phone, before uploading |
| Trusted server? | yes, it sees raw events | no, it only sees noisy reports |
| Mechanism | contribution bounding + Laplace noise (PipelineDP) | k-ary randomised response |
| Code | `src/central_dp.py` | `src/local_dp.py` |
| Releases | Q1 events/category, Q2 users/category, Q3 events/category×hour | category distribution (Q1 as shares) |

Both give the same guarantee: **ε-DP per user** (not per event). The tests are in `tests/test_dp.py` and the error metrics in `src/metrics.py`.

```bash
source .venv/bin/activate
pip install -r requirements.txt          # adds scipy (pinned)
pytest -v                                # 20 tests: 9 baseline + 11 DP
```

---

## Part 1: Central DP with PipelineDP

```bash
python src/central_dp.py --data data/n100k --out results/central/n100k --epsilon 1
```

Outputs in `results/central/n100k/`:
- the three tables (same shape as the baseline, plus each count's noise std);
- `explain.txt`, PipelineDP's own description of what it did (see below);
- `summary.json`, with the accuracy against the ground truth and the run time.

### How the code works (read `dp_aggregate()` in `central_dp.py`)

```python
accountant = NaiveBudgetAccountant(total_epsilon=eps, total_delta=0)   # 1
engine     = DPEngine(accountant, SparkRDDBackend(spark.sparkContext)) # 2
params = AggregateParams(metrics=[COUNT, PRIVACY_ID_COUNT],            # 3
                         max_partitions_contributed=L0,
                         max_contributions_per_partition=Linf,
                         noise_kind=LAPLACE)
extractors = DataExtractors(privacy_id_extractor=lambda r: r.user_id,  # 4
                            partition_extractor=lambda r: r.category)
result = engine.aggregate(rdd, params, extractors,
                          public_partitions=CATEGORIES)                # 5
accountant.compute_budgets()                                           # 6
result.collect()                                                       # 7
```

1. **Budget accountant.** One accountant for the whole release, so the total ε covers *all* statistics (sequential composition). δ = 0 because Laplace noise with public partitions is pure ε-DP.
2. **Engine + backend.** The same DP code runs on Spark (`SparkRDDBackend`) or in plain Python (`LocalBackend`, which the tests use).
3. **Parameters.** Which statistics to compute, and the **contribution bounds**.
4. **Extractors.** `privacy_id_extractor` decides **who is protected**: the user, not the event. Getting this wrong (for example, protecting events) is the classic DP bug flagged in your risk table.
5. **Public partitions.** The 50 categories are known in advance, so every category is released, even with zero users. Otherwise the mere *presence* of a rare category (say, "addiction_support") in the output could reveal that somebody used it.
6. **Budget split.** This only happens once *all* aggregations are declared. It must come before the first `collect()`.
7. **Run.** As in Spark, nothing runs before an action.

### Read `explain.txt`: it's your report's method section

```
Contribution bounding:
  max_partitions_contributed=4        ← L0
  max_contributions_per_partition=10  ← Linf
 5. Computed DP count with
     Laplace mechanism:  parameter=160.0  eps=0.25  l1_sensitivity=40.0
 6. Computed DP privacy_id_count with
     Laplace mechanism:  parameter=16.0  eps=0.25  l1_sensitivity=4.0
 ...
 5. Computed DP count (per category × hour) with
     Laplace mechanism:  parameter=80.0  eps=0.5  l1_sensitivity=40.0
```

What these numbers mean:
- **Sensitivity Δ = L₀ × L∞ = 4 × 10 = 40.** One user can change the counts by at most 40 in total.
- **Budget split, total ε = 1.** The two aggregations get 0.5 each (equal weights). Inside the first one, the two metrics get 0.25 each.
- **Laplace scale b = Δ / εᵢ = 40 / 0.25 = 160**, so the noise std = √2 · 160 ≈ 226 per category count.
- **Distinct users** have Δ = L₀ = 4 (each user counts at most once per category), so their noise is 10× smaller.

### Choosing the bounds (defaults, and what S3 will sweep)

The defaults were picked from the data, to keep about 90% of users unclipped. Measured on N = 100k:

| Per user | median | 90th pct | 99th pct | Default |
|---|---|---|---|---|
| categories used (→ L₀) | 3 | 4 | 6 | **L₀ = 4** |
| events per category (→ L∞) | 3 | 10 | 30 | **L∞ = 10** |
| (category, hour) cells (→ L₀ for Q3) | 7 | 21 | 43 | **L₀ₕ = 20** |
| events per cell (→ L∞ for Q3) | 1 | 2 | 4 | **L∞ₕ = 2** |

**Important for the report:** choosing bounds by looking at the private data is itself a (small) privacy leak. In a real deployment, bounds come from domain knowledge or from a DP quantile computation, and PipelineDP has `calculate_private_contribution_bounds` for that. Here the data is synthetic, so it's fine, but state the assumption.

### First result (N = 100k, ε = 1)

| Statistic | MRE | Spearman (ranking) | Largest error |
|---|---|---|---|
| Q1 events / category | 19.7% | 0.996 | 83,924 |
| Q2 users / category | 5.0% | 0.9998 | 1,533 |
| Q3 events / category × hour | 48.1% | 0.913 | 6,025 |

**Noise is not the main error for popular categories; bias is.** The noise std is only about 226, yet "messaging" is off by about 84,000. That's because L∞ = 10 cuts off power users' extra events: the counts are **systematically too low**, and most so for popular categories. That's the bias-versus-noise trade-off that **RQ3 / scenario S3** measures. Raising L∞ removes bias but raises the noise.

Q3 cells are small (50 × 24 = 1,200 cells), so the same noise gives a much larger relative error there.

---

## Part 2: Local DP with randomised response

```bash
python src/local_dp.py --data data/n100k --out results/local/n100k --epsilon 1 --seed 0
```

### The three steps (in `local_dp.py`)

1. **On each device** (`simulate_devices`): pick at most `m` of the user's events and randomise each one with `krr_perturb` at ε/m.
2. **Ingestion:** the noisy reports are written as Parquet. That's all the server ever receives.
3. **On the server, in Spark** (`spark_estimate`): count reports per category, then de-bias with `krr_estimate`.

### k-RR in four lines

```python
p = e^ε / (e^ε + k − 1)        # keep the true category
q = 1   / (e^ε + k − 1)        # report one particular other category
keep  = random() < p
other = random other category (never the true one)
```

Check: p / q = e^ε, which is exactly the ε-LDP guarantee. A test verifies this for several ε and k.

The estimator ĉᵢ = (cᵢ − n·q) / (p − q) "undoes" the expected amount of fake reports. It is **unbiased**: correct on average, but with high variance.

### The design decision your plan didn't spell out: user-level LDP

Users have up to ~1,000 events. If every event were randomised with ε, one user's total privacy loss would be **ε × (number of events)**, which is not comparable to central DP at all. So:

> each user sends **at most m reports, each randomised with ε/m** → total ε per user (sequential composition).

The default is `m = 1`. That gives the strongest signal per report: with k = 50 and ε = 1, p is only 0.053, so the true category is reported just 5% of the time. Larger m means more reports but noisier ones. That's a parameter you can explore in Phase 3.

Consequence: LDP estimates the **category distribution** (shares) of users' events, not total event counts. To compare it with central DP in RQ2, use the shares (`est_share`), or rank metrics (Spearman, precision@k).

### First result (ε = 1, m = 1)

| | Central DP, Q1 | Local DP, N = 100k | Local DP, N = 1M |
|---|---|---|---|
| Spearman vs truth | **0.996** (N = 100k) | 0.52 | 0.81 |
| precision@10 | 1.0 | 0.7 | 1.0 |
| MRE (shares) | — | 164% | 49% |

As your plan predicted, **local DP needs far more users**: with 10× more users it's still worse than central DP at 100k. That's the core finding for RQ2.

---

## Part 3: The tests (T0 for DP)

`tests/test_dp.py` has 11 tests and runs in about 3 seconds on PipelineDP's LocalBackend:

| Test | What it proves |
|---|---|
| `budget_split_and_noise_scale_follow_theory` | std = √2·Δ/εᵢ for all three statistics, with the εᵢ split as described above |
| `noise_is_laplace_with_expected_scale` | over 400 runs, the noise is centred at 0, has the right std, and passes a KS test for the Laplace shape |
| `contribution_bounding_limits_a_power_user` | a user with 1,000 events counts as ≤ L∞ = 10 |
| `cross_partition_bounding_limits_categories` | a user in 5 categories keeps only L₀ = 2 |
| `all_public_partitions_are_released` | all 50 categories (and 1,200 cells) appear, even when empty |
| `krr_probabilities` | p + (k−1)q = 1 and p/q = e^ε |
| `krr_keeps_true_value_with_probability_p` | the empirical keep rate equals p |
| `krr_estimator_is_unbiased` | the mean of 300 estimates equals the true counts |
| `devices_send_at_most_m_reports` | user-level privacy: no user sends more than m reports |

**The tests were checked against deliberate bugs.** Doubling L∞ in the code, or dropping `− q` from the estimator, makes 4 tests fail, so they really catch the kinds of mistakes your risk table worries about.

---

## Part 4: Pitfalls I hit, so you don't have to

1. **`PYTHON_VERSION_MISMATCH`.** PipelineDP runs Python lambdas *inside* Spark tasks, in separate worker processes. Spark started them with the system Python instead of `.venv`'s. Fix (already in `make_spark()`): set `PYSPARK_PYTHON = sys.executable`. The baseline never hit this because it only used built-in DataFrame functions, which run in the JVM.
2. **Call `compute_budgets()` after declaring all aggregations and before collecting.** Otherwise PipelineDP raises an error, because it doesn't yet know how to split ε.
3. **DP noise isn't seedable.** PipelineDP uses secure noise on purpose, so the same run gives different results. The 30 repetitions in Phase 3 come for free. Only the LDP device simulation takes a `--seed`.

---

## Part 5: Runtime (for RQ4)

On the test machine (only 2 CPU cores, `local[*]`), at ε = 1:

| N users | Events | Baseline, all 3 queries (s) | Central DP (s) | Local DP device / server (s) |
|---|---|---|---|---|
| 100k | 1.2 M | ≈ 3.3 | 38 | 1.3 / 1.8 |
| 1M | 12.2 M | ≈ 12.8 | **595** (≈ 10 min) | 15.6 / 2.1 |

Central DP is **much** slower than the baseline. There are two reasons, and you should explain both in the report:
- **The algorithm.** Contribution bounding groups the raw events by user (extra shuffles) before counting. That's the cost you expected.
- **The implementation.** PipelineDP works on RDDs of Python objects, so every row is serialised between the JVM and Python workers. The baseline uses DataFrames, which stay inside the JVM.

That second point is a genuine engineering finding about DP libraries on Spark. **Re-measure on your Mac**: more cores means faster runs.

**Plan Phase 3 around this.** 30 repetitions × 7 ε values of central DP at 1M users would take about 35 hours on this machine. Run the full ε / bounds sweeps on n10k and n100k, and use n1m only for a few points (for example ε = 1) and the runtime study. Accuracy at 1M users, ε = 1, for reference: Q1 MRE 19% (bias), Spearman 0.9999; Q3 MRE 18%. More users make the noise relatively smaller, but the bounding bias stays.

---

## Done = Milestone M2 ✅

- [ ] `pip install -r requirements.txt`, then `pytest -v` → 20 passed
- [ ] `central_dp.py` run on n10k / n100k (n1m takes a few minutes); read `explain.txt`
- [ ] `local_dp.py` run on n100k / n1m
- [ ] Commit: `git add . && git commit -m "Phase 2: central DP (PipelineDP) and local DP (k-RR)"`

### Next: Phase 3 (experiments)

A script that loops over ε ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8}, both models, N, and the bounds, with 30 repetitions each, and writes one row per run to a CSV. Then plots of error vs ε, central vs local, and bias vs noise for different L∞. Central DP at N = 1M is slow, so run the large sweeps on n10k / n100k, and use n1m for selected points and the runtime study.
