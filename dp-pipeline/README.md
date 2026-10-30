# DP Pipeline — Phase 1: Baseline (Milestone M1)

Course project, DUC: *Implementing Differential Privacy in Distributed Data Aggregation Pipelines*.

Phase 1 is complete when you have two things (project plan §4.3–4.4, milestone M1):

- [x] **Synthetic data generator.** Realistic, sensitive-looking app-usage events with exact ground truth.
- [x] **Baseline Spark job.** Exact (non-private) counts. These are the accuracy reference and the runtime baseline for every DP experiment later.

Both are in this folder and tested. This guide explains **how to run them on your Mac**
and **why each part is built the way it is**, so you can explain it in the seminar and the report.

```
dp-pipeline/
├── setup.sh                one-command setup: creates .venv, checks versions, runs tests
├── requirements.txt        pinned versions (tested together)
├── .python-version         Python version used for the results (3.11)
├── src/
│   ├── generate_data.py    Step 1: synthetic events → Parquet + ground truth
│   └── baseline_job.py     Step 2: PySpark job, 3 exact queries + timings
├── tests/
│   └── test_baseline.py    Step 3: T0 correctness tests (pytest)
├── data/                   generated datasets (not in git; re-create any time)
└── results/baseline/       Spark output CSVs + runtime.json per dataset size
```

---

## Step 0 — Set up the environment (once)

**Versions are a real risk in this project** (your risk table rates it *High*). These rules come from the packages themselves:

| Component | Requirement | Why |
|---|---|---|
| Python | **3.11–3.13** (tested on 3.11 and 3.12) | PipelineDP (Phase 2) refuses other versions |
| Java | **17 or 21** | Spark 4.0 runs on the JVM and needs Java 17+ |
| macOS | 15 or newer | the `python-dp` wheels are built for macOS 15+ |

Anyone can rebuild the exact environment with **one command**. It creates a local `.venv` from the pinned `requirements.txt`, checks the Python and Java versions, and runs the tests to prove it works:

```bash
# Java 17 for Spark (skip if `java -version` already shows 17 or 21)
brew install openjdk@17
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc && source ~/.zshrc

cd dp-pipeline
bash setup.sh                       # or: PYTHON=python3.12 bash setup.sh
source .venv/bin/activate           # in every new terminal
```

It ends with `9 passed`, and then you're ready. In VS Code, pick the interpreter `.venv/bin/python`.

**Why `.venv` is not in Git:** a virtual environment contains machine-specific binaries and absolute paths, so a copied `.venv` breaks on other computers. What makes the pipeline reproducible is the recipe: `requirements.txt` (exact versions), `setup.sh` (exact steps) and `.python-version` (`3.11`, the version used for the reported results; pyenv and uv read this file automatically).

**pyenv users:** run `pyenv install 3.11` if you don't have it. `.python-version` then selects it in this folder, and `bash setup.sh` uses it.

PipelineDP is installed now even though it's only used in Phase 2. That way any version conflict shows up today, not in week 3.

---

## Step 1 — Generate the synthetic data

```bash
python src/generate_data.py --users 10000   --out data/n10k
python src/generate_data.py --users 100000  --out data/n100k
python src/generate_data.py --users 1000000 --out data/n1m     # ~30 s, ~90 MB
```

Each run creates:

- `events/part-*.parquet`: one row per event, `user_id | category | ts`
- `ground_truth/*.csv`: the exact answers, computed with **pandas** (not Spark)
- `metadata.json`: parameters, seed and summary stats

### Why the generator looks like this

| Design choice | What it does | Why it matters later |
|---|---|---|
| **Zipf popularity** (`--zipf-s 1.1`) | "messaging" is very common; "addiction_support" is very rare | DP noise is the same size for every category, so rare categories get a large *relative* error (RQ1). The sensitive categories are deliberately placed in the rare tail. |
| **Log-normal events per user** | median 7 events, but a few users have 1,000+ | These "power users" are why **contribution bounding** (L∞) is needed in central DP (RQ3) |
| **A few favourite categories per user** (1 + Poisson(3), max 12) | people use a handful of apps | Makes the **L₀** bound (max categories per user) meaningful |
| **Daily pattern** | quiet at 3 a.m., peak at 20:00 | Needed for the per-hour query; night-time cells are small, so they're noisy under DP |
| **Fixed seed** | same seed → identical data | Reproducibility; experiments use 30 different seeds |
| **Several Parquet files** | 500k rows per file | Spark reads files in parallel, like many devices uploading batches |
| **Ground truth from pandas** | independent of Spark | Lets the tests catch Spark bugs (T0) |

Measured with seed 42:

| Users N | Events | Median events/user | Max events/user | Parquet files |
|---|---|---|---|---|
| 10,000 | 121,539 | 7 | 414 | 1 |
| 1,000,000 | 12,196,359 | 7 | 1,105 | 25 |

**Try it:** run with `--zipf-s 1.5` and compare `ground_truth/events_per_category.csv`. The tail gets much thinner. This is a parameter you can vary and discuss as a limitation ("synthetic data not realistic enough" is in your risk table).

---

## Step 2 — Run the baseline Spark job

```bash
python src/baseline_job.py --data data/n10k --out results/baseline/n10k --explain
```

It computes the three queries from your plan:

| Query | Spark code | SQL equivalent |
|---|---|---|
| Q1 events per category | `groupBy("category").agg(count("*"))` | `SELECT category, COUNT(*) … GROUP BY category` |
| Q2 distinct users per category | `groupBy("category").agg(countDistinct("user_id"))` | `COUNT(DISTINCT user_id)` |
| Q3 events per category × hour | `withColumn("hour", hour("ts")).groupBy("category","hour")` | `GROUP BY category, HOUR(ts)` |

### Four Spark ideas to understand (and mention in your report)

1. **SparkSession and `local[*]`.** `local[*]` runs Spark on your laptop with one worker thread per CPU core. It's the same code that would run on a cluster; only the `--master` URL changes. This is why it counts as *distributed* processing even on one machine.

2. **Lazy evaluation.** `groupBy(...).agg(...)` only builds a plan; nothing runs until an *action* such as `.toPandas()`, `.count()` or `.write`. That's why the timer wraps `toPandas()`.

3. **Shuffles: look at the `--explain` output.** Every `Exchange hashpartitioning(...)` line is a **shuffle**: rows are re-sent across partitions (across machines, on a cluster) so that equal keys end up together.
   - Q1 and Q3 have **one** Exchange, and it's cheap: `partial_count` first adds up counts *inside* each partition, so only small subtotals are shuffled.
   - Q2 has **two** Exchanges: first by `(category, user_id)` to remove duplicates, then by `category`.

   This is exactly the cost structure your central-DP step will add. Contribution bounding must group raw rows by user before counting, so you should expect DP runtime to look more like Q2 than Q1.

4. **Configuration that matters.**
   - `spark.sql.shuffle.partitions = 8`: the default (200) is sized for clusters and makes laptop runs slower.
   - Session time zone `UTC`: keeps `hour(ts)` consistent with the generator.
   - A warm-up `count()` before timing, so JVM start-up isn't counted in Q1's time.

**Try it:** while the 1M job runs, open <http://localhost:4040> (the Spark UI). The *Stages* tab shows the shuffle read/write sizes for each Exchange. A screenshot of this is great material for the report.

---

## Step 3 — Run the correctness tests (T0)

```bash
pytest -v
```

There are 9 tests, all passing:

- **Generator:** reproducible with the same seed; every user has ≥ 1 event; Zipf skew (top category > 5× the median one); heavy tail (max > 10× median); daily pattern (20:00 > 3× 03:00).
- **Spark = ground truth:** all three queries match pandas **exactly**, row by row. The test writes several Parquet files on purpose, so the shuffle path really gets exercised.

When you add DP in Phase 2, extend this file with the other two T0 checks: the noise follows Lap(Δ/ε), and the local-DP estimator is unbiased.

---

## Step 4 — Record the baseline runtime

```bash
for n in n10k n100k n1m; do
  python src/baseline_job.py --data data/$n --out results/baseline/$n
done
```

Each run writes `results/baseline/<n>/runtime.json`. Reference numbers from the test machine (Linux, `local[*]`, Spark 4.0.1). **Re-run on your Mac and use your own numbers in the report.**

| Users N | Events | Q1 events/cat (s) | Q2 distinct users/cat (s) | Q3 events/cat×hour (s) | Total incl. start-up (s) |
|---|---|---|---|---|---|
| 10,000 | 0.12 M | 0.96 | 0.73 | 0.47 | 6.0 |
| 100,000 | 1.2 M | 1.07 | 1.52 | 0.75 | 7.2 |
| 1,000,000 | 12.2 M | 1.80 | **8.91** | 2.06 | 17.4 |

How to read this: up to 100k users, the run time is dominated by Spark's fixed start-up cost. At 1M users, Q2 is about 5× slower than Q1 because of its second shuffle. For RQ4, always compare DP runtime against these baseline numbers, measured on the same machine and with the same settings.

For the experiments, time each configuration several times (your plan says 30 runs) and report the mean ± 95% CI. Single timings vary by ±10–20%.

---

## Done = Milestone M1 ✅

- [ ] `bash setup.sh` → 9 passed (Step 0)
- [ ] Datasets for N = 10⁴, 10⁵, 10⁶ generated (Step 1)
- [ ] `pytest -v` → 9 passed (Step 3)
- [ ] `results/baseline/*/runtime.json` from **your** machine (Step 4)
- [ ] Commit to Git: `git init && git add . && git commit -m "Phase 1: baseline"` (`data/` is ignored)

### What to write in the report for Phase 1

- Generator design and parameters: use the "Why the generator looks like this" table.
- A plot of events per category (log scale) showing the Zipf tail, and a histogram of events per user showing the heavy tail.
- The baseline runtime table and the Q1-vs-Q2 shuffle observation.
- Limitation: categories are independent of time of day, and favourite categories don't change over the week.

## Next: Phase 2 (weeks 3–5)

- `src/central_dp.py`: the same three queries through **PipelineDP's Spark backend**, using `max_partitions_contributed` = L₀, `max_contributions_per_partition` = L∞, Laplace noise, and the category list given as *public partitions*.
- `src/local_dp.py`: k-ary randomised response per event (in NumPy, before writing Parquet), then the unbiased estimator on Spark.
- **Docker note:** `python-dp` has no Linux ARM wheel. On an Apple-Silicon Mac, build any Docker image with `--platform linux/amd64`. Otherwise PipelineDP won't install inside the container.
