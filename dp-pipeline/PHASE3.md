# Phase 3: Experiments (Milestone M3)

Two scripts:

| Script | Job |
|---|---|
| `src/experiments.py` | runs all experiments and appends **one result line per run** |
| `src/plot_results.py` | turns the results into 4 figures and 4 summary tables (one per RQ) |

## Why only 3 experiments for 4 scenarios

| Experiment | Covers | Settings | Runs |
|---|---|---|---|
| `grid` | **S1 + S2 + S4** | ε ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8} × N ∈ {10k, 100k, 1M} × {central, local} × 30 reps | 1,260 |
| `bounds` | **S3** | L∞ ∈ {1, 2, 5, 10, 20, 50, 100} (L₀ = 4) and L₀ ∈ {1, 2, 8, 16} (L∞ = 10); N = 100k, ε = 1, 30 reps | 330 |
| `baseline` | S4 reference | non-private job × 3 sizes × 5 reps | 15 |

- **S1 (ε sweep)** is the central-DP part of the grid: no extra runs needed.
- **S4 (runtime)** uses the times recorded in every grid run, plus the baseline runs for comparison.

---

## Step 1: Install and do a quick test (about 3 minutes)

```bash
cd ~/Documents/StockholmUniversity/DUC/DUC_PROJECT/dp-pipeline
source .venv/bin/activate
pip install -r requirements.txt            # adds matplotlib
python src/experiments.py --quick          # n10k, 3 ε, 2 reps → results/experiments_quick/
python src/plot_results.py --results results/experiments_quick
open results/experiments_quick/figures     # look at the 4 PNGs
```

If the figures appear, everything works. The quick results are only a smoke test (2 reps, so the bands are wide); don't use them in the report.

## Step 2: Run the real experiments (overnight)

```bash
mkdir -p results/experiments
caffeinate -i python src/experiments.py --experiment all 2>&1 | tee -a results/experiments/log.txt
```

- **`caffeinate -i`** stops your Mac from sleeping while it runs (macOS built-in). Keep the charger plugged in.
- **`tee -a … log.txt`** shows progress and also saves it to a log.
- **Order:** baseline → grid (10k → 100k → 1M) → bounds.

**Expected time on your Mac** (from your Phase 2 timings: central DP 6 s / 13 s / 80 s per run):

| Part | Approx. time |
|---|---|
| baseline | < 5 min |
| grid, N = 10k | ≈ 25 min |
| grid, N = 100k | ≈ 50 min |
| grid, N = 1M | ≈ **5 h** (central DP 80 s × 210 runs) |
| bounds (N = 100k) | ≈ 75 min |
| **Total** | **≈ 7–8 h** |

### It's safe to stop and restart

Every finished run is saved immediately. If the Mac sleeps, crashes or you press Ctrl-C, **just run the same command again**: finished runs are skipped. You can check progress any time:

```bash
wc -l results/experiments/runs.jsonl        # number of finished runs (goal: 1,605)
tail -5 results/experiments/log.txt
```

### Short on time? Split it

```bash
# small and medium sizes first (≈ 2.5 h incl. bounds)
caffeinate -i python src/experiments.py --experiment all --sizes n10k n100k
# 1M users with 10 reps instead of 30 (≈ 1.7 h); add more reps later if you want
caffeinate -i python src/experiments.py --experiment grid --sizes n1m --reps 10
```

The plots compute confidence intervals from however many reps exist, so 10 reps at 1M simply give wider bands. Say so in the report.

## Step 3: Make the figures

```bash
python src/plot_results.py
open results/experiments/figures
```

---

## What's recorded

**`results/experiments/runs.jsonl`**: one JSON object per line, one line per run:
- **keys:** `experiment, model, dataset, epsilon, l0, linf, rep, n_users`
- **time:** `seconds` (local DP also has `device_seconds` and `server_seconds`)
- **central DP:** `q1_*`, `q2_*`, `q3_*` metrics (MRE, MAE, Linf_error, precision_at_10, spearman), plus `q1_noise_std`
- **both models:** `q1share_*`, the metrics on **category shares**, which is how central and local DP are compared (local DP estimates shares)
- **local DP:** `q1_negative_estimates`, the number of impossible negative counts

Read it in Python with `pd.read_json("results/experiments/runs.jsonl", lines=True)`. It's JSON Lines rather than CSV because baseline and DP runs have different columns.

**`results/experiments/cells_q1.csv`**: the Q1 error for **each of the 50 categories** in every run. RQ3 and any per-category analysis use it, for example "how do sensitive rare categories do?".

**Repetitions:** central DP noise is fresh in every run (PipelineDP's secure noise can't be seeded). Local DP uses `seed = rep`, so it's reproducible.

---

## How to read each figure

### `fig_rq1_error_vs_epsilon.png` (RQ1)
Central DP, mean relative error vs ε, one panel per query and one line per N.
- **Q3** should fall steadily with ε: it's noise-dominated, and the noise scale is Δ/ε.
- **Q1** should **flatten** at large ε: the error that remains is **bias** from contribution bounding, which ε can't remove.
- **"From which ε are results useful?"** Read off where each line crosses your chosen threshold, for example 10% MRE.

### `fig_rq2_central_vs_local.png` (RQ2)
Category **shares**: central (solid) vs local (dashed), with a colour per N.
- The **vertical gap** between the solid and dashed lines is the cost of not trusting the server.
- **Read across colours:** how much larger must N be for local DP to match central DP at the same ε?
- The **right panel (Spearman)** shows whether the *ranking* survives, even when values are noisy.

### `fig_rq3_bounds.png` (RQ3)
Three curves for each bound:
- **bias:** data dropped by bounding, **computed exactly** from the raw events. Each user keeps min(events, L∞) per category, and each category survives with probability min(1, L₀ / categories used). This formula was checked against PipelineDP itself: the totals agree within 0.04%.
- **noise:** the expected size of the Laplace noise, also exact, from Δ/ε.
- **total:** the measured error over all reps, with a 95% CI.

Bias falls and noise rises as the bounds grow, so total error is **U-shaped**. The label marks the best setting. This plot is the direct answer to RQ3.

### `fig_rq4_runtime.png` (RQ4)
Seconds per run vs N (log–log) for the baseline, central DP and local DP. `summary_rq4.csv` contains the **overhead factor** (DP time ÷ baseline time) for each N.
- A slope of about 1 on a log–log plot means runtime grows linearly with data size.
- Central DP's overhead comes from the per-user shuffle for contribution bounding and from Python RDD serialisation (see PHASE2.md, Part 5).

### `summary_rq*.csv`
The numbers behind each figure (mean, 95% CI half-width, number of reps). Use them for tables in the report.

---

## Done = Milestone M3 ✅

- [ ] `--quick` works and makes 4 figures
- [ ] Full run finished: `wc -l results/experiments/runs.jsonl` → 1,605 (fewer if you used `--reps 10` at 1M)
- [ ] `python src/plot_results.py` → 4 figures and 4 summary tables
- [ ] Commit: `git add . && git commit -m "Phase 3: experiments and figures"`. The results are small (about 10 MB), so commit them; that makes the report reproducible.

Then Phase 4: the report, built around these 4 figures.
