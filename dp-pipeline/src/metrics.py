"""
Utility metrics: compare a private (noisy) result with the exact ground truth.
Used by the DP jobs for a quick summary, and in Phase 3 for the experiments.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compare(dp: pd.DataFrame, truth: pd.DataFrame, keys: list[str],
            dp_col: str, truth_col: str, top_k: int = 10) -> dict:
    """Join DP estimates with the truth on `keys` and compute error metrics.

    Cells missing from the truth (e.g. a category with no events at some hour)
    count as true value 0, because the DP output always contains them.
    """
    m = (dp[keys + [dp_col]].rename(columns={dp_col: "_est"})
         .merge(truth[keys + [truth_col]].rename(columns={truth_col: "_true"}),
                on=keys, how="left"))
    est = m["_est"].to_numpy(float)
    true = m["_true"].fillna(0).to_numpy(float)
    err = est - true
    nonzero = true > 0
    # precision@k: how many of the true top-k cells are also in the DP top-k
    top_true = set(np.argsort(-true)[:top_k])
    top_est = set(np.argsort(-est)[:top_k])
    return {
        "cells": int(len(m)),
        "MAE": float(np.mean(np.abs(err))),
        "MRE": float(np.mean(np.abs(err[nonzero]) / true[nonzero])),
        "Linf_error": float(np.max(np.abs(err))),
        f"precision@{top_k}": len(top_true & top_est) / top_k,
        "spearman": float(spearmanr(est, true).statistic),
    }
