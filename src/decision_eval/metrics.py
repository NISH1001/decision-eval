"""Metrics over a results frame (one row per item). The results page recomputes the same definitions in JS."""

import numpy as np
import pandas as pd


def balanced_acc(d):
    """Mean of the accuracies on correct and on wrong shown answers: a constant answer scores exactly 0.5."""
    return np.mean([d[d.shown_is_correct == t].verify_correct.mean() for t in (True, False)
                    if (d.shown_is_correct == t).any()])


def auc(d):
    t, f = d[d.shown_is_correct].verify_p_yes.values, d[~d.shown_is_correct].verify_p_yes.values
    return ((t[:, None] > f[None, :]) + 0.5 * (t[:, None] == f[None, :])).mean()


def metrics(suite, items):
    """mode -> (label, metric fn, chance level), for the modes this suite has."""
    out = {}
    if "choice" in suite.MODES:
        out["choice"] = ("Choice accuracy", lambda d: d.mc_correct.mean(), 1 / len(items[0]["options"]))
    if "noul" in suite.MODES:
        out["noul"] = ("Verify balanced accuracy", balanced_acc, 0.5)
    if "score" in suite.MODES:
        out["score"] = ("Score-level accuracy", lambda d: d.score_correct.mean(), 1 / len(suite.SCORE_LABELS))
    return out


def bootstrap_ci(d, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def summary_table(dfs, suite, items):
    """One row per model: each mode's score with a 95% bootstrap CI, plus AUC, yes-rate, level MAE, latency, cost."""
    rows = []
    for name, d in dfs.items():
        row = {"model": name}
        for mode, (label, fn, _) in metrics(suite, items).items():
            row[label] = fn(d)
            row[label + " CI"] = tuple(bootstrap_ci(d, fn))
        if "noul" in suite.MODES:
            row["Verify AUC"] = auc(d)
            row["Says yes"] = d.verify_pred.mean()
        if "score" in suite.MODES:
            row["Score-level MAE"] = d.score_abs_err.mean()
        row["Median latency (ms)"] = d.latency_s.median() * 1000
        row["Cost (USD)"] = d.cost_usd.sum()
        rows.append(row)
    return pd.DataFrame(rows)
