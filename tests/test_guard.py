import json
from pathlib import Path

import pandas as pd
import pytest

from decision_eval.metrics import flag_metrics
from decision_eval.reflexguard import best_threshold

RESULTS = Path(__file__).parents[1] / "results"


def test_flag_metrics_by_hand():
    d = pd.DataFrame({"kind": ["harmful", "harmful", "benign", "hard_negative"],
                      "p_noul": [0.9, 0.2, 0.1, 0.8], "threshold_noul": 0.5, "latency_ms_noul": 10.0})
    d["flag_noul"] = d.p_noul >= d.threshold_noul
    m = flag_metrics(d, "noul")  # TP=1 FN=1 FP=1 TN=1
    assert (m["recall"], m["precision"], m["f1"], m["accuracy"]) == (0.5, 0.5, 0.5, 0.5)
    assert (m["benign_fp"], m["hard_fp"]) == (0.0, 1.0)
    assert m["auc"] == 0.75  # of 4 (positive, negative) pairs, 0.9 beats both negatives and 0.2 beats 0.1


def test_best_threshold_prefers_higher_on_ties():
    assert best_threshold([(0.9, True), (0.8, True), (0.3, False), (0.1, False)]) == 0.8


# reflex-guard's own reported numbers (data/eval/results.csv at 999d51d), single-mode configs
REPORTED = {("jev", "score"): (0.303, 0.7596, 0.0106, 0.0806), ("gpt-6-luna", "noul"): (0.71, 0.7692, 0.0426, 0.121),
            ("laya", "noul"): (0.36, 0.7115, 0.0957, 0.2097)}


@pytest.mark.parametrize("key,mode", list(REPORTED))
def test_import_reproduces_reflexguard_report(key, mode):
    d = pd.read_csv(RESULTS / "guard-core" / f"results_{key}.csv")
    m = flag_metrics(d, mode)
    t, recall, benign_fp, hard_fp = REPORTED[(key, mode)]
    assert m["threshold"] == pytest.approx(t, abs=1e-3)
    assert (m["recall"], m["benign_fp"], m["hard_fp"]) == pytest.approx((recall, benign_fp, hard_fp), abs=1e-3)


def test_guard_items_match_results():
    for suite in ("guard-core", "guard-ood"):
        items = json.loads((RESULTS / suite / "items.json").read_text())
        for f in (RESULTS / suite).glob("results_*.csv"):
            assert pd.read_csv(f).prompt.tolist() == [it["prompt"] for it in items], f
