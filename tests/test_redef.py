import json
from collections import Counter
from pathlib import Path

import pytest

from decision_eval.runner import request
from decision_eval.suites import redef, redef_commit

RESULTS = Path(__file__).parents[1] / "results"
ITEMS = json.loads((RESULTS / "redef" / "items.json").read_text())


def test_variants_share_the_items():
    """The two variants differ only in what goes in the state, so their scores compare item for item."""
    assert json.loads((RESULTS / "redef-commit" / "items.json").read_text()) == ITEMS


def test_balanced_and_stratified():
    assert len(ITEMS) == 200 and sum(it["shown_is_correct"] for it in ITEMS) == 100
    # validation split without MIX: buggy 193 new / 114 failed fixes, clean 687 fixes / 341 improvements
    assert Counter(it["category"] for it in ITEMS) == {"improvement, buggy": 63, "bug fix, buggy": 37,
                                                       "bug fix, works": 67, "improvement, works": 33}
    for it in ITEMS:
        assert it["options"] == redef.OPTIONS and it["answer"] in it["options"]
        assert it["shown_is_correct"] == (it["category"] in redef.GROUPS["buggy"])


@pytest.mark.parametrize("suite,keys", [(redef, {"file", "function", "diff"}),
                                        (redef_commit, {"file", "function", "commit_message", "diff"})])
def test_state_never_carries_the_label(suite, keys):
    state, questions = request(suite, ITEMS[0])
    assert set(state) == keys and set(questions) == {"mc", "verify", "size"}


def test_stratified_quota_sums_to_n():
    import random
    rows = [{"commit_type": t} for t in "a" * 193 + "b" * 114]
    got = Counter(r["commit_type"] for r in redef._stratified(rows, 100, random.Random(0)))
    assert got == {"a": 63, "b": 37}


def test_score_mode_flags_at_the_cutoff():
    """No gold level: a predicted level at or above SCORE_FLAG_AT means "introduces a bug", checked against the label."""
    from decision_eval.runner import score
    item = {**ITEMS[0], "shown_is_correct": True}
    ans = {"mc": {"choice": "A", "probabilities": {"A": 1.0, "B": 0, "C": 0, "D": 0}}, "verify": {"noul": 0.2}}
    low = score(redef, item, {**ans, "size": {"probabilities": {"0": 0.1, "1": 0.6, "2": 0.3, "3": 0, "4": 0}}})
    high = score(redef, item, {**ans, "size": {"probabilities": {"0": 0, "1": 0.2, "2": 0.5, "3": 0.3, "4": 0}}})
    assert (low["score_flag"], low["score_correct"], high["score_flag"], high["score_correct"]) == (False, False, True, True)
    assert low["score_risk"] == pytest.approx((0.6 + 0.6) / 4)
