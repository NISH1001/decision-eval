import json
from collections import Counter
from pathlib import Path

import pytest

from decision_eval.runner import request
from decision_eval.suites import redef, redef_commit, redef_message

RESULTS = Path(__file__).parents[1] / "results"
ITEMS = json.loads((RESULTS / "redef" / "items.json").read_text())


def test_variants_share_the_items():
    """The three variants differ only in what goes in the state, so their scores compare item for item."""
    for sid in ("redef-commit", "redef-message"):
        assert json.loads((RESULTS / sid / "items.json").read_text()) == ITEMS


def test_balanced_and_stratified():
    assert len(ITEMS) == 200 and sum(it["shown_is_correct"] for it in ITEMS) == 100
    # validation split without MIX: buggy 193 new / 114 failed fixes, clean 687 fixes / 341 improvements
    assert Counter(it["category"] for it in ITEMS) == {"improvement, buggy": 63, "bug fix, buggy": 37,
                                                       "bug fix, works": 67, "improvement, works": 33}
    for it in ITEMS:
        assert it["options"] == redef.OPTIONS and it["answer"] in it["options"]
        assert it["shown_is_correct"] == (it["category"] in redef.GROUPS["buggy"])


@pytest.mark.parametrize("suite,keys", [(redef, {"file", "function", "diff"}),
                                        (redef_commit, {"file", "function", "commit_message", "diff"}),
                                        (redef_message, {"file", "function", "commit_message"})])
def test_state_never_carries_the_label(suite, keys):
    state, questions = request(suite, ITEMS[0])
    assert set(state) == keys and set(questions) == {"mc", "verify"}


def test_stratified_quota_sums_to_n():
    import random
    rows = [{"commit_type": t} for t in "a" * 193 + "b" * 114]
    got = Counter(r["commit_type"] for r in redef._stratified(rows, 100, random.Random(0)))
    assert got == {"a": 63, "b": 37}
