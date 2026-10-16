import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

from decision_eval.suites import stem_synth

ITEMS = stem_synth.build()


def test_build_is_deterministic():
    assert stem_synth.build() == ITEMS


def test_matches_committed_items():
    """The committed results were scored on exactly these items; a generator change must not slip past."""
    committed = Path(__file__).parents[1] / "results" / "stem-synth" / "items.json"
    assert json.loads(committed.read_text()) == ITEMS


def test_item_invariants():
    assert len(ITEMS) == 180
    for it in ITEMS:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4, it
        assert it["answer"] in it["options"]
        assert it["shown_answer"] in it["options"]
        assert it["shown_is_correct"] == (it["shown_answer"] == it["answer"])
        assert it["score_level"] == stem_synth.mag_bin(it["magnitude"])
        assert it["category"] in stem_synth.CATEGORIES


def test_categories_balanced():
    counts = {c: sum(it["category"] == c for it in ITEMS) for c in stem_synth.CATEGORIES}
    assert set(counts.values()) == {15}


ARITH = re.compile(r"^What is (-?[\d.]+) ([-+×÷]) (-?[\d.]+)\?$")


@pytest.mark.parametrize("it", [it for it in ITEMS if ARITH.match(it["prompt"])], ids=lambda it: it["prompt"])
def test_arithmetic_answers(it):
    a, op, b = ARITH.match(it["prompt"]).groups()
    a, b = Fraction(a), Fraction(b)
    want = {"+": a + b, "-": a - b, "×": a * b, "÷": a / b}[op]
    assert Fraction(it["answer"]) == want


def test_answers_keep_short_decimals():
    """Regression: 3-significant-figure rounding once turned 112.5 into "112" and broke the answer key."""
    assert stem_synth.num(112.5) == "112.5"
    assert stem_synth.num(3307.5) == "3307.5"
    assert stem_synth.num(0.5875) == "0.588"


def test_committed_judgebench_items():
    """All 620 JudgeBench pairs (gpt then claude split), noul showing the correct response on exactly half."""
    items = json.loads((Path(__file__).parents[1] / "results" / "judgebench" / "items.json").read_text())
    assert len(items) == 620 and sum(it["shown_is_correct"] for it in items) == 310
    assert all(it["answer"] in it["options"] and len(it["options"]) == 2 for it in items)
