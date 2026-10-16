import json

from decision_eval.runner import request, run
from decision_eval.suites import stem_synth

ITEM = stem_synth.build(n=12)[10]


def test_request_shape():
    """The exact request the published results were produced with; changing it invalidates them."""
    state, qs = request(stem_synth, ITEM)
    assert state == {"problem": ITEM["prompt"]}
    assert list(qs) == ["mc", "verify", "size"]
    assert qs["mc"] == {"type": "choice", "instructions": f"{ITEM['prompt']} Pick the correct answer.",
                        "criteria": dict(zip("ABCD", ITEM["options"]))}
    assert qs["verify"] == {"type": "noul",
                            "instructions": f"Is this answer correct? {ITEM['prompt']} Answer: {ITEM['shown_answer']}"}
    assert qs["size"]["criteria"] == stem_synth.SCORE_LABELS
    json.dumps(qs)  # must be JSON-serialisable for the hosted backend


def test_scoring_with_a_perfect_fake_model():
    def oracle(state, qs):
        it = next(i for i in ITEMS if i["prompt"] == state["problem"])
        gold = "ABCD"[it["options"].index(it["answer"])]
        return ({"mc": {"choice": gold, "probabilities": {k: float(k == gold) for k in "ABCD"}},
                 "verify": {"noul": 1.0 if it["shown_is_correct"] else 0.0},
                 "size": {"probabilities": {str(k): float(k == it["score_level"]) for k in range(5)}}},
                "oracle", 0.0, 0)

    ITEMS = stem_synth.build(n=24)
    rows = run(stem_synth, ITEMS, oracle)
    assert all(r["mc_correct"] and r["verify_correct"] and r["score_correct"] for r in rows)
    assert all(r["mc_conf"] == 1.0 and r["score_abs_err"] == 0 for r in rows)
