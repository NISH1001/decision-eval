"""Ask every item of a suite, in every mode it supports, in one request per item, and score the answers."""

import time
from concurrent.futures import ThreadPoolExecutor

LABELS = "ABCDEFGH"


def _criteria(suite, item):
    """Choice criteria: by default the options themselves under A, B, ...; a suite whose options are long (the text
    is in the state) supplies short labels instead via criteria(item). Keys always map to options in order."""
    if hasattr(suite, "criteria"):
        return suite.criteria(item)
    return {LABELS[k]: o for k, o in enumerate(item["options"])}


def request(suite, item):
    """(state, questions) for one item. Question keys are fixed per mode: mc, verify, size."""
    ins = suite.instructions(item)
    questions = {}
    if "choice" in suite.MODES:
        questions["mc"] = dict(type="choice", instructions=ins["choice"], criteria=_criteria(suite, item))
    if "noul" in suite.MODES:
        questions["verify"] = dict(type="noul", instructions=ins["noul"])
    if "score" in suite.MODES:
        questions["size"] = dict(type="score", criteria=suite.SCORE_LABELS, instructions=ins["score"])
    return suite.state(item), questions


def score(suite, item, answers):
    """Per-mode correctness and probabilities for one item; columns of modes the suite lacks are left out."""
    row = {}
    if "choice" in suite.MODES:
        crit = _criteria(suite, item)
        gold = LABELS[item["options"].index(item["answer"])]
        mc = answers["mc"]
        row.update(options=" | ".join(f"{k}:{v}" for k, v in crit.items()), gold_label=gold,
                   mc_pred=mc["choice"], mc_pred_text=item["options"][LABELS.index(mc["choice"])],
                   mc_correct=mc["choice"] == gold,
                   mc_conf=max(mc["probabilities"].values()), mc_p_gold=mc["probabilities"][gold])
    if "noul" in suite.MODES:
        p_yes = float(answers["verify"]["noul"])  # noul answers come back as P(yes)
        row.update(verify_p_yes=p_yes, verify_pred=p_yes >= 0.5,
                   verify_correct=(p_yes >= 0.5) == item["shown_is_correct"])
    if "score" in suite.MODES:
        sp = {int(k): v for k, v in answers["size"]["probabilities"].items()}
        pred, gold = max(sp, key=sp.get), item["score_level"]
        row.update(score_pred=pred, score_correct=pred == gold, score_abs_err=abs(pred - gold),
                   score_p_gold=sp.get(gold, 0.0))
    return row


def run_one(suite, item, predict):
    state, questions = request(suite, item)
    t0 = time.perf_counter()
    answers, model, usage = predict(state, questions)
    dt = time.perf_counter() - t0
    row = {**item, **score(suite, item, answers), "latency_s": dt, "input_tokens": usage["input_tokens"],
           "output_tokens": usage["output_tokens"], "cost_usd": usage["cost_usd"], "request_id": usage["request_id"],
           "model": model}
    marks = " ".join(f"{k}={'✓' if row[c] else '✗'}" for k, c in
                     (("mc", "mc_correct"), ("ver", "verify_correct"), ("size", "score_correct")) if c in row)
    print(f"[{item['id']:3d}] {str(item['category']):<26} {marks} {dt * 1000:5.0f}ms  {item['prompt'][:70]}")
    return row


def run(suite, items, predict, workers=1):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: run_one(suite, it, predict), items))
