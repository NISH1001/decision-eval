"""Embed a suite's results in the decision-eval results page (a self-contained Jekyll post).

The page carries all of its data in one inline <script type="application/json" id="de-data"> block between
<!-- DATA:START --> and <!-- DATA:END -->, as {"suites": [suite, ...]}. Exporting a suite inserts or replaces it
there and leaves the other suites untouched. Summaries come from metrics.summary_table, so the page and the
paper table agree.

Per-suite schema (the page draws everything from it, so a new suite needs no page changes):
    id, title, blurb
    meta:   suite, title, n, modes, categories, groups, chance {mode: p}, score_labels, generated
    items:  [category, prompt, options | null, correct option index | null, shown answer | null,
             shown answer is correct | null, correct score level | null]
    models: [{key, name, checkpoint, encoder, params, runtime, model_id, summary, preds}]
            preds per item: [picked option, top prob, P(correct option), P(yes), predicted level, latency ms]
"""

import json
import math
import re
from datetime import date

from .backends import MODELS
from .metrics import metrics, summary_table

BLOCK = re.compile(r"(<!-- DATA:START -->\n<script type=\"application/json\" id=\"de-data\">\n\{% raw %\})(.*?)"
                   r"(\{% endraw %\}\n</script>\n<!-- DATA:END -->)", re.S)


def _num(v, nd=4):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def suite_payload(suite, items, dfs):
    """dfs: {backend key: results frame}, rows in item order."""
    for k, d in dfs.items():
        assert d.prompt.tolist() == [it["prompt"] for it in items], f"{k}: results do not match the items"
    modes = suite.MODES
    tab = summary_table({MODELS[k]["name"]: d for k, d in dfs.items()}, suite, items).set_index("model")
    ms = metrics(suite, items)
    models = []
    for k, d in dfs.items():
        s = tab.loc[MODELS[k]["name"]]
        summary = {m: [s[ms[m][0]], *s[ms[m][0] + " CI"]] for m in ms}
        if "noul" in modes:
            summary.update(auc=s["Verify AUC"], says_yes=s["Says yes"])
        if "score" in modes:
            summary["mae"] = s["Score-level MAE"]
        summary.update(latency_ms=s["Median latency (ms)"], cost_usd=float(d.cost_usd.sum()))
        preds = []
        for it, r in zip(items, d.to_dict("records")):
            preds.append([
                it["options"].index(r["mc_pred_text"]) if "choice" in modes else None,
                _num(r.get("mc_conf")), _num(r.get("mc_p_gold")), _num(r.get("verify_p_yes")),
                int(r["score_pred"]) if "score" in modes else None,
                round(r["latency_s"] * 1000),
            ])
        models.append({"key": k, **MODELS[k], "model_id": str(d.model.iloc[0]), "summary": summary, "preds": preds})
    meta = {"suite": suite.ID, "title": suite.TITLE, "n": len(items), "modes": modes,
            "categories": suite.CATEGORIES, "groups": getattr(suite, "GROUPS", None),
            "chance": {m: c for m, (_, _, c) in ms.items()},
            "score_labels": getattr(suite, "SCORE_LABELS", None), "generated": date.today().isoformat()}
    return {"id": suite.ID, "title": suite.TITLE, "blurb": suite.BLURB, "meta": meta,
            "items": [[it["category"], it["prompt"], it.get("options"),
                       it["options"].index(it["answer"]) if it.get("options") else None,
                       it.get("shown_answer"), it.get("shown_is_correct"), it.get("score_level")] for it in items],
            "models": models}


def embed(post_path, payload):
    text = post_path.read_text()
    m = BLOCK.search(text)
    if not m:
        raise SystemExit(f"{post_path}: no DATA:START / DATA:END block")
    data = json.loads(m.group(2)) if m.group(2).strip() else {"suites": []}
    data["suites"] = [s for s in data["suites"] if s["id"] != payload["id"]] + [payload]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float)
    if "</" in blob or "{%" in blob or "{{" in blob:
        raise SystemExit("the data contains a sequence that would break the page (</, {% or {{)")
    post_path.write_text(text[:m.start(2)] + blob + text[m.end(2):])
    return data, len(blob)
