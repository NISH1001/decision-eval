"""Model backends. Each returns predict(state, questions) -> (answers, model id, cost in USD, output tokens),
with answers in the Laya / Jev decisions format: {qid: {"choice": ..., "probabilities": {...}} | {"noul": p}}.
"""

import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="laya")

# backend key -> display name and what the results page says about each model
MODELS = {
    "laya": dict(name="Laya", checkpoint="convaiinnovations/laya (english)", encoder="ModernBERT-large",
                 params="~395M", runtime="local · PyTorch"),
    "laya-typed": dict(name="Laya typed", checkpoint="convaiinnovations/laya (typed-decisions)",
                       encoder="ModernBERT-large", params="~395M", runtime="local · PyTorch"),
    "gliner": dict(name="GLiNER", checkpoint="fastino/GLiNER2.5-Decide (fp32 ONNX)", encoder="DeBERTa-v3-large",
                   params="~435M", runtime="local · ONNX Runtime"),
    "jev": dict(name="Jev", checkpoint="typesafe/jev-1.14 (OpenRouter)", encoder="undisclosed",
                params="undisclosed", runtime="hosted · OpenRouter"),
}


def get(key, jev_model="typesafe/jev-1.14"):
    """(predict, default worker count) for a backend key."""
    if key == "laya":
        return laya_backend("english"), 1
    if key == "laya-typed":
        return laya_backend("typed-decisions"), 1
    if key == "gliner":
        return gliner_backend(), 1
    if key == "jev":
        return jev_backend(jev_model), 8
    raise SystemExit(f"unknown backend {key!r}; available: {', '.join(MODELS)}")


def laya_backend(checkpoint="english"):
    from laya import Router

    router = Router()
    router.preload([checkpoint])

    def predict(state, questions):
        res = router.predict(state, questions, model=checkpoint)
        return res["answers"], res["routing"]["model"], 0.0, 0

    return predict


def jev_backend(model="typesafe/jev-1.14"):
    key = os.environ.get("OPENROUTER_API_KEY")
    for env_file in (".env.local", ".env"):  # the environment wins, then .env.local, then .env
        if key or not Path(env_file).exists():
            continue
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'") or None
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (environment, .env.local or .env; see .env.example)")

    def predict(state, questions):
        body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
        for attempt in range(4):
            req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    res = json.load(resp)
                usage = res.get("usage", {})
                return res["answers"], res["model"], usage.get("cost", 0.0), usage.get("output_tokens", 0)
            except Exception as e:  # rate limits / transient errors
                if attempt == 3:
                    raise
                print(f"  retry after {e}")
                time.sleep(2 ** attempt)

    return predict


def gliner_backend(repo="nishparadox/gliner2.5-decide-onnx", onnx_file="model.onnx"):
    """GLiNER2.5-Decide via its reference ONNX runtime, adapted to the Laya/Jev question format."""
    import importlib.util

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, allow_patterns=[onnx_file, "tokenizer.json", "gliner_onnx.py"]))
    spec = importlib.util.spec_from_file_location("gliner_onnx", path / "gliner_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gliner_onnx"] = mod  # its dataclasses look the module up here
    spec.loader.exec_module(mod)
    model = mod.GlinerOnnx(str(path / onnx_file), str(path / "tokenizer.json"))

    def clean(t):  # "(" and ")" are reserved in gliner2 prompts
        return t.replace("(", "[").replace(")", "]")

    def predict(state, questions):
        tasks, back = [], {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                # the option text itself is the label; map it back to the letter afterwards
                back[qid] = {clean(v): k for k, v in q["criteria"].items()}
                tasks.append(mod.Task(qid, dict.fromkeys(back[qid]), clean(q["instructions"])))
            elif q["type"] == "noul":
                tasks.append(mod.Task(qid, {"yes": "the statement is correct", "no": "the statement is wrong"},
                                      clean(q["instructions"])))
            else:  # score: an ordinal task over the bins, read back as bin indices
                back[qid] = {clean(c): str(i) for i, c in enumerate(q["criteria"])}
                tasks.append(mod.Task(qid, dict.fromkeys(back[qid]), clean(q["instructions"])))
        probs = model.probabilities(clean(" ".join(map(str, state.values()))), tasks)
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"noul": probs[qid]["yes"]}
                continue
            pr = {back[qid][lab]: p for lab, p in probs[qid].items()}
            answers[qid] = {"choice": max(pr, key=pr.get), "probabilities": pr}
        return answers, f"{repo}/{onnx_file}", 0.0, 0

    return predict
