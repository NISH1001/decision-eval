"""Model backends.

Each backend is predict(state, questions) -> (answers, model id, usage), where answers are in the Laya / Jev
decisions format, {qid: {"choice": ..., "probabilities": {...}} | {"noul": p}}, and usage is
{input_tokens, output_tokens, cost_usd, request_id} (None where a backend has no such notion).
"""

import datetime
import json
import os
import sys
import time
import urllib.error
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
    "gpt-6-luna": dict(name="GPT-6 Luna", checkpoint="openai/gpt-6-luna (Decisions API, beta)", encoder="undisclosed",
                       params="undisclosed", runtime="hosted · OpenAI"),
}

# OpenAI bills the Decisions endpoint for input tokens only (no output, cache-read or cache-write charges).
# List price for gpt-6-luna as of 2026-10-16: https://developers.openai.com/api/docs/guides/decisions
OPENAI_DECISIONS_USD_PER_INPUT_TOKEN = {"gpt-6-luna": 0.10 / 1e6}

def dated(model_id, run_date=None):
    """`typesafe/jev-1.14-20261014` -> `typesafe/jev-1.14-20261014 (snapshot 2026-10-14, run 2026-10-16)`."""
    import re

    run_date = run_date or datetime.date.today().isoformat()
    m = re.search(r"-(\d{4})(\d{2})(\d{2})$", model_id)
    return f"{model_id} (snapshot {m[1]}-{m[2]}-{m[3]}, run {run_date})" if m else f"{model_id} (run {run_date})"


NO_USAGE = {"input_tokens": None, "output_tokens": None, "cost_usd": 0.0, "request_id": None}


def get(key, jev_model="typesafe/jev-1.14", device="auto"):
    """(predict, default worker count) for a backend key. `device` applies to the local models: auto, cpu, cuda,
    or mps (Laya only)."""
    if key == "laya":
        return laya_backend("english", device), 1
    if key == "laya-typed":
        return laya_backend("typed-decisions", device), 1
    if key == "gliner":
        return gliner_backend(device=device), 1
    if key == "jev":
        return jev_backend(jev_model), 8
    if key == "gpt-6-luna":
        return openai_backend("gpt-6-luna"), 8
    raise SystemExit(f"unknown backend {key!r}; available: {', '.join(MODELS)}")


def _api_key(name):
    """The environment wins, then .env.local, then .env (see .env.example)."""
    key = os.environ.get(name)
    for env_file in (".env.local", ".env"):
        if key or not Path(env_file).exists():
            continue
        for line in Path(env_file).read_text().splitlines():
            if line.startswith(name + "="):
                key = line.split("=", 1)[1].strip().strip("\"'") or None
    if not key:
        raise SystemExit(f"set {name} (environment, .env.local or .env; see .env.example)")
    return key


def _post(url, key, payload, attempts=4):
    """POST JSON with retries on rate limits and transient errors; returns (response json, response headers)."""
    body = json.dumps(payload).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp), resp.headers
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429 or attempt == attempts - 1:
                raise SystemExit(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
            print(f"  retry after HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts - 1:
                raise
            print(f"  retry after {e}")
        time.sleep(2 ** attempt)


def _hf_revision(repo):
    """The commit of the cached Hugging Face snapshot a local model loads from (refs/main), or "unknown"."""
    from huggingface_hub import constants

    ref = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip()[:7] if ref.exists() else "unknown"


def laya_backend(checkpoint="english", device="auto"):
    from importlib.metadata import version

    from laya import Router

    router = Router(device=None if device == "auto" else device)  # auto: cuda, then Apple MPS (Metal), then CPU
    router.preload([checkpoint])
    device = getattr(router.load(checkpoint), "device", "?")  # laya picks cuda, then Apple MPS (Metal), then CPU
    model_id = (f"convaiinnovations/laya@{_hf_revision('convaiinnovations/laya')} ({checkpoint}), laya {version('laya')}, "
                f"{device}")

    def predict(state, questions):
        res = router.predict(state, questions, model=checkpoint)
        return res["answers"], model_id, NO_USAGE

    return predict


def jev_backend(model="typesafe/jev-1.14"):
    """Jev on OpenRouter's decisions API. The response names the dated snapshot (e.g. typesafe/jev-1.14-20261014)
    and reports tokens and the billed cost."""
    key = _api_key("OPENROUTER_API_KEY")

    def predict(state, questions):
        res, _ = _post("https://openrouter.ai/api/alpha/decisions", key,
                       {"model": model, "state": state, "questions": questions})
        u = res.get("usage", {})
        return res["answers"], dated(res["model"]), {"input_tokens": u.get("input_tokens"),
                                              "output_tokens": u.get("output_tokens"),
                                              "cost_usd": u.get("cost", 0.0), "request_id": res.get("id")}

    return predict


def openai_backend(model="gpt-6-luna"):
    """OpenAI's Decisions API (POST /v1/decisions, beta). Same questions, translated to its schema:
    choice -> choice (choices: value + description), noul -> predicate (probability = P(yes)),
    score -> score (levels, lowest first). The state is sent as its JSON text, the same object Jev receives.

    The response names only the model alias (no dated snapshot is published), so the model id recorded is the
    alias plus the model's `created` date from GET /v1/models/<model>. Cost is input tokens x the list price."""
    key = _api_key("OPENAI_API_KEY")
    meta = json.load(urllib.request.urlopen(urllib.request.Request(
        f"https://api.openai.com/v1/models/{model}", headers={"Authorization": f"Bearer {key}"}), timeout=30))
    created = datetime.datetime.fromtimestamp(meta["created"], datetime.UTC).date().isoformat()
    price = OPENAI_DECISIONS_USD_PER_INPUT_TOKEN[model]

    def translate(qid, q):
        if q["type"] == "choice":
            return {"type": "choice", "name": qid, "instructions": q["instructions"],
                    "choices": [{"value": k, "description": v} for k, v in q["criteria"].items()]}
        if q["type"] == "noul":
            return {"type": "predicate", "name": qid, "instructions": q["instructions"]}
        return {"type": "score", "name": qid, "instructions": q["instructions"],
                "levels": [{"label": c, "description": c} for c in q["criteria"]]}

    def predict(state, questions):
        res, headers = _post("https://api.openai.com/v1/decisions", key, {
            "model": model, "input": json.dumps(state, ensure_ascii=False),
            "questions": [translate(qid, q) for qid, q in questions.items()]})
        answers = {}
        for a in res["answers"]:
            if a["type"] == "refusal":
                raise SystemExit(f"{model} refused question {a['name']!r}: {a}")
            if a["type"] == "predicate":
                answers[a["name"]] = {"noul": a["probability"]}
            elif a["type"] == "choice":
                answers[a["name"]] = {"choice": a["choice"],
                                      "probabilities": {p["value"]: p["probability"] for p in a["probabilities"]}}
            else:  # score: probabilities keyed by level index, as the other backends return them
                answers[a["name"]] = {"probabilities": {str(p["value"]): p["probability"] for p in a["probabilities"]}}
        u = res.get("usage", {})
        return answers, f"{res['model']} (created {created}, run {datetime.date.today().isoformat()})", {
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "cost_usd": (u.get("input_tokens") or 0) * price, "request_id": headers.get("x-request-id")}

    return predict


def gliner_backend(repo="nishparadox/gliner2.5-decide-onnx", onnx_file="model.onnx", device="auto"):
    """GLiNER2.5-Decide via its reference ONNX runtime, adapted to the Laya/Jev question format.

    device: cpu, cuda (needs the gliner-gpu extra, i.e. onnxruntime-gpu, and a working CUDA/cuDNN), or auto (cuda when
    its provider is available, else cpu). CoreML is not offered: it supports only part of this graph and its Metal
    compiler fails on it. A requested provider that does not end up active raises instead of falling back silently."""
    import importlib.util

    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    if device not in ("auto", "cpu", "cuda"):
        raise SystemExit(f"gliner: device must be auto, cpu or cuda, not {device!r}")
    if device == "cuda" and not has_cuda:
        raise SystemExit("gliner: --device cuda needs the CUDA provider: install the gliner-gpu extra (onnxruntime-gpu) "
                         f"on a machine with CUDA; available here: {ort.get_available_providers()}")
    path = Path(snapshot_download(repo, allow_patterns=[onnx_file, "tokenizer.json", "gliner_onnx.py"]))
    model_id = f"{repo}@{path.name[:7]}/{onnx_file}"  # snapshot dir name is the commit hash
    spec = importlib.util.spec_from_file_location("gliner_onnx", path / "gliner_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gliner_onnx"] = mod  # its dataclasses look the module up here
    spec.loader.exec_module(mod)
    model = mod.GlinerOnnx(str(path / onnx_file), str(path / "tokenizer.json"))
    # the reference runtime pins the CPU provider; rebuild its session on the requested one
    if device == "cuda" or (device == "auto" and has_cuda):
        model.session = ort.InferenceSession(str(path / onnx_file), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if model.session.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit(f"gliner: CUDA provider failed to initialise (check CUDA/cuDNN); got {model.session.get_providers()}")
    model_id += f", {model.session.get_providers()[0]}"

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
        return answers, model_id, NO_USAGE

    return predict
