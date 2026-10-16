"""judgebench: pick the objectively correct one of two long responses (JudgeBench, ScalerLab, 2024, MIT).

Each item is a hard question with two responses from the same model, exactly one of which is correct (label A>B
or B>A). Both splits are used, in order: `gpt` (350 pairs, GPT-4o responses) then `claude` (270 pairs, Claude 3.5
Sonnet responses), 620 items. The question and both responses go in the state, so they are sent once and
every model receives the same request; the options are just "response A" / "response B" (Laya cannot take
~400-token options, and long text in the options would be sent twice).
  choice: which response correctly answers the question?                         (chance 50%)
  noul:   is response X a correct answer? (X is the correct one on exactly half)  (chance 50%)
Local encoders read only the start of long inputs (~512 tokens); the results page says how much each one saw.
Source: https://huggingface.co/datasets/ScalerLab/JudgeBench
"""

import json
import random
import urllib.request

ID = "judgebench"
TITLE = "JudgeBench (long-form)"
BLURB = ("Hard questions with two long responses (GPT-4o or Claude 3.5 Sonnet), exactly one correct: pick it (choice) "
         "or verify one (noul). "
         "All 620 pairs of ScalerLab/JudgeBench (2024).")
MODES = ["choice", "noul"]
DEFAULT_N = 620  # both splits
DEFAULT_SEED = 0
SPLITS = ["gpt", "claude"]
SOURCE = "ScalerLab/JudgeBench"
# the results page shows each question and response cut to this many characters (scores use the full text)
DISPLAY_CHARS = 700

_MMLU = ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history",
         "law", "math", "other", "philosophy", "physics", "psychology"]
CATEGORIES = [f"mmlu-pro-{s}" for s in _MMLU] + ["livebench-reasoning", "livebench-math", "livecodebench"]
GROUPS = {"knowledge (MMLU-Pro)": CATEGORIES[:14], "reasoning & math (LiveBench)": CATEGORIES[14:16],
          "coding (LiveCodeBench)": CATEGORIES[16:]}


def _split(split):
    """Every row of one split, in dataset order, from the Hugging Face datasets server."""
    rows = []
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={SOURCE}&config=default&split={split}"
               f"&offset={len(rows)}&length=100")
        page = [r["row"] for r in json.load(urllib.request.urlopen(url, timeout=60))["rows"]]
        if not page:
            return rows
        rows += page


def _rows(n):
    """The first n rows of the splits in SPLITS order (gpt, then claude)."""
    rows = []
    for split in SPLITS:
        if len(rows) >= n:
            break
        rows += _split(split)
    return rows[:n]


def build(n=DEFAULT_N, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    rows = _rows(n)
    show_correct = set(rng.sample(range(len(rows)), len(rows) // 2))  # noul shows the correct response on exactly half
    items = []
    for i, r in enumerate(rows):
        if r["source"] not in CATEGORIES:
            raise SystemExit(f"unexpected JudgeBench source {r['source']!r}")
        a, b = r["response_A"], r["response_B"]
        correct = "A" if r["label"] == "A>B" else "B"
        shown = correct if i in show_correct else ("B" if correct == "A" else "A")
        items.append(dict(id=i, pair_id=r["pair_id"], category=r["source"], prompt=r["question"], options=[a, b],
                          answer=a if correct == "A" else b, shown_label=shown,
                          shown_answer=a if shown == "A" else b, shown_is_correct=shown == correct,
                          score_level=None, response_model=r["response_model"]))
    return items


def state(item):
    return {"question": item["prompt"], "response_A": item["options"][0], "response_B": item["options"][1]}


def criteria(item):
    return {"A": "response A", "B": "response B"}


def instructions(item):
    return {
        "choice": "Which response correctly answers the question?",
        "noul": f"Is response {item['shown_label']} a correct answer to the question?",
    }
