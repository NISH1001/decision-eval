"""redef: does this code change introduce a bug? (ReDef, Nam et al., FSE 2026, MIT).

Each item is one real function-level change from a large C/C++ project (Linux, Chromium, LLVM, ...), shown as a
unified diff of the function with its file path and name. Nothing else is sent: not the commit message, the revert,
or ReDef's labelling notes
(redef-commit asks the same items with the commit message added). Both modes are scored against ReDef's own labels:
  noul:   does this change introduce a bug?   gold `defective_modification` (1 = the change was later reverted
          for a technical problem); 100 buggy + 100 clean, so a constant answer scores 50% balanced accuracy
  choice: what kind of change is this?        gold `commit_type`, ReDef's 2 x 2 of intent (bug fix or not) and
          outcome (works or was reverted as buggy); chance 25%
ReDef assigned both labels from each commit's and revert's messages, with three unanimous GPT-4o votes and a
manual audit (92% of 100 sampled defective labels correct). Its 9 MIX items (votes split between the two buggy
kinds) are left out, since they have no single commit_type.

Items come from ReDef's validation split (1,344 changes; train is never used). Figshare serves the data only to a
browser, so download it once by hand and point REDEF_DIR at the unzipped folder (it must hold 1_valid.jsonl):
  https://figshare.com/s/4f202bc0921e26b41dc2  ->  ReDef.zip
The items as generated are committed in results/redef/items.json.
"""

import difflib
import json
import os
import random
from pathlib import Path

ID = "redef"
TITLE = "ReDef: code change"
BLURB = ("Real function-level changes from 22 C/C++ projects, shown as a diff: does it introduce a bug (noul), and "
         "what kind of change is it (choice)? 100 buggy + 100 clean from the validation split of ReDef (FSE 2026).")
MODES = ["choice", "noul"]
DEFAULT_N = 200  # half buggy, half clean
DEFAULT_SEED = 0
SOURCE = "https://figshare.com/s/4f202bc0921e26b41dc2"
DISPLAY_CHARS = 1200

# ReDef's commit_type, worded from the definitions in its labelling prompts; options are always in this order
KINDS = {
    "DEFECTIVE_TO_CLEAN": "a bug fix that works",
    "CLEAN_TO_CLEAN": "an improvement (feature, refactor, optimization or cleanup) that works",
    "CLEAN_TO_DEFECTIVE": "an improvement (feature, refactor, optimization or cleanup) that introduces a bug",
    "DEFECTIVE_TO_DEFECTIVE": "a bug-fix attempt that fails or breaks something",
}
CATEGORIES = ["bug fix, works", "improvement, works", "improvement, buggy", "bug fix, buggy"]
GROUPS = {"clean": CATEGORIES[:2], "buggy": CATEGORIES[2:]}
_CAT = dict(zip(KINDS, CATEGORIES))
OPTIONS = list(KINDS.values())


def _rows():
    d = Path(os.environ.get("REDEF_DIR", Path.home() / ".cache" / "decision-eval" / "redef")).expanduser()
    f = d / "1_valid.jsonl"
    if not f.exists():
        raise SystemExit(f"no {f}: download ReDef.zip from {SOURCE} in a browser, unzip it, and set REDEF_DIR "
                         "to the folder holding 1_valid.jsonl")
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def diff(before, after):
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "before", "after", lineterm="", n=3))


def _stratified(rows, n, rng):
    """n rows drawn in proportion to commit_type (largest remainder, so the counts sum to n), in dataset order."""
    by = {}
    for r in rows:
        by.setdefault(r["commit_type"], []).append(r)
    quota = {k: n * len(v) / len(rows) for k, v in by.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    picked = [r for k in sorted(by) for r in rng.sample(by[k], take[k])]
    return sorted(picked, key=rows.index)


def build(n=DEFAULT_N, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    rows = [r for r in _rows() if r["commit_type"] in KINDS]
    buggy = _stratified([r for r in rows if r["defective_modification"] == 1], n // 2, rng)
    clean = _stratified([r for r in rows if r["defective_modification"] == 0], n - n // 2, rng)
    picked = buggy + clean
    rng.shuffle(picked)
    items = []
    for i, r in enumerate(picked):
        d = diff(r["function_before"], r["function_after"])
        is_buggy = r["defective_modification"] == 1
        items.append(dict(id=i, category=_CAT[r["commit_type"]], project=r["project"], file_path=r["file_path"],
                          function_name=r["function_name"], commit=r["commit"], date=r["date"],
                          commit_message=r["commit_message"],
                          prompt=f"{r['file_path']} :: {r['function_name']}\n{d}", diff=d,
                          options=OPTIONS, answer=KINDS[r["commit_type"]],
                          shown_answer="introduces a bug", shown_is_correct=is_buggy, score_level=None))
    return items


def state(item):
    return {"file": item["file_path"], "function": item["function_name"], "diff": item["diff"]}


def instructions(item):
    return {
        "choice": "What kind of change is this diff?",
        "noul": "Does this change introduce a bug?",
    }
