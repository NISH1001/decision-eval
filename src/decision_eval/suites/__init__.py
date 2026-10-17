"""Suites: fixed item sets, each asked in every question mode it supports.

A suite is a module with:
    ID, TITLE, BLURB         identity and a one-line description (shown on the results page)
    MODES                    subset of ["choice", "noul", "score"], in that order
    CATEGORIES, GROUPS       item categories, and an optional {group: [category]} grouping for figures
    SCORE_LABELS             ordered levels for score mode (omit if the suite has no score mode)
    DEFAULT_N, DEFAULT_SEED  default size and seed
    build(n, seed)           -> list of items: {id, category, prompt, options, answer, shown_answer,
                                shown_is_correct, score_level, ...}; fields for unused modes may be None
    state(item)              -> the state dict sent to the model
    instructions(item)       -> {mode: instruction text}; the runner attaches options and score labels
    criteria(item)           -> optional short choice labels, for suites whose options are long (judgebench)

A suite with TASK = "flag" (the guardrail suites) is binary detection instead: each mode gives P(flag)
per item, and IMPORTED = True means its results come from elsewhere (see reflexguard.py), not `run`.

To add a suite, write the module and register it below.
"""

from . import guard_core, guard_ood, judgebench, redef, redef_commit, redef_message, stem_synth

SUITES = {m.ID: m for m in (stem_synth, judgebench, redef, redef_commit, redef_message, guard_core, guard_ood)}


def get(suite_id):
    try:
        return SUITES[suite_id]
    except KeyError:
        raise SystemExit(f"unknown suite {suite_id!r}; available: {', '.join(SUITES)}") from None
