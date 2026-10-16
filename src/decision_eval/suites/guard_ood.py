"""guard-ood: reflex-guard's out-of-distribution guardrail set (988 prompts from Aegis 2.0, XSTest v2 and
OR-Bench). Nothing is tuned here: every model keeps the thresholds from the core dev split."""

from ._guard import CATEGORIES, DEFAULT_N, DEFAULT_SEED, IMPORTED, KINDS, MODES, TASK, build  # noqa: F401

ID = "guard-ood"
TITLE = "Guardrails: OOD"
BLURB = ("988 prompts from Aegis 2.0, XSTest v2 and OR-Bench, scored with the thresholds from the core dev split: "
         "how well each guard transfers.")
