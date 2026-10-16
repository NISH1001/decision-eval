"""guard-core: reflex-guard's core guardrail set, test split (322 prompts: 104 harmful, 94 benign, 124 hard
negatives). Science-domain prompts, jailbreaks and prompt injections, with benign twins and look-alike hard
negatives. Thresholds are picked on the core dev split."""

from ._guard import CATEGORIES, DEFAULT_N, DEFAULT_SEED, IMPORTED, KINDS, MODES, TASK, build  # noqa: F401

ID = "guard-core"
TITLE = "Guardrails: core"
BLURB = ("reflex-guard's core guardrail set, test split: 322 prompts (104 harmful, 94 benign, 124 hard negatives) over "
         "8 harm categories. Thresholds tuned on the dev split.")
