"""Shared definition of the guardrail suites (imported from reflex-guard, not run here).

Each item is a prompt to flag or pass. Every guard asks all 8 harm categories in each mode; a prompt's P(flag)
in a mode is its highest category score, and it is flagged when that reaches the model's threshold for the mode.
See decision_eval/reflexguard.py for the import and the threshold protocol.
"""

TASK = "flag"
IMPORTED = True
MODES = ["noul", "choice", "score"]
CATEGORIES = None  # each set has its own 8 categories: read from results/<suite>/categories.json on import
KINDS = ["harmful", "hard_negative", "benign"]  # positives first
DEFAULT_N = None
DEFAULT_SEED = None


def build(n=None, seed=None):
    raise SystemExit("guard suites are imported from reflex-guard: "
                     "decision-eval import-reflexguard --repo <path to reflex-guard>")
