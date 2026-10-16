# decision-eval

Benchmark zero-shot decision models on identical requests: **accuracy, calibration, latency and cost**, suite
by suite. Results: <https://nishparadox.com/research/decision-eval/>.

A decision model is a zero-shot classifier with a typed interface. Given a text and a set of questions, it
returns calibrated probabilities over the answers you allow, in one of three modes:

| mode | you ask | it returns |
|---|---|---|
| `choice` | pick one of *k* options | a distribution over the options |
| `noul` | a yes/no question | P(yes) |
| `score` | a level on an ordered scale | a distribution over the levels |

decision-eval asks every item of a suite in every mode the suite supports, in **one request per item**, and
sends byte-identical requests to every model.

## Models

| backend | model | runs |
|---|---|---|
| `laya` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (english) | local, PyTorch |
| `laya-typed` | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | local, PyTorch |
| `gliner` | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) via [an fp32 ONNX export](https://huggingface.co/nishparadox/gliner2.5-decide-onnx) | local, ONNX Runtime |
| `jev` | [typesafe/jev-1.14](https://openrouter.ai/typesafe/jev-1.14) | hosted, OpenRouter decisions API |
| `gpt-6-luna` | [gpt-6-luna](https://developers.openai.com/api/docs/guides/decisions) | hosted, OpenAI Decisions API (beta) |

Hosted backends record per item: input and output tokens, the billed cost, the API request id, and the exact
model version the API reports. Jev returns a dated snapshot (`typesafe/jev-1.14-20261014`). OpenAI returns only
the alias, so the recorded id adds the model's `created` date from `GET /v1/models/gpt-6-luna`; its cost is
input tokens × the list price (the Decisions endpoint bills input only).

## Suites

| suite | modes | items |
|---|---|---|
| `stem-synth` | choice, noul, score | 180 generated math and physics problems in 12 tiers, single-digit arithmetic to calculus, complex analysis and E&M; distractors are common mistakes |
| `guard-core` | flag (noul, choice, score) | reflex-guard's core guardrail set, test split: 322 prompts (104 harmful, 94 benign, 124 hard negatives), 8 harm categories; thresholds tuned on its dev split |
| `guard-ood` | flag (noul, choice, score) | 988 prompts from Aegis 2.0, XSTest v2 and OR-Bench, scored with the core-dev thresholds (nothing tuned) |

The guardrail suites are imported from [reflex-guard](https://github.com/NISH1001/reflex-guard)'s eval caches
(`decision-eval import-reflexguard --repo <path>`), not re-run: a prompt's P(flag) in a mode is its highest category
score, positives are harmful prompts, and per model and mode the threshold maximising balanced accuracy on the core
dev split is applied unchanged to core test and OOD. The import reproduces reflex-guard's own report exactly; recall,
precision, F1, accuracy, false-positive rates and AUC are computed the same way for every model.

## Usage

Without cloning, straight from GitHub (results go to `./results/`):

```bash
R="git+https://github.com/NISH1001/decision-eval"
uvx --from "$R" decision-eval suites
uvx --from "decision-eval[laya] @ $R" decision-eval run --suite stem-synth --backend laya
uvx --from "$R" decision-eval run --suite stem-synth --backend gpt-6-luna    # OPENAI_API_KEY in the environment
```

From a clone:

```bash
cp .env.example .env            # hosted backends: OPENROUTER_API_KEY (jev), OPENAI_API_KEY (gpt-6-luna)

uv run decision-eval suites
uv run --extra laya   decision-eval run --suite stem-synth --backend laya
uv run --extra laya   decision-eval run --suite stem-synth --backend laya-typed
uv run --extra gliner decision-eval run --suite stem-synth --backend gliner
uv run                decision-eval run --suite stem-synth --backend jev
uv run                decision-eval run --suite stem-synth --backend gpt-6-luna

uv run --extra figures decision-eval figures --suite stem-synth     # paper figures (PDF/PNG) + LaTeX table
uv run decision-eval export --suite stem-synth --post <post.md>     # embed results in the results page
```

`run` writes `results/<suite>/items.json` (the items, deterministic in `-n` and `--seed`) and
`results/<suite>/results_<backend>.csv` (one row per item: every mode's prediction, probability, correctness,
latency, billed cost). It refuses to mix results from different item sets. `figures` writes to
`results/<suite>/figures/`.

## Metrics

- **choice**: accuracy; chance 1/*k*.
- **noul**: balanced accuracy (mean of the accuracies on correct and on wrong shown answers, so a constant
  answer scores exactly 50%) and AUC.
- **score**: exact-level accuracy and mean absolute level error; chance 1/levels. Levels are usually
  unbalanced, so the figures also show an always-the-most-common-level baseline.
- **calibration**: reliability diagram and ECE over the probability of the picked option.
- **latency**: wall-clock per item with every question answered; hosted backends include the network.
- **cost**: what the run billed (from the API's usage field).

Intervals are 95% bootstrap over items (1,000 resamples).

## Adding a suite

A suite is a module in `src/decision_eval/suites/` exposing `ID`, `TITLE`, `BLURB`, `MODES`, `CATEGORIES`,
optional `GROUPS` and `SCORE_LABELS`, and `build(n, seed)`, `state(item)`, `instructions(item)`; see
[`suites/__init__.py`](src/decision_eval/suites/__init__.py). Register it in `SUITES`. The runner, metrics,
figures and the results page work from that interface, so a suite with only `noul` (say, a guardrail suite)
needs no other changes.

## License

MIT
