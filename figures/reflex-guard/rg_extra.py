"""Run reflex-guard's eval (scripts/eval.py, unchanged) for two more models on a copy of the repo:
d1-ultra (JevGuard pointed at liquid/d1-ultra) and laya-typed (LayaGuard on the typed-decisions checkpoint)."""
import sys
from pathlib import Path

repo = Path(sys.argv[1]); sys.path[:0] = [str(repo / "scripts"), str(repo)]
import os
for line in Path("/Users/npantha/dev/nish/projects/decision-eval/.env").read_text().splitlines():  # keys, never printed
    k, _, v = line.partition("=")
    if k.strip() == "OPENROUTER_API_KEY" and v.strip():
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
import eval as ev  # reflex-guard's scripts/eval.py
from reflexguard.impls.jev import JevGuard
from reflexguard.impls.laya import LayaGuard

D1 = "liquid/d1-ultra-20261015"


class TypedRouter:  # route every call to Laya's typed-decisions checkpoint
    def __init__(self, router):
        self.r = router
    def route(self, state, questions):
        return {"model": "typed-decisions"}
    def load(self, model):
        return self.r.load(model)
    def predict(self, state, questions):
        return self.r.predict(state, questions, model="typed-decisions")
    def predict_batch(self, reqs):
        return self.r.predict_batch(reqs)


base = ev.make_guard


def make_guard(model, mode, categories, shared):
    m = ev.ALL_MODES if mode == ev.PER_CHUNK else ev.MODES[mode]
    if model == "d1-ultra":
        return JevGuard(categories=categories, model=D1, mode=m, **ev.PATIENT)
    if model == "laya-typed":
        if "router" not in shared:
            from laya import Router
            shared["router"] = Router(); shared["router"].preload(["typed-decisions"])
        return LayaGuard(categories=categories, mode=m, router=TypedRouter(shared["router"]))
    return base(model, mode, categories, shared)


ev.make_guard = make_guard
ev.REMOTE = ev.REMOTE | {"d1-ultra"}
ev.REMOTE_CONCURRENCY = ev.REMOTE_CONCURRENCY | {"d1-ultra": 16}
ev.parse = (lambda f: lambda model: (model, {}) if model in ("d1-ultra", "laya-typed") else f(model))(ev.parse)

for s in sys.argv[3].split(","):
    ev.use_set(s)
    ev.run(sys.argv[2].split(","), int(sys.argv[4]) if len(sys.argv) > 4 else None)
