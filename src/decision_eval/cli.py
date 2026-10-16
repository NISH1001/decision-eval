"""decision-eval: benchmark zero-shot decision models suite by suite.

    decision-eval suites                                         # list suites
    decision-eval run --suite stem-synth --backend jev           # ask every item; writes results/<suite>/
    decision-eval figures --suite stem-synth                     # paper figures + LaTeX table from all results
    decision-eval export --suite stem-synth --post <post.md>     # embed results in the results page
    decision-eval import-reflexguard --repo <reflex-guard>       # guardrail suites from reflex-guard's eval

Results live in results/<suite>/: items.json (the items as generated) and results_<backend>.csv.
"""

import argparse
import json
from pathlib import Path

from . import backends, suites
from .runner import run


def _results_dir(args):
    return Path(args.results) / args.suite


def _load(args):
    import pandas as pd

    suite, out = suites.get(args.suite), _results_dir(args)
    items_path = out / "items.json"
    if not items_path.exists():
        raise SystemExit(f"no {items_path}; run `decision-eval run --suite {args.suite} ...` first")
    items = json.loads(items_path.read_text())
    dfs = {k: pd.read_csv(out / f"results_{k}.csv") for k in backends.MODELS if (out / f"results_{k}.csv").exists()}
    if not dfs:
        raise SystemExit(f"no results_<backend>.csv in {out}")
    return suite, items, dfs


def cmd_suites(args):
    for s in suites.SUITES.values():
        print(f"{s.ID:<16} {', '.join(s.MODES):<20} {s.BLURB}")


def cmd_run(args):
    import pandas as pd

    suite, out = suites.get(args.suite), _results_dir(args)
    if getattr(suite, "IMPORTED", False):
        raise SystemExit(f"{suite.ID} is imported, not run here: decision-eval import-reflexguard --repo <path>")
    out.mkdir(parents=True, exist_ok=True)
    items = suite.build(args.n or suite.DEFAULT_N, args.seed if args.seed is not None else suite.DEFAULT_SEED)
    items_path = out / "items.json"
    if items_path.exists() and json.loads(items_path.read_text()) != items:
        raise SystemExit(f"{items_path} holds a different item set (other -n / --seed?); move it aside first, "
                         "so results from different item sets never mix")
    items_path.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    predict, workers = backends.get(args.backend, args.jev_model)
    df = pd.DataFrame(run(suite, items, predict, workers=args.workers or workers))
    df.to_csv(out / f"results_{args.backend}.csv", index=False)
    print(f"wrote {out / f'results_{args.backend}.csv'} ({len(df)} items)")
    if df.cost_usd.sum():
        print(f"billed USD {df.cost_usd.sum():.6f}")


def cmd_figures(args):
    from .figures import make_figures

    suite, items, dfs = _load(args)
    if getattr(suite, "TASK", "answer") == "flag":
        raise SystemExit("paper figures for flag suites come from the results page's Paper view (SVG / PNG per figure)")
    make_figures({backends.MODELS[k]["name"]: d for k, d in dfs.items()}, suite, items, _results_dir(args) / "figures")


def cmd_export(args):
    from .export import embed, suite_payload

    suite, items, dfs = _load(args)
    if getattr(suite, "TASK", "answer") == "flag":
        from .export import flag_payload
        from .reflexguard import COSTS
        cats = json.loads((_results_dir(args) / "categories.json").read_text())
        payload = flag_payload(suite, items, dfs, {k: v for (s, k), v in COSTS.items() if s == suite.ID}, list(cats))
    else:
        payload = suite_payload(suite, items, dfs)
    data, size = embed(Path(args.post).expanduser(), payload)
    print(f"embedded {suite.ID} ({len(items)} items × {len(dfs)} models); page now carries "
          f"{[s['id'] for s in data['suites']]}, {size / 1024:.0f} KB of data")


def cmd_import_reflexguard(args):
    from .reflexguard import import_all

    commit, written = import_all(args.repo, args.results, {})
    for suite, keys in written.items():
        print(f"{suite}: {', '.join(keys)}  (reflex-guard @ {commit})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="decision-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="results root (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("suites", help="list suites").set_defaults(fn=cmd_suites)
    r = sub.add_parser("run", help="ask every item of a suite with one backend")
    r.add_argument("--suite", required=True)
    r.add_argument("--backend", required=True, choices=list(backends.MODELS))
    r.add_argument("-n", type=int, help="number of items (default: the suite's)")
    r.add_argument("--seed", type=int, help="item seed (default: the suite's)")
    r.add_argument("--workers", type=int, help="parallel requests (default: 8 for jev, 1 for local models)")
    r.add_argument("--jev-model", default="typesafe/jev-1.14")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("figures", help="paper figures and LaTeX table from every backend's results")
    f.add_argument("--suite", required=True)
    f.set_defaults(fn=cmd_figures)
    e = sub.add_parser("export", help="embed a suite's results in the results page")
    e.add_argument("--suite", required=True)
    e.add_argument("--post", required=True, help="path to the decision-eval Jekyll post")
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import-reflexguard", help="import the guardrail suites from a reflex-guard checkout")
    i.add_argument("--repo", required=True, help="path to the reflex-guard repository")
    i.set_defaults(fn=cmd_import_reflexguard)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
