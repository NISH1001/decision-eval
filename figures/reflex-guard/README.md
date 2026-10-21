# reflex-guard overview figures with d1-ultra and Laya typed

The overview (`core-overview.png`, `ood-overview.png`), per-category (`*-per-category.png`) and paper figures
(`paper_auc`, `paper_rates`, `paper_tradeoff`, `paper_latency`, PNG and PDF) are reflex-guard's own figures (`scripts/plot_eval.py` at
[89fa938](https://github.com/NISH1001/reflex-guard/tree/89fa938), unchanged) with two more models: Liquid AI's
d1-ultra (`liquid/d1-ultra-20261015`) and Laya typed (`convaiinnovations/laya@7b928d8`, typed-decisions,
laya 0.3.20). They are used in [reflex-guard issue #1](https://github.com/NISH1001/reflex-guard/issues/1).

- `rg_extra.py` scored the two models with reflex-guard's `scripts/eval.py` on a copy of the repo, patching only
  `make_guard`: d1-ultra as `JevGuard(model="liquid/d1-ultra-20261015")`, Laya typed as `LayaGuard` routed to the
  typed-decisions checkpoint.
- `rg_plot.py` drew the figures (`plot` and `paper`, as `eval.py` calls them), adding the two models to `STYLE` with the two unused Okabe–Ito hues (orange
  `#E69F00`, yellow `#F0E442`), each with its own hatch and marker.

Run as `uv run --extra laya --extra jev python rg_extra.py <reflex-guard copy> d1-ultra,laya-typed core,ood`, then
`uv run --extra laya python rg_plot.py <reflex-guard copy> <out dir>`.
