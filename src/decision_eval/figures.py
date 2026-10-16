"""Static paper figures (PDF + PNG) and a LaTeX table, for any suite. Needs the `figures` extra."""

import numpy as np

from .metrics import bootstrap_ci, metrics, summary_table

STYLE = {  # Okabe–Ito (colorblind-safe), each with its own hatch so print / greyscale still works
    "Laya": dict(color="#0072B2", hatch="///", marker="o", ls="-"),
    "Laya typed": dict(color="#56B4E9", hatch="\\\\\\", marker="s", ls="--"),
    "GLiNER": dict(color="#009E73", hatch="...", marker="^", ls="-."),
    "Jev": dict(color="#D55E00", hatch="xx", marker="D", ls=":"),
}


def _paper_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300)
    print(f"figure -> {out / stem}.pdf / .png")


def _bars(ax, x, vals, name, i, k, w, err=None):
    st = STYLE.get(name, dict(color="#888888", hatch=""))
    ax.bar(x + (i - (k - 1) / 2) * w, vals, w, color=st["color"], hatch=st["hatch"], edgecolor="#222222",
           linewidth=0.5, label=name, yerr=err, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#222222"),
           zorder=2)


def _short(c):
    return str(c).split(" ", 1)[1] if str(c).split(" ", 1)[0].isdigit() else str(c)


def fig_by_category(dfs, suite, items, out):
    """Per-category score in each of the suite's modes, one panel per mode."""
    import matplotlib.pyplot as plt

    names, cats, ms = list(dfs), suite.CATEGORIES, metrics(suite, items)
    groups = getattr(suite, "GROUPS", None) or {"": cats}
    x, w = np.arange(len(cats)), 0.8 / len(names)
    fig, axes = plt.subplots(len(ms), 1, figsize=(7.16, 2.2 * len(ms)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (mode, (label, fn, chance)), letter in zip(axes, ms.items(), "abcdef"):
        for i, n in enumerate(names):
            d = dfs[n]
            _bars(ax, x, [fn(d[d.category == c]) for c in cats], n, i, len(names), w)
        ax.axhline(chance, color="#222222", ls=(0, (4, 2)), lw=0.8, zorder=3)
        ax.text(-1.45, chance + 0.015, f"chance\n{chance:.0%}", ha="left", va="bottom", fontsize=7, linespacing=0.9)
        if mode == "score":  # a model that always guesses the category's most common level
            d0 = dfs[names[0]]
            maj = [d0[d0.category == c].score_level.value_counts(normalize=True).max() for c in cats]
            ax.scatter(x, maj, marker="_", s=260, color="#222222", lw=1.4, zorder=4, label="most-common-level baseline")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlim(-1.5, len(cats) - 0.5)
        ax.set_ylabel(f"({letter}) {mode}\n{label.lower()}")
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        for b in np.cumsum([len(v) for v in groups.values()])[:-1]:
            ax.axvline(b - 0.5, color="#999999", lw=0.5, zorder=1)
    starts = np.cumsum([0] + [len(v) for v in groups.values()])
    for a0, a1, gname in zip(starts[:-1], starts[1:], groups):
        axes[0].text((a0 + a1 - 1) / 2, 1.08, gname, ha="center", fontsize=7.5, style="italic", color="#444444")
    axes[-1].set_xticks(x, [_short(c) for c in cats], rotation=30, ha="right")
    h, lab = axes[-1].get_legend_handles_labels()
    order = [i for i, s in enumerate(lab) if s in dfs] + [i for i, s in enumerate(lab) if s not in dfs]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=len(lab), frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.8)
    _save(fig, out, "fig1_accuracy_by_category")
    plt.close(fig)


def fig_overall(dfs, suite, items, out):
    """Overall score per mode with 95% bootstrap CIs, chance marked per mode."""
    import matplotlib.pyplot as plt

    names, ms = list(dfs), metrics(suite, items)
    x, w = np.arange(len(ms)), 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, n in enumerate(names):
        vals, errs = [], [[], []]
        for _, fn, _ in ms.values():
            v = fn(dfs[n])
            lo, hi = bootstrap_ci(dfs[n], fn)
            vals.append(v)
            errs[0].append(max(v - lo, 0))
            errs[1].append(max(hi - v, 0))
        _bars(ax, x, vals, n, i, len(names), w, err=errs)
    for xi, (_, _, c) in zip(x, ms.values()):
        ax.plot([xi - 0.45, xi + 0.45], [c, c], color="#222222", ls=(0, (4, 2)), lw=0.8, zorder=3)
    ax.plot([], [], color="#222222", ls=(0, (4, 2)), lw=0.8, label="chance")
    ax.set_xticks(x, [label.replace(" ", "\n", 1) for label, _, _ in ms.values()])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.28))
    _save(fig, out, "fig2_overall_by_mode")
    plt.close(fig)


def fig_calibration(dfs, out):
    """Reliability diagram for choice mode."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot([0.25, 1], [0.25, 1], color="#888888", lw=0.8, label="perfect calibration")
    bins = np.linspace(0.25, 1.0001, 7)
    for n, d in dfs.items():
        st = STYLE.get(n, dict(color="#888888", ls="-", marker="o"))
        idx = np.clip(np.digitize(d.mc_conf, bins) - 1, 0, len(bins) - 2)
        pts = [(d.mc_conf[idx == k].mean(), d.mc_correct[idx == k].mean(), (idx == k).sum())
               for k in range(len(bins) - 1) if (idx == k).any()]
        ece = sum(c * abs(cf - a) for cf, a, c in pts) / len(d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st["color"], ls=st["ls"], lw=1.2,
                marker=st["marker"], ms=4.5, mec="#222222", mew=0.5, label=f"{n} (ECE {ece:.2f})")
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence (top-option probability)")
    ax.set_ylabel("Accuracy")
    ax.grid(color="#dddddd", lw=0.5)
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, out, "fig3_calibration_choice")
    plt.close(fig)


def latex_table(tab, suite, items, out):
    ms = metrics(suite, items)
    cols = [label for label, _, _ in ms.values()] + [c for c in ("Verify AUC", "Score-level MAE") if c in tab]
    lines = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
             "Model & " + " & ".join(cols) + r" & Latency (ms) \\",
             " & " + " & ".join(f"(chance {c * 100:.0f}\\%)" for _, _, c in ms.values())
             + " & " * (len(cols) - len(ms)) + r" & \\", r"\midrule"]
    for _, r in tab.iterrows():
        cells = [f"{r[c] * 100:.1f} [{r[c + ' CI'][0] * 100:.0f}, {r[c + ' CI'][1] * 100:.0f}]" if c + " CI" in r
                 else f"{r[c]:.2f}" for c in cols]
        lines.append(f"{r['model']} & " + " & ".join(cells) + f" & {r['Median latency (ms)']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_summary.tex").write_text("\n".join(lines) + "\n")


def make_figures(dfs, suite, items, out):
    _paper_rc()
    out.mkdir(parents=True, exist_ok=True)
    fig_by_category(dfs, suite, items, out)
    fig_overall(dfs, suite, items, out)
    if "choice" in suite.MODES:
        fig_calibration(dfs, out)
    tab = summary_table(dfs, suite, items)
    latex_table(tab, suite, items, out)
    flat = tab.drop(columns=[c for c in tab.columns if c.endswith(" CI")])
    flat.round(4).to_csv(out / "summary.csv", index=False)
    print(flat.round(3).to_string(index=False))
