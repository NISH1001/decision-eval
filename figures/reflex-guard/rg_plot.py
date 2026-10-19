"""Draw reflex-guard's overview figures (scripts/plot_eval.py, unchanged) with d1-ultra and Laya typed added to STYLE.
Okabe-Ito: the five existing models use green, vermillion, blue, reddish purple and sky blue; the new two take the
remaining orange and yellow, each with its own hatch and marker."""
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2]); sys.path[:0] = [str(repo / "scripts"), str(repo)]
import eval as ev
import plot_eval as pe

pe.STYLE["laya-typed"] = ("Laya typed", "#F0E442", "|||", "p", "laptop GPU")
pe.STYLE["d1-ultra"] = ("d1-ultra", "#E69F00", "+++", "*", "API")
keep = ["laya", "laya-typed", "gliner:fp32:orders=3", "gliner:int8:orders=3", "jev", "openai", "d1-ultra"]
for name in ("core", "ood"):
    results, rows, test, categories = ev.compute_set(name)
    results = sorted((r for r in results if r["model"] in keep), key=lambda r: keep.index(r["model"]))
    pe.plot(results, rows, test, categories, out=out / name, name=name)
