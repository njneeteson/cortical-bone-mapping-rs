"""Run time and memory of every method, and comparisons with the original
Python implementation (Bonelab) and with SciPy.

* scaling: ``cargo run --release --example benchmark`` fits every method on
  cylinder phantoms at HR-pQCT resolution (60.7 µm voxels, outer radius
  8 mm, cortex 1.2 ± 0.5 mm, PSF 0.1 mm, noise 60) with 8, 32 and 128 slices,
  timing the whole pipeline and counting the peak heap;
* Bonelab: ``compare_bonelab.py``, the same profiles fitted by Bonelab's
  classes (8 and 32 slices);
* SciPy: ``compare_scipy.py``, the same problems solved with
  ``scipy.optimize`` (8 and 32 slices).

Run on an otherwise idle machine. Writes docs/results/benchmarks.json and the
docs/images/benchmark_* figures.

    python docs/scripts/benchmarks.py /path/to/Bonelab
    python docs/scripts/benchmarks.py /path/to/Bonelab --redraw   # redraw from the last run, if the code is unchanged
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import compare_bonelab
import compare_scipy
from _common import RESULTS, ROOT, load_run, provenance, save, save_run, style, write_json

SCALING = ("local", "multigrid", "regularized k=64", "regularized k=500", "reg+curvature", "volumetric")
NAMES = {"local": "local", "multigrid": "multigrid", "regularized k=64": "regularised, 64 neighbours",
         "regularized k=500": "regularised, 500 neighbours", "reg+curvature": "regularised + curvature, 64",
         "volumetric": "volumetric"}
INPUTS = dict(scaling_slices=(8, 32, 128), reference_slices=(8, 32))


def rust_scaling():
    out = Path(tempfile.gettempdir()) / "cbm_benchmark.json"
    subprocess.run(["cargo", "run", "--release", "--example", "benchmark", "-p", "cortical-bone-mapping", "--",
                    "--json", str(out)], cwd=ROOT, check=True)
    return json.loads(out.read_text())


# ------------------------------------------------------------------ figures

def fig_scaling(rows, plt):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(wspace=0.28))
    for m in SCALING:
        r = sorted((x for x in rows if x["method"] == m), key=lambda x: x["nodes"])
        n = [x["nodes"] for x in r]
        axes[0].plot(n, [x["time"] for x in r], "o-", lw=2, ms=7, label=NAMES[m])
        axes[1].plot(n, [x["peak_bytes"] / 1e6 for x in r], "o-", lw=2, ms=7, label=NAMES[m])
    for ax, label in zip(axes, ("run time (s)", "peak memory (MB)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("surface nodes")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "benchmark_scaling.png")


def fig_reference(bonelab, scipy_rows, plt):
    """Fit time of the same problems: Bonelab, SciPy and this crate (32 slices)."""
    n = max(r["nodes"] for r in bonelab)
    methods = ["local", "multigrid", "regularized k=64"]
    bl = {r["method"]: r for r in bonelab if r["nodes"] == n}
    sc = {r["method"]: r for r in scipy_rows if r["nodes"] == n}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(methods))
    w = 0.26
    bars = [("Bonelab (Python)", [bl[m]["bonelab_time"] if m in bl else np.nan for m in methods], "#9e9e9e"),
            ("SciPy", [sc[m]["scipy_time"] if m in sc else np.nan for m in methods], "#f2a93b"),
            ("cortical-bone-mapping-rs", [bl[m]["rust_fit_time"] if m in bl else np.nan for m in methods], "#2f7ed8")]
    for i, (label, vals, colour) in enumerate(bars):
        b = ax.bar(x + (i - 1) * w, vals, w, label=label, color=colour)
        for rect, v in zip(b, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v * 1.15, f"{v:.3g} s", ha="center", va="bottom",
                        fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(x, ["local", "multigrid", "regularised, 64 neighbours"])
    ax.set_ylabel("fit time (s)")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"The same {n} profiles fitted by each implementation")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    save(fig, "benchmark_reference.png")


def draw(d):
    plt = style()
    fig_scaling(d["scaling"], plt)
    fig_reference(d["bonelab"], d["scipy"], plt)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    commit = subprocess.run(["git", "-C", args[0], "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    inputs = dict(INPUTS, bonelab_commit=commit)
    saved = load_run("benchmarks", __file__, inputs) if "--redraw" in sys.argv else None
    if saved is None:
        prov = provenance(__file__, inputs)
        d = dict(scaling=rust_scaling(), bonelab=compare_bonelab.main(args[0]), scipy=compare_scipy.main())
        write_json(RESULTS / "benchmarks.json", prov, d)
        save_run("benchmarks", prov, d)
    else:
        d = saved["data"]
    draw(d)


if __name__ == "__main__":
    main()
