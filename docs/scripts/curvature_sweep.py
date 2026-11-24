"""Sweep of the curvature penalties for segmentation refinement.

The refinement of docs/scripts/segmentation_experiment.py (hull start,
Table 1 parameters with 1 / 3 mm sampling, estimated rho_c, HR-pQCT voxels
and noise) on small shaft phantoms where the penalty matters most: healthy,
a neighbour 0.5 mm away, and four cortical breaks. The shafts are smaller
than the experiment's (periosteal radius 5 mm, 24 slices, a 2.5 mm
neighbour) so that a run takes seconds; the comparison is between penalties,
not an estimate of the absolute errors.

For the regularised and the volumetric refinement it varies the scale of the
curvature measurement (the Gaussian surface Laplacian), and for each scale:

  change      curvature added relative to the starting surface (weight 1);
  absolute    curvature above kappa (0.3 and 1 / mm), weights 0.1 and 1;
  variation   how fast the curvature changes, weights 0.1 and 1;

and records the metrics of the segmentation experiment, the number of moves
and convergence.

    python docs/scripts/curvature_sweep.py OUT.json SCENARIO [SCENARIO ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation_experiment as se  # noqa: E402

import cortical_bone_mapping_rs as cbm  # noqa: E402

se.SHAFT_RADIUS = 5.0
SLICES = 24
SCENARIOS = {
    "healthy": dict(se.HEALTHY, dims=(SLICES, 220, 220)),
    "neighbour 0.5 mm": dict(se.HEALTHY, dims=(SLICES, 290, 290), neighbour=(2.5, 0.6, 0.5, np.pi / 4)),
    # the four break sizes of the experiment, centred in the shorter slab
    "breaks": dict(se.HEALTHY, dims=(SLICES, 220, 220),
                   breaks=[(0.3, 0.3, 0.73, 0.3), (1.9, 0.5, 0.73, 0.5), (3.5, 0.8, 0.73, 0.8),
                           (5.0, 1.2, 0.73, 1.2)]),
}
SCALES = (0.25, 0.5, 1.0)
REFINE = dict(refine_iterations=15, verbose=False, progress_every=0)


def penalty(method, mode, weight, kappa=None):
    if method == "regularized":
        kw = dict(method="regularized", curvature=mode, curvature_weight=weight)
    else:
        kw = dict(method="volumetric", vol_curvature=mode, vol_curvature_weight=weight)
    if kappa is not None:
        kw["curvature_kappa"] = kappa
    return kw


def configurations():
    for method in ("regularized", "volumetric"):
        for scale in SCALES:
            yield method, scale, "change, weight 1", penalty(method, "change", 1.0)
            for kappa in (0.3, 1.0):
                for w in (0.1, 1.0):
                    yield method, scale, f"absolute, kappa {kappa:g}, weight {w:g}", penalty(method, "absolute", w,
                                                                                            kappa)
            for w in (0.1, 1.0):
                yield method, scale, f"variation, weight {w:g}", penalty(method, "variation", w)


def run(sc, method, scale, label, kw, rows):
    o = se.options(sc.site)
    t0 = time.perf_counter()
    rr = cbm.refine_segmentation(sc.img, sc.hull_mask, options=o, curvature_scale=scale, **REFINE,
                                 **se.for_site(kw, sc.site))
    row = dict(scenario=sc.name, method=method, scale=scale, penalty=label, time_s=time.perf_counter() - t0,
               iterations=len(rr.history), converged=rr.converged, stalled=rr.stalled)
    row.update(sc.mask_metrics(rr.periosteal_mask, rr.cortical_mask))
    truth, keep = sc.truth(rr.result.points, rr.result.normals, o)
    row.update(se.line_metrics(se.fit_edges(rr.result), truth, keep))
    se.report(rows, row)


def main():
    path = Path(sys.argv[1])
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["scenario"], r["method"], r["scale"], r["penalty"]) for r in rows}
    for name in sys.argv[2:]:
        sc = se.Shaft(name, SCENARIOS[name])
        sc.hull_mask = sc.hull()
        print("==", name, flush=True)
        for method, scale, label, kw in configurations():
            if (name, method, scale, label) in done:
                continue
            run(sc, method, scale, label, kw, rows)
            path.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
