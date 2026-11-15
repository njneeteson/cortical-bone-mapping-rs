"""Can cortical bone mapping automate the manual correction of a Buie
dual-threshold segmentation?

For realistic bone phantoms (cbm.BonePhantom) with exact ground truth, compare
three pipelines that all start from the image alone:

  buie        Buie et al. (2007) dual threshold (paper parameters, 450/320).
  buie+cbm    CBM (regularised) fitted on Buie's periosteal surface; the
              cortex is the region between the fitted periosteal and
              endosteal surfaces.
  buie+refine CBM segmentation refinement started from Buie's periosteal
              surface, then the same cortex definition.

Metrics (interior slices only, away from the stack ends):
  * Dice of the periosteal region and of the cortex;
  * mean absolute distance of the periosteal and endosteal boundaries from
    the true ones (in-plane, per slice, both directions);
  * Ct.Th as cortical area / periosteal perimeter averaged over slices
    (the semi-derived definition used in the paper's patient analysis),
    compared with the same quantity on the true masks;
  * for the CBM pipelines also the per-node thickness error against the
    exact thickness along each sampling line.

    python docs/scripts/buie_experiment.py      # writes docs/buie_experiment.json and images
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm

DOCS = Path(__file__).resolve().parents[1]
H = 0.0607

SCENARIOS = {
    "healthy": dict(thickness_mean=0.9, bvtv=0.2, endo_bvtv=0.4, porosity=0.03, endo_porosity=0.15),
    "osteoporotic": dict(thickness_mean=0.45, thickness_min=0.1, bvtv=0.12, endo_bvtv=0.35,
                         porosity=0.06, endo_porosity=0.35, rho_bone=800.0),
    "thick": dict(thickness_mean=1.5, bvtv=0.28, endo_bvtv=0.5, porosity=0.02, endo_porosity=0.1),
}
CBM = dict(spacing=H, outside=1.0, inside=3.0, normal_constraint="in_plane", method="regularized",
           reg_neighbours=500, reg_sigma=1.924)
MARGIN = 3  # slices excluded at each end


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def boundary(m):
    """In-plane boundary voxels of each slice."""
    er = ndimage.binary_erosion(m, structure=np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], bool))
    return m & ~er


def surface_distance(a, b):
    """Mean symmetric in-plane boundary distance (mm), per slice."""
    out = []
    for sa, sb in zip(a, b):
        ba, bb = boundary(sa[None])[0], boundary(sb[None])[0]
        if not ba.any() or not bb.any():
            continue
        da = ndimage.distance_transform_edt(~bb) * H
        db = ndimage.distance_transform_edt(~ba) * H
        out.append(0.5 * (da[ba].mean() + db[bb].mean()))
    return float(np.mean(out))


def ct_th(peri, cort):
    """Semi-derived Ct.Th: mean cortical area / periosteal perimeter."""
    vals = []
    for sp, sc in zip(peri, cort):
        per = boundary(sp[None])[0].sum() * H * np.pi / 4  # digital perimeter correction
        if per > 0:
            vals.append(sc.sum() * H * H / per)
    return float(np.mean(vals))


def evaluate(name, peri, cort, truth_peri, truth_cort, truth_endo):
    s = slice(MARGIN, -MARGIN)
    p, c = peri[s], cort[s]
    endo = p & ~c
    return dict(
        pipeline=name,
        dice_periosteal=dice(p, truth_peri[s]),
        dice_cortex=dice(c, truth_cort[s]),
        periosteal_distance_mm=surface_distance(p, truth_peri[s]),
        endosteal_distance_mm=surface_distance(endo, truth_endo[s]),
        ct_th_mm=ct_th(p, c),
        ct_th_true_mm=ct_th(truth_peri[s], truth_cort[s]),
    )


def node_errors(ph, res):
    t_true = ph.true_thickness(res)
    ok = np.isfinite(t_true)
    e = res.thickness[ok] - t_true[ok]
    return dict(node_bias_mm=float(e.mean()), node_rmse_mm=float(np.sqrt((e**2).mean())))


def main():
    rows, examples = [], {}
    for scen, kw in SCENARIOS.items():
        ph = cbm.BonePhantom(dims=(40, 200, 200), noise=80.0, seed=7, **kw)
        img, tp, te = ph.generate()
        tc = tp & ~te
        t0 = time.perf_counter()
        bp, bc, bt = cbm.buie_segmentation(img)
        t_buie = time.perf_counter() - t0
        r = evaluate("buie", bp, bc, tp, tc, te)
        r.update(scenario=scen, time_s=t_buie)
        rows.append(r)

        rho_c = kw.get("rho_bone", 1000.0)
        t0 = time.perf_counter()
        fit = cbm.cortical_bone_mapping(img, bp, rho_c=rho_c, **CBM)
        p, e, c = fit.masks(img.shape)
        r = evaluate("buie+cbm", p, c, tp, tc, te)
        r.update(scenario=scen, time_s=time.perf_counter() - t0, **node_errors(ph, fit))
        rows.append(r)

        t0 = time.perf_counter()
        rr = cbm.refine_segmentation(img, bp, rho_c=rho_c, **CBM)
        r = evaluate("buie+refine", rr.periosteal_mask, rr.cortical_mask, tp, tc, te)
        r.update(scenario=scen, time_s=time.perf_counter() - t0, iterations=len(rr.history),
                 **node_errors(ph, rr.result))
        rows.append(r)
        examples[scen] = (img, tp, te, bp, bt, rr.periosteal_mask, rr.endosteal_mask)
        for row in rows[-3:]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (DOCS / "buie_experiment.json").write_text(json.dumps(rows, indent=2))
    figure(examples)


def figure(examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(examples), figsize=(5 * len(examples), 9.6))
    for col, (scen, (img, tp, te, bp, bt, rp, re)) in enumerate(examples.items()):
        k = img.shape[0] // 2
        for row, (title, peri, endo, colr) in enumerate([
            ("Buie (2007)", bp, bt, "tab:red"),
            ("Buie → CBM refinement", rp, re, "tab:cyan"),
        ]):
            ax = axes[row, col]
            ax.imshow(img[k], cmap="gray", vmin=-150, vmax=1150)
            ax.contour(tp[k], [0.5], colors="lime", linestyles="--", linewidths=0.9)
            ax.contour(te[k], [0.5], colors="lime", linestyles="--", linewidths=0.9)
            ax.contour(peri[k], [0.5], colors=colr, linewidths=1.0)
            ax.contour(endo[k], [0.5], colors=colr, linewidths=1.0)
            ax.set_title(f"{scen}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Truth (green dashed) vs segmentation (solid); 60.7 µm voxels, noise σ = 80")
    out = DOCS / "images" / "buie_vs_cbm.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
