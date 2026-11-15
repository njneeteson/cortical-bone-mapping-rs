"""Can cortical bone mapping automate the manual correction of a Buie
dual-threshold segmentation?

Every pipeline starts from the image alone. Buie et al. (2007) gives the
periosteal mask (paper parameters, 450/320); CBM is then fitted on its
surface with the parameters of Table 1 of Neeteson & Boyd (2025)
(``Options.preset``), except for the sampling distances, which are scaled to
the phantoms (1 mm outside, 3 mm inside instead of 5 / 15 mm). CBM is *not*
told the cortical density: ``rho_c="estimate"`` (see docs/method.md).

Pipelines:
  buie                 Buie (2007).
  buie (no prepro.)    Buie without the largest-bone pre-processing (as
                       Bonelab's autocontour), neighbour scenarios only.
  local, multigrid, regularized, reg+curvature, curvature
                       CBM on Buie's periosteal surface; the cortex is the
                       region between the fitted surfaces.
  refine               CBM segmentation refinement (reg+curvature) from
                       Buie's periosteal surface.

Scenarios (realistic phantoms with exact ground truth, 60.7 um voxels,
noise sigma = 80):
  healthy, osteoporotic, thick          long-bone shafts (cbm.BonePhantom);
  neighbour 1.0 / 0.3 / 0 mm            a fibula / ulna at that gap;
  breaks                                cortical breaks 0.3-2.4 mm wide;
  knee, knee narrow                     tibia / femur contact region
                                        (cbm.JointPhantom), joint space
                                        1.0 and 0.3 mm, normals along the axis.

Metrics:
  * along each sampling line of the fit (for Buie: the lines of the
    regularized fit): errors of the periosteal and endosteal edge positions
    and of the thickness against the exact crossings of the true surfaces
    (for Buie and refinement the edges are read from the masks);
  * shafts also: Dice of the cortex, mean symmetric 3D boundary distances,
    and semi-derived Ct.Th (cortical area / periosteal perimeter per slice);
  * near breaks: mean distance of the periosteal boundary from the true
    (bridging) periosteal surface; neighbour scenarios: fraction of the
    neighbouring bone inside the periosteal mask.

A separate table compares cortical-density choices for the regularized fit:
the true tissue density (oracle), Table 1's fixed 1000, the estimate, and the
per-node profile maximum.

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
MARGIN = 3  # slices excluded at each end of the shafts
NOISE = 80.0

HEALTHY = dict(thickness_mean=0.9, bvtv=0.2, endo_bvtv=0.4, porosity=0.03, endo_porosity=0.15, rho_bone=1050.0)
SHAFTS = {
    "healthy": HEALTHY,
    "osteoporotic": dict(thickness_mean=0.45, thickness_min=0.1, bvtv=0.12, endo_bvtv=0.35, porosity=0.06,
                         endo_porosity=0.35, rho_bone=850.0),
    "thick": dict(thickness_mean=1.5, bvtv=0.28, endo_bvtv=0.5, porosity=0.02, endo_porosity=0.1, rho_bone=1100.0),
    "neighbour 1.0 mm": dict(HEALTHY, dims=(40, 240, 240), neighbour=(1.5, 0.5, 1.0, np.pi / 4)),
    "neighbour 0.3 mm": dict(HEALTHY, dims=(40, 240, 240), neighbour=(1.5, 0.5, 0.3, np.pi / 4)),
    "neighbour 0 mm": dict(HEALTHY, dims=(40, 240, 240), neighbour=(1.5, 0.5, 0.0, np.pi / 4)),
    "breaks": dict(HEALTHY, breaks=[(0.3, 0.3, 1.2, 1.2), (1.9, 0.8, 1.2, 1.2), (3.5, 1.6, 1.2, 1.2),
                                    (5.0, 2.4, 1.2, 1.2)]),
}
KNEES = {
    "knee": dict(joint_space=1.0, rho_bone=950.0),
    "knee narrow": dict(joint_space=0.3, rho_bone=950.0),
}
METHODS = {
    "local": dict(method="local"),
    "multigrid": dict(method="multigrid"),
    "regularized": dict(method="regularized"),
    "reg+curvature": dict(method="regularized", curvature="change", curvature_weight=1.0),
    "curvature": dict(method="regularized", reg_lambda=0.0, curvature="change", curvature_weight=1.0),
}
SAMPLING = dict(outside=1.0, inside=3.0)


def options(site):
    return cbm.Options.preset(site, H, rho_c="estimate", **SAMPLING)


# ---------------------------------------------------------------- metrics

def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def boundary(m):
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=1)


def surface_distance(a, b, region=None):
    """Mean symmetric 3D boundary distance (mm), boundaries restricted to the
    interior slices (and to ``region`` if given)."""
    ba, bb = boundary(a), boundary(b)
    for m in (ba, bb):
        m[:MARGIN] = False
        m[-MARGIN:] = False
        if region is not None:
            m &= region
    if not ba.any() or not bb.any():
        return float("nan")
    da = ndimage.distance_transform_edt(~boundary(b), sampling=H)
    db = ndimage.distance_transform_edt(~boundary(a), sampling=H)
    return float(0.5 * (da[ba].mean() + db[bb].mean()))


def one_sided_distance(a, b, region):
    """Mean distance of a's boundary voxels within ``region`` from b's boundary."""
    ba = boundary(a) & region
    if not ba.any():
        return float("nan")
    return float(ndimage.distance_transform_edt(~boundary(b), sampling=H)[ba].mean())


def ct_th(peri, cort):
    vals = []
    for sp, sc in zip(peri[MARGIN:-MARGIN], cort[MARGIN:-MARGIN]):
        er = ndimage.binary_erosion(sp, structure=ndimage.generate_binary_structure(2, 1))
        per = (sp & ~er).sum() * H * np.pi / 4
        if per > 0:
            vals.append(sc.sum() * H * H / per)
    return float(np.mean(vals))


def mask_edges(points, normals, peri, marrow, x0, x1):
    """First crossings into ``peri`` and then into ``marrow`` along lines,
    from trilinear interpolation of the masks (crossing at 0.5)."""
    xs = np.arange(x0, x1 + 1e-9, 0.25 * H)
    edges = np.full((len(points), 2), np.nan)
    for start in range(0, len(points), 4096):
        p, n = points[start:start + 4096], normals[start:start + 4096]
        q = (p[:, None, :] + xs[None, :, None] * n[:, None, :]) / H
        coords = q.reshape(-1, 3).T
        vp = ndimage.map_coordinates(peri.astype(np.float32), coords, order=1).reshape(len(p), -1) > 0.5
        vm = ndimage.map_coordinates(marrow.astype(np.float32), coords, order=1).reshape(len(p), -1) > 0.5
        for i in range(len(p)):
            ip = np.flatnonzero(vp[i])
            if len(ip) == 0:
                continue
            k = ip[0]
            edges[start + i, 0] = xs[k]
            im = np.flatnonzero(vm[i, k:])
            if len(im):
                edges[start + i, 1] = xs[k + im[0]]
    return edges


def line_metrics(edges, truth, keep):
    ok = keep & np.isfinite(truth).all(1) & np.isfinite(edges).all(1)
    if not ok.any():
        return {}
    ep = edges[ok, 0] - truth[ok, 0]
    ee = edges[ok, 1] - truth[ok, 1]
    et = (edges[ok, 1] - edges[ok, 0]) - (truth[ok, 1] - truth[ok, 0])
    return dict(peri_edge_mae_mm=float(np.abs(ep).mean()), endo_edge_mae_mm=float(np.abs(ee).mean()),
                thickness_bias_mm=float(et.mean()), thickness_rmse_mm=float(np.sqrt((et ** 2).mean())),
                lines=int(ok.sum()))


def fit_edges(res):
    m, t = res.center, res.thickness
    return np.column_stack([m - t / 2, m + t / 2])


# ---------------------------------------------------------------- scenarios

class Shaft:
    site = "radius_tibia"

    def __init__(self, name, kw):
        kw = dict(kw)
        dims = kw.pop("dims", (40, 200, 200))
        self.name = name
        self.ph = cbm.BonePhantom(dims=dims, noise=NOISE, seed=7, **kw)
        self.img, self.tp, self.te = self.ph.generate()
        self.labels = self.ph.labels()
        self.tc = self.tp & ~self.te & (self.labels != 2)
        self.rho_true = kw["rho_bone"]
        self.break_region = None
        if (self.labels == 2).any():
            self.break_region = ndimage.binary_dilation(
                self.labels == 2, structure=np.ones((1, 9, 9), bool))

    def buie(self, **kw):
        return cbm.buie_segmentation(self.img, **kw)

    def truth(self, points, normals, o):
        e = self.ph.edges_along(points, normals, -o.outside, o.inside)
        keep = np.ones(len(points), bool)
        if self.break_region is not None:
            # lines through (or next to) a break have no true cortex
            idx = np.round(points / H).astype(int)
            idx = np.clip(idx, 0, np.array(self.img.shape) - 1)
            keep = ~self.break_region[tuple(idx.T)]
        # interior slices only
        keep &= (points[:, 0] > MARGIN * H) & (points[:, 0] < (self.img.shape[0] - 1 - MARGIN) * H)
        return e, keep

    def masks_metrics(self, peri, cort):
        s = slice(MARGIN, -MARGIN)
        endo = peri & ~cort
        out = dict(dice_cortex=dice(cort[s], self.tc[s]),
                   periosteal_distance_mm=surface_distance(peri, self.tp),
                   endosteal_distance_mm=surface_distance(endo, self.te),
                   ct_th_mm=ct_th(peri, cort), ct_th_true_mm=ct_th(self.tp, self.tc))
        if self.break_region is not None:
            out["break_periosteal_distance_mm"] = one_sided_distance(peri, self.tp, self.break_region)
        if (self.labels == 1).any():
            out["neighbour_included"] = float(peri[self.labels == 1].mean())
        return out


class Knee(Shaft):
    site = "knee"

    def __init__(self, name, kw):
        self.name = name
        self.ph = cbm.JointPhantom(dims=(140, 200, 200), joint_level=0.5, noise=NOISE, seed=5, **kw)
        self.img, tp, te = self.ph.generate()
        self.tp, self.te = tp > 0, te > 0
        self.tc = self.tp & ~self.te
        self.labels = np.zeros(self.img.shape, np.uint8)
        self.rho_true = kw["rho_bone"]
        self.break_region = None

    def buie(self, **kw):
        # both bones are of interest
        return cbm.buie_segmentation(self.img, largest_bone_only=False, **kw)

    def truth(self, points, normals, o):
        e = self.ph.edges_along(points, normals, -o.outside, o.inside)
        keep = (points[:, 0] > MARGIN * H) & (points[:, 0] < (self.img.shape[0] - 1 - MARGIN) * H)
        return e[:, :2], keep

    def masks_metrics(self, peri, cort):
        return {}  # side shells are not fitted (normals along the axis): line metrics only


# ---------------------------------------------------------------- experiment

def run_scenario(sc, rows, examples):
    o = options(sc.site)
    t0 = time.perf_counter()
    bp, bc, bt = sc.buie()
    t_buie = time.perf_counter() - t0

    fits = {}
    for name, kw in METHODS.items():
        t0 = time.perf_counter()
        fits[name] = (cbm.cortical_bone_mapping(sc.img, bp, options=o, **kw), time.perf_counter() - t0)
    ref = fits["regularized"][0]

    def add(pipeline, peri, cort, edges, points, normals, seconds, extra=None):
        truth, keep = sc.truth(points, normals, o)
        r = dict(scenario=sc.name, pipeline=pipeline, time_s=seconds)
        r.update(sc.masks_metrics(peri, cort))
        r.update(line_metrics(edges, truth, keep))
        r.update(extra or {})
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    add("buie", bp, bc, mask_edges(ref.points, ref.normals, bp, bt, -o.outside, o.inside),
        ref.points, ref.normals, t_buie)
    if isinstance(sc, Shaft) and sc.ph.__class__ is cbm.BonePhantom and sc.ph.neighbour is not None:
        bp2, bc2, bt2 = sc.buie(largest_bone_only=False)
        add("buie (no prepro.)", bp2, bc2, mask_edges(ref.points, ref.normals, bp2, bt2, -o.outside, o.inside),
            ref.points, ref.normals, t_buie)
    for name, (res, sec) in fits.items():
        p, e, c = res.masks(sc.img.shape)
        add(name, p, c, fit_edges(res), res.points, res.normals, sec, dict(rho_c=res.rho_c))

    t0 = time.perf_counter()
    rr = cbm.refine_segmentation(sc.img, bp, options=o, **METHODS["reg+curvature"])
    sec = time.perf_counter() - t0
    add("refine", rr.periosteal_mask, rr.cortical_mask, fit_edges(rr.result), rr.result.points,
        rr.result.normals, sec, dict(iterations=len(rr.history), rho_c=rr.result.rho_c))
    examples[sc.name] = (sc, bp, bt, rr)
    return ref, bp


def rho_c_table(sc, bp, rows):
    o = options(sc.site)
    for label, rc in (("oracle", sc.rho_true), ("fixed 1000", 1000.0), ("estimate", "estimate"),
                      ("profile max", "profile_max")):
        res = cbm.cortical_bone_mapping(sc.img, bp, options=o, method="regularized", rho_c=rc)
        truth, keep = sc.truth(res.points, res.normals, o)
        r = dict(scenario=sc.name, rho_c_mode=label, rho_c=float(res.rho_c), rho_true=sc.rho_true)
        r.update(line_metrics(fit_edges(res), truth, keep))
        rows.append(r)
        print(r, flush=True)


def main():
    rows, rho_rows, examples = [], [], {}
    scenarios = [(Shaft, n, kw) for n, kw in SHAFTS.items()] + [(Knee, n, kw) for n, kw in KNEES.items()]
    for cls, name, kw in scenarios:
        print("==", name, flush=True)
        sc = cls(name, kw)
        _, bp = run_scenario(sc, rows, examples)
        if name in ("healthy", "osteoporotic", "thick", "knee"):
            rho_c_table(sc, bp, rho_rows)
        if name not in ("healthy", "osteoporotic", "neighbour 0.3 mm", "breaks", "knee narrow"):
            examples.pop(name, None)
    (DOCS / "buie_experiment.json").write_text(json.dumps(dict(pipelines=rows, rho_c=rho_rows), indent=2))
    figure(examples)


def figure(examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(examples)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8.6))
    for col, name in enumerate(names):
        sc, bp, bt, rr = examples[name]
        knee = isinstance(sc, Knee)
        if knee:
            # coronal section through the contact point
            j = int(round(sc.img.shape[1] / 2 + sc.ph.contact_offset[0] / H))
            sl = lambda a: a[:, j, :]  # noqa: E731
        else:
            k = sc.img.shape[0] // 2
            sl = lambda a: a[k]  # noqa: E731
        for row, (title, peri, marrow, colr) in enumerate([
            ("Buie (2007)", bp, bt, "tab:red"),
            ("Buie → CBM refinement", rr.periosteal_mask, rr.endosteal_mask, "tab:cyan"),
        ]):
            ax = axes[row, col]
            ax.imshow(sl(sc.img), cmap="gray", vmin=-150, vmax=1250, origin="lower" if knee else "upper")
            ax.contour(sl(sc.tp), [0.5], colors="lime", linestyles="--", linewidths=0.9)
            ax.contour(sl(sc.te), [0.5], colors="lime", linestyles="--", linewidths=0.9)
            ax.contour(sl(peri), [0.5], colors=colr, linewidths=1.0)
            if not (knee and row == 1):
                ax.contour(sl(marrow), [0.5], colors=colr, linewidths=1.0)
            ax.set_title(f"{name}: {title}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Truth (green dashed) vs segmentation (solid); 60.7 µm voxels, noise σ = 80. "
                 "Knee: coronal section; the refined endosteal mask is only defined on the articular surfaces.",
                 fontsize=10)
    out = DOCS / "images" / "buie_vs_cbm.png"
    fig.savefig(out, dpi=85, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
