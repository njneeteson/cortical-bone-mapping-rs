"""Automatic periosteal and endosteal segmentation with cortical bone mapping.

Can CBM replace the manual correction of a threshold-based segmentation?
Every pipeline starts from the image alone, on realistic phantoms with exact
ground truth (60.7 um voxels, noise sigma = 80):

  healthy, osteoporotic, thick      long-bone shafts (cbm.BonePhantom)
  neighbour 2.0 / 1.0 / 0.5 mm      an ulna-like bone at that gap (soft tissue between)

The shafts are distal-radius sized (periosteal radius 8.5 mm, ~230 mm^2) with
HR-pQCT (XtremeCT II) values: see docs/segmentation.md#phantoms.

Pipelines:

  buie                    Buie et al. (2007), paper parameters, 450/320, as
                          the reference standard-of-care segmentation;
  buie (no prepro.)       without the largest-bone pre-processing, as in
                          Bonelab's autocontour (neighbour scenarios only);
  hull                    the starting segmentation for CBM: threshold 320,
                          median, largest component, per-slice convex hull
                          (cbm.hull_segmentation);
  local, multigrid, regularized, reg+curvature, curvature, volumetric
                          one CBM fit on the hull's surface; the cortex is the
                          region between the fitted surfaces;
  refine, refine+curvature, refine volumetric
                          CBM segmentation refinement from the hull
                          (regularized, regularized + curvature, and the
                          volumetric model).

CBM uses the parameters of Table 1 of Neeteson & Boyd (2025)
(``Options.preset``) except for the sampling distances, which are scaled to
the phantoms (1 mm outside, 3 mm inside instead of 5 / 15 mm), and is not
told the cortical density (``rho_c="estimate"``).

Metrics:

  * along each sampling line of the fit (for Buie and the hull: the lines of
    the regularized fit): mean absolute errors of the periosteal and endosteal
    edge positions, and bias / RMSE of the thickness, against the exact
    crossings of the true surfaces;
  * shafts also: Dice of the cortex, mean symmetric 3D boundary distances of
    the periosteal and endosteal surfaces, and semi-derived Ct.Th (cortical
    area / periosteal perimeter per slice);
  * neighbours: fraction of the neighbouring bone inside the periosteal mask.

A second table compares cortical-density choices for the regularized fit on
the hull: the true tissue density (oracle), Table 1's fixed 1000, the
estimate, and the per-node profile maximum.

    python docs/scripts/segmentation_experiment.py   # docs/results/segmentation_experiment.json + figure
    python docs/scripts/segmentation_experiment.py --refine-only   # re-run the refinements only
    python docs/scripts/segmentation_experiment.py --figure        # redraw the figures only

The figures are docs/images/segmentation.png (Buie, 1D and volumetric
refinement on the middle slice) and docs/images/segmentation_volumetric.png
(thickness maps of the refinements, unrolled around the shaft).
"""

import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm

DOCS = Path(__file__).resolve().parents[1]
H = 0.0607
MARGIN = 3  # slices excluded at each end
NOISE = 80.0
SHAFT_RADIUS = 8.5  # mm: a distal radius (Tt.Ar ~ 230 mm^2)

HEALTHY = dict(thickness_mean=0.9, bvtv=0.2, endo_bvtv=0.4, porosity=0.03, endo_porosity=0.15, rho_bone=1050.0)
SHAFTS = {
    "healthy": HEALTHY,
    "osteoporotic": dict(thickness_mean=0.45, thickness_min=0.2, bvtv=0.12, endo_bvtv=0.35, porosity=0.06,
                         endo_porosity=0.35, rho_bone=850.0),
    "thick": dict(thickness_mean=1.5, bvtv=0.28, endo_bvtv=0.5, porosity=0.02, endo_porosity=0.1, rho_bone=1100.0),
    # an ulna-like neighbour (radius 3.5 mm, cortex 0.8 mm); there is always soft tissue between
    "neighbour 2.0 mm": dict(HEALTHY, dims=(40, 500, 500), neighbour=(3.5, 0.8, 2.0, np.pi / 4)),
    "neighbour 1.0 mm": dict(HEALTHY, dims=(40, 480, 480), neighbour=(3.5, 0.8, 1.0, np.pi / 4)),
    "neighbour 0.5 mm": dict(HEALTHY, dims=(40, 470, 470), neighbour=(3.5, 0.8, 0.5, np.pi / 4)),
}
FITS = {
    "local": dict(method="local"),
    "multigrid": dict(method="multigrid"),
    "regularized": dict(method="regularized"),
    "reg+curvature": dict(method="regularized", curvature="change", curvature_weight=1.0),
    "curvature": dict(method="regularized", reg_lambda=0.0, curvature="change", curvature_weight=1.0),
    "volumetric": dict(method="volumetric"),
}
REFINES = {
    "refine": FITS["regularized"],
    "refine+curvature": FITS["reg+curvature"],
    "refine volumetric": FITS["volumetric"],
}
SAMPLING = dict(outside=1.0, inside=3.0)
FIGURE = ("healthy", "osteoporotic", "neighbour 0.5 mm")
FIGURE_REFINES = ("refine+curvature", "refine volumetric")
# the refinements drawn in the figures, kept so that they can be redrawn without re-running
FIGURE_CACHE = Path(tempfile.gettempdir()) / "cbm_segmentation_figure.pkl"


def options(site):
    return cbm.Options.preset(site, H, rho_c="estimate", **SAMPLING)


def for_site(kw, site):
    """A method's options for a site: on the shafts every normal is in-plane,
    so the volumetric model's axial blur is tied to the in-plane one."""
    if kw.get("method") == "volumetric" and site == "radius_tibia":
        return dict(kw, vol_sigma_z="tied")
    return kw


# ---------------------------------------------------------------- metrics

def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def boundary(m):
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=1)


def surface_distance(a, b):
    """Mean symmetric 3D boundary distance (mm) over the interior slices."""
    ba, bb = boundary(a), boundary(b)
    for m in (ba, bb):
        m[:MARGIN] = False
        m[-MARGIN:] = False
    if not ba.any() or not bb.any():
        return float("nan")
    da = ndimage.distance_transform_edt(~boundary(b), sampling=H)
    db = ndimage.distance_transform_edt(~boundary(a), sampling=H)
    return float(0.5 * (da[ba].mean() + db[bb].mean()))


def ct_th(peri, cort):
    vals = []
    for sp, sc in zip(peri[MARGIN:-MARGIN], cort[MARGIN:-MARGIN]):
        er = ndimage.binary_erosion(sp, structure=ndimage.generate_binary_structure(2, 1))
        per = (sp & ~er).sum() * H * np.pi / 4
        if per > 0:
            vals.append(sc.sum() * H * H / per)
    return float(np.mean(vals))


def mask_edges(points, normals, peri, marrow, x0, x1):
    """First crossings into ``peri`` and then into ``marrow`` along lines
    (trilinear interpolation of the masks, crossing at 0.5)."""
    xs = np.arange(x0, x1 + 1e-9, 0.25 * H)
    edges = np.full((len(points), 2), np.nan)
    for start in range(0, len(points), 4096):
        p, n = points[start:start + 4096], normals[start:start + 4096]
        coords = ((p[:, None, :] + xs[None, :, None] * n[:, None, :]) / H).reshape(-1, 3).T
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
        dims = kw.pop("dims", (40, 360, 360))
        self.name = name
        self.ph = cbm.BonePhantom(dims=dims, radius=SHAFT_RADIUS, noise=NOISE, seed=7, **kw)
        self.img, self.tp, self.te = self.ph.generate()
        self.labels = self.ph.labels()
        self.tc = self.tp & ~self.te
        self.rho_true = kw["rho_bone"]
        self.has_neighbour = self.ph.neighbour is not None

    def buie(self, **kw):
        return cbm.buie_segmentation(self.img, **kw)

    def hull(self):
        return cbm.hull_segmentation(self.img)

    def truth(self, points, normals, o):
        e = self.ph.edges_along(points, normals, -o.outside, o.inside)
        keep = (points[:, 0] > MARGIN * H) & (points[:, 0] < (self.img.shape[0] - 1 - MARGIN) * H)
        return e, keep

    def mask_metrics(self, peri, cort):
        s = slice(MARGIN, -MARGIN)
        out = dict(dice_cortex=dice(cort[s], self.tc[s]),
                   periosteal_distance_mm=surface_distance(peri, self.tp),
                   endosteal_distance_mm=surface_distance(peri & ~cort, self.te),
                   ct_th_mm=ct_th(peri, cort), ct_th_true_mm=ct_th(self.tp, self.tc))
        if self.has_neighbour:
            out["neighbour_included"] = float(peri[self.labels == 1].mean())
        return out


# ---------------------------------------------------------------- experiment

def report(rows, row):
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


def run_scenario(sc, rows):
    o = options(sc.site)
    t0 = time.perf_counter()
    hull = sc.hull()
    t_hull = time.perf_counter() - t0
    t0 = time.perf_counter()
    fits = {name: cbm.cortical_bone_mapping(sc.img, hull, options=o, **kw) for name, kw in [("regularized", FITS["regularized"])]}
    fits["regularized"].info["time"] = time.perf_counter() - t0
    ref = fits["regularized"]

    def add(pipeline, peri, cort, edges, points, normals, seconds, extra=None):
        truth, keep = sc.truth(points, normals, o)
        r = dict(scenario=sc.name, pipeline=pipeline, time_s=seconds)
        r.update(sc.mask_metrics(peri, cort))
        r.update(line_metrics(edges, truth, keep))
        r.update(extra or {})
        report(rows, r)

    def add_buie(label, **kw):
        t0 = time.perf_counter()
        bp, bc, bt = sc.buie(**kw)
        sec = time.perf_counter() - t0
        add(label, bp, bc, mask_edges(ref.points, ref.normals, bp, bt, -o.outside, o.inside),
            ref.points, ref.normals, sec)
        return bp, bt

    buie = add_buie("buie")
    if sc.has_neighbour:
        add_buie("buie (no prepro.)", largest_bone_only=False)
    # the hull has no endosteal surface: its cortex is empty, only the periosteal metrics apply
    add("hull", hull, np.zeros_like(hull), mask_edges(ref.points, ref.normals, hull, hull, -o.outside, o.inside)
        * np.array([1.0, np.nan]), ref.points, ref.normals, t_hull)

    for name, kw in FITS.items():
        t0 = time.perf_counter()
        res = ref if name == "regularized" else cbm.cortical_bone_mapping(sc.img, hull, options=o, **for_site(kw, sc.site))
        sec = ref.info["time"] if name == "regularized" else time.perf_counter() - t0
        p, e, c = res.masks(sc.img.shape)
        add(name, p, c, fit_edges(res), res.points, res.normals, sec,
            dict(rho_c=res.rho_c, budget_exhausted=res.info["budget_exhausted"]))

    refined = {}
    for name, kw in REFINES.items():
        t0 = time.perf_counter()
        rr = cbm.refine_segmentation(sc.img, hull, options=o, **for_site(kw, sc.site))
        sec = time.perf_counter() - t0
        add(name, rr.periosteal_mask, rr.cortical_mask, fit_edges(rr.result), rr.result.points,
            rr.result.normals, sec, dict(iterations=len(rr.history), converged=rr.converged,
                                         rho_c=rr.result.rho_c))
        if name in FIGURE_REFINES:
            refined[name] = rr
    return hull, buie, refined


def rho_c_table(sc, hull, rows):
    o = options(sc.site)
    for label, rc in (("oracle", sc.rho_true), ("fixed 1000", 1000.0), ("estimate", "estimate"),
                      ("profile max", "profile_max")):
        res = cbm.cortical_bone_mapping(sc.img, hull, options=o, method="regularized", rho_c=rc)
        truth, keep = sc.truth(res.points, res.normals, o)
        r = dict(scenario=sc.name, rho_c_mode=label, rho_c=float(res.rho_c), rho_true=sc.rho_true)
        r.update(line_metrics(fit_edges(res), truth, keep))
        report(rows, r)


def refine_rows(sc, rows):
    """Only the refinement pipelines of a scenario (and Buie, for the
    figure); returns (Buie's periosteal and marrow masks, the figures'
    refinements by name)."""
    o = options(sc.site)
    hull = sc.hull()
    refined = {}
    for name, kw in REFINES.items():
        t0 = time.perf_counter()
        rr = cbm.refine_segmentation(sc.img, hull, options=o, **for_site(kw, sc.site))
        sec = time.perf_counter() - t0
        truth, keep = sc.truth(rr.result.points, rr.result.normals, o)
        r = dict(scenario=sc.name, pipeline=name, time_s=sec)
        r.update(sc.mask_metrics(rr.periosteal_mask, rr.cortical_mask))
        r.update(line_metrics(fit_edges(rr.result), truth, keep))
        r.update(dict(iterations=len(rr.history), converged=rr.converged, rho_c=rr.result.rho_c))
        report(rows, r)
        if name in FIGURE_REFINES:
            refined[name] = rr
    bp, _, bt = sc.buie()
    return (bp, bt), refined


def figure_only():
    """Redraw the figures, re-running Buie and the figures' refinements unless
    they are cached from an earlier run."""
    if FIGURE_CACHE.exists():
        figure(pickle.loads(FIGURE_CACHE.read_bytes()))
        return
    examples = {}
    for name in FIGURE:
        print("==", name, flush=True)
        sc = Shaft(name, SHAFTS[name])
        hull, o = sc.hull(), options(sc.site)
        refined = {r: cbm.refine_segmentation(sc.img, hull, options=o, **for_site(REFINES[r], sc.site))
                   for r in FIGURE_REFINES}
        bp, _, bt = sc.buie()
        examples[name] = (sc, (bp, bt), refined)
    figure(examples)


def refine_only():
    """Re-run the refinement pipelines only (after a change to refinement),
    replacing their rows in the saved results, and redraw the figure."""
    path = DOCS / "results" / "segmentation_experiment.json"
    saved = json.loads(path.read_text())
    rows, examples = [], {}
    scenarios = [(Shaft, n, kw) for n, kw in SHAFTS.items()]
    for cls, name, kw in scenarios:
        print("==", name, flush=True)
        sc = cls(name, kw)
        buie, refined = refine_rows(sc, rows)
        if name in FIGURE:
            examples[name] = (sc, buie, refined)
    new = {(r["scenario"], r["pipeline"]): r for r in rows}
    saved["pipelines"] = [new.get((r["scenario"], r["pipeline"]), r) for r in saved["pipelines"]]
    path.write_text(json.dumps(saved, indent=2))
    figure(examples)


def main(only=None):
    rows, rho_rows, examples = [], [], {}
    scenarios = [(Shaft, n, kw) for n, kw in SHAFTS.items()]
    for cls, name, kw in scenarios:
        if only and name not in only:
            continue
        print("==", name, flush=True)
        sc = cls(name, kw)
        hull, buie, refined = run_scenario(sc, rows)
        if name in ("healthy", "osteoporotic", "thick"):
            rho_c_table(sc, hull, rho_rows)
        if name in FIGURE:
            examples[name] = (sc, buie, refined)
    if only:
        return
    (DOCS / "results" / "segmentation_experiment.json").write_text(json.dumps(dict(pipelines=rows, rho_c=rho_rows), indent=2))
    figure(examples)


def figure_data(sc, buie, refined):
    """What the figures draw of one scenario: the middle slice of the image,
    the truth and every segmentation, a zoom on the gap to a neighbour, and
    per node of each refinement its position around the shaft, thickness and
    the true thickness along its line."""
    o = options(sc.site)
    k = sc.img.shape[0] // 2
    bp, bt = buie
    d = dict(img=sc.img[k], tp=sc.tp[k], te=sc.te[k], segs={"Buie (2007)": (bp[k], bt[k])}, maps={}, zoom=None)
    for name, rr in refined.items():
        d["segs"][name] = (rr.periosteal_mask[k], rr.endosteal_mask[k])
        res = rr.result
        truth, keep = sc.truth(res.points, res.normals, o)
        centre = res.points[:, 1:].mean(0)
        angle = np.degrees(np.arctan2(res.points[:, 1] - centre[0], res.points[:, 2] - centre[1])) % 360
        ok = keep & np.isfinite(truth).all(1)
        d["maps"][name] = dict(angle=angle[ok], z=res.points[ok, 0], thickness=res.thickness[ok],
                               truth=truth[ok, 1] - truth[ok, 0])
    if sc.has_neighbour:
        # the neighbour's voxel nearest the periosteum is the middle of the gap
        dist = ndimage.distance_transform_edt(~d["tp"])
        nb = sc.labels[k] == 1
        i, j = np.unravel_index(np.argmin(np.where(nb, dist, np.inf)), nb.shape)
        d["zoom"] = (i, j)
    return d


def figure(examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {n: (v if isinstance(v, dict) else figure_data(*v)) for n, v in examples.items()}
    try:
        FIGURE_CACHE.write_bytes(pickle.dumps(data))
    except OSError:
        pass
    names = list(data)
    rows = [("Buie (2007)", "Buie (2007)", "tab:red"),
            ("refine+curvature", "hull → 1D refinement (curvature)", "tab:orange"),
            ("refine volumetric", "hull → volumetric refinement", "tab:cyan")]
    zoomed = [n for n in names if data[n]["zoom"] is not None]
    cols = [(n, None) for n in names] + [(n, data[n]["zoom"]) for n in zoomed]
    half = 55  # zoom half-width, voxels (3.3 mm)

    # middle slices
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 3.7 * len(rows)))
    for c, (name, zoom) in enumerate(cols):
        d = data[name]
        for r, (key, title, colr) in enumerate(rows):
            ax = axes[r, c]
            peri, marrow = d["segs"][key]
            ax.imshow(d["img"], cmap="gray", vmin=-150, vmax=1250)
            for m in (d["tp"], d["te"]):
                ax.contour(m, [0.5], colors="lime", linestyles="--", linewidths=0.9)
            for m in (peri, marrow):
                ax.contour(m, [0.5], colors=colr, linewidths=1.2 if zoom else 1.0)
            if zoom:
                i, j = zoom
                ax.set_xlim(j - half, j + half)
                ax.set_ylim(i + half, i - half)
                ax.set_title(f"{name}, the gap\n{title}", fontsize=9)
            else:
                ax.set_title(f"{name}\n{title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Truth (green, dashed) and segmentation (solid). 60.7 µm voxels, noise σ = 80, middle slice.",
                 fontsize=10)
    fig.tight_layout()
    out = DOCS / "images" / "segmentation.png"
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # thickness maps, unrolled around the shaft
    from scipy.stats import binned_statistic_2d

    fig, axes = plt.subplots(3, len(names), figsize=(4.4 * len(names), 6.6), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    tmax = max(np.percentile(data[n]["maps"]["refine volumetric"]["truth"], 99) for n in names)
    map_rows = [("volumetric: thickness", "mm", "refine volumetric", "thickness", "viridis", (0, tmax)),
                ("volumetric: error", "µm", "refine volumetric", "error", "RdBu_r", (-250, 250)),
                ("1D + curvature: error", "µm", "refine+curvature", "error", "RdBu_r", (-250, 250))]
    a_bins = np.linspace(0, 360, 181)  # 2°
    for r, (title, unit, key, what, cmap, (lo, hi)) in enumerate(map_rows):
        for c, name in enumerate(names):
            m, ax = data[name]["maps"][key], axes[r, c]
            v = m["thickness"] if what == "thickness" else 1000 * (m["thickness"] - m["truth"])
            z_bins = np.arange(m["z"].min(), m["z"].max() + 2 * H, 2 * H)
            img = binned_statistic_2d(m["z"], m["angle"], v, "mean", bins=[z_bins, a_bins]).statistic
            im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                           extent=(0, 360, z_bins[0], z_bins[-1]), interpolation="nearest")
            ax.set_facecolor("0.85")
            label = (f"{name}\n" if r == 0 else "") + title
            if what == "error":
                label += f", RMSE {np.sqrt(np.mean(v ** 2)):.0f} µm"
            ax.set_title(label, fontsize=9)
            ax.set_xticks([0, 90, 180, 270, 360])
            if c == 0:
                ax.set_ylabel("z (mm)")
            if r == len(map_rows) - 1:
                ax.set_xlabel("angle around the shaft (°)")
        fig.colorbar(im, ax=axes[r, :], shrink=0.95, pad=0.01).set_label(unit)
    fig.suptitle("Refinement from the hull, unrolled around the shaft (2° × 0.12 mm bins). "
                 "Error = fitted − true thickness along each node's line.", fontsize=10)
    out = DOCS / "images" / "segmentation_volumetric.png"
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1:] == ["--refine-only"]:
        sys.exit(refine_only())
    if sys.argv[1:] == ["--figure"]:
        sys.exit(figure_only())
    sys.exit(main(sys.argv[1:] or None))
