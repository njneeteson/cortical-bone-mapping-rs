"""Refinement experiments on the femoral intercondylar notch of the clinical
knee CT (docs/refinement_experiments.md).

The image is not part of the repository. Run docs/scripts/ct_example.py with
``--cache knee.pkl`` first; this script crops the notch region from that cache
and refines the femur's starting hull there with each variant:

    python docs/scripts/notch_experiment.py knee.pkl --out results/ run [VARIANT ...]
    python docs/scripts/notch_experiment.py knee.pkl --out results/ figures

``run`` writes one ``<variant>.pkl`` and ``<variant>.json`` per variant into
``--out``; ``figures`` draws docs/images/notch_*.png from them. Everything
farther than 10 mm outside the femur's start is blanked in the figures (display only:
the analysis uses the whole crop, soft tissue included).

Measures (no ground truth on real data):

* converged / stalled, moves, wall time and its split;
* notch fill: volume inside the refined surface farther than 2 mm from the
  thresholded femur (marrow filled), and thresholded bone left outside it;
* convex agreement: on nodes where the starting hull lies within 1 mm of the
  thresholded surface (the hull is already right there), the distance from
  the refined surface to the thresholded surface;
* edge agreement: fitted periosteal edge minus the steepest rise of each
  profile (a model-free edge), over profiles with a clear rise.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images"
# the notch region of the right-knee crop (image indices z, y, x)
CROP = (slice(120, 250), slice(170, 312), slice(45, 200))
FEMUR = 1

BASE = dict(
    outside=4.0, inside=8.0, resolution=0.25,
    taubin_iterations=50, taubin_passband=0.1,
    method="regularized", reg_lambda=1.0, reg_sigma=3.0, reg_neighbours=500,
    rho_c="estimate",
    init=(0.0, 2.0, 0.5, 0.0, 100.0),
    lower=(-4.0, 0.05, 0.05, -300.0, -300.0),
    upper=(8.0, 8.0, 2.0, 400.0, 1200.0),
    progress_every=0,
)
BASE_REFINE = dict(refine_iterations=150, relaxation=0.5, keep_history=True)
CLOSING = 4.0
THRESHOLD = {"value": 267.0}  # set from the cache (the Otsu threshold of the knee)  # mm: closes gaps up to ~8 mm, keeps the ~20 mm notch open
FIELDS = dict(global_params=())

# name: (description, Options overrides, RefineOptions overrides[, start])
# start: "hull" (per-slice convex hull, default) or a closing radius in mm
VARIANTS = {
    "closing_freeze": ("4 mm closing, level set, adaptive, density fields, regional convergence",
                       FIELDS, dict(level_set=True, freeze=True), CLOSING),
    "hull_freeze": ("convex hull, level set, adaptive, density fields, regional convergence",
                    FIELDS, dict(level_set=True, freeze=True), "hull"),
    "closing": ("4 mm closing, level set, adaptive, density fields (no freezing)",
                FIELDS, dict(level_set=True), CLOSING),
    "closing_freeze_global_sigma": ("as closing_freeze, global sigma",
                                    dict(global_params=("sigma",)), dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_inside_15": ("as closing_freeze, 15 mm inside profiles",
                                 dict(global_params=(), inside=15.0, upper=(15.0, 8.0, 2.0, 400.0, 1200.0)),
                                 dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_neighbours_250": ("as closing_freeze, 250 regularisation neighbours",
                                      dict(global_params=(), reg_neighbours=250),
                                      dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_lambda_0.1": ("as closing_freeze, lambda 0.1",
                                  dict(global_params=(), reg_lambda=0.1),
                                  dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_volumetric": ("as closing_freeze, the volumetric model as the inner fit",
                                  dict(method="volumetric", vol_scanner_axis=0),
                                  dict(level_set=True, freeze=True), CLOSING),
    # curvature penalties on the fitted surfaces, measured at 1 mm (the nodes are ~0.65 mm apart);
    # the volumetric ones capped at 15 moves (not converged by then = not solved)
    "closing_freeze_change": ("as closing_freeze, change-of-curvature penalty (weight 1)",
                              dict(FIELDS, curvature="change", curvature_weight=1.0, curvature_scale=1.0),
                              dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_absolute_0.3": ("as closing_freeze, absolute curvature above 0.3 / mm (weight 0.3)",
                                    dict(FIELDS, curvature="absolute", curvature_kappa=0.3, curvature_weight=0.3,
                                         curvature_scale=1.0),
                                    dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_absolute_1": ("as closing_freeze, absolute curvature above 1 / mm (weight 0.3)",
                                  dict(FIELDS, curvature="absolute", curvature_kappa=1.0, curvature_weight=0.3,
                                       curvature_scale=1.0),
                                  dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_variation_0.1": ("as closing_freeze, curvature-variation penalty (weight 0.1)",
                                     dict(FIELDS, curvature="variation", curvature_weight=0.1, curvature_scale=1.0),
                                     dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_variation_1": ("as closing_freeze, curvature-variation penalty (weight 1)",
                                   dict(FIELDS, curvature="variation", curvature_weight=1.0, curvature_scale=1.0),
                                   dict(level_set=True, freeze=True), CLOSING),
    "closing_freeze_volumetric_spacing": ("as closing_freeze_volumetric_change, curvature at 4 node spacings",
                                          dict(method="volumetric", vol_scanner_axis=0, vol_curvature="change",
                                               vol_curvature_weight=0.3),
                                          dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    "closing_freeze_volumetric_adaptive": ("as closing_freeze_volumetric_spacing, adaptive band (up to 2x the reach)",
                                           dict(method="volumetric", vol_scanner_axis=0, vol_curvature="change",
                                                vol_curvature_weight=0.3, vol_adaptive_band=2.0),
                                           dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    "closing_freeze_volumetric_adaptive_soft_tissue": ("as closing_freeze_volumetric_adaptive, global soft tissue",
                                                       dict(method="volumetric", vol_scanner_axis=0,
                                                            vol_curvature="change", vol_curvature_weight=0.3,
                                                            vol_adaptive_band=2.0, vol_global_soft_tissue=True),
                                                       dict(level_set=True, freeze=True, refine_iterations=15),
                                                       CLOSING),
    "closing_freeze_volumetric_soft_tissue": ("as closing_freeze_volumetric_change, one global soft-tissue density",
                                              dict(method="volumetric", vol_scanner_axis=0, vol_curvature="change",
                                                   vol_curvature_weight=0.3, curvature_scale=1.0,
                                                   vol_global_soft_tissue=True),
                                              dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    "closing_freeze_volumetric_change": ("as closing_freeze_volumetric, change-of-curvature penalty (weight 0.3)",
                                         dict(method="volumetric", vol_scanner_axis=0, vol_curvature="change",
                                              vol_curvature_weight=0.3, curvature_scale=1.0),
                                         dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    "closing_freeze_volumetric_absolute_0.3": ("as closing_freeze_volumetric, absolute curvature above 0.3 / mm",
                                               dict(method="volumetric", vol_scanner_axis=0, curvature_kappa=0.3,
                                                    vol_curvature_weight=0.3, curvature_scale=1.0),
                                               dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    "closing_freeze_volumetric_variation": ("as closing_freeze_volumetric, curvature-variation penalty (weight 0.1)",
                                            dict(method="volumetric", vol_scanner_axis=0, vol_curvature="variation",
                                                 vol_curvature_weight=0.1, curvature_scale=1.0),
                                            dict(level_set=True, freeze=True, refine_iterations=15), CLOSING),
    # the earlier stages, from the convex hull, with the 20 moves they were run with
    "same_nodes": ("convex hull, same nodes moved every iteration, global densities, full fits",
                   {}, dict(refine_iterations=20, edge_ratio=0.0, stall_iterations=0,
                            region_tolerance=float("inf")), "hull"),
    "level_set": ("convex hull, level set, global densities, full fits",
                  {}, dict(refine_iterations=20, level_set=True, edge_ratio=0.0, stall_iterations=0,
                           region_tolerance=float("inf")), "hull"),
    "level_set_adaptive": ("convex hull, level set, global densities, adaptive inner fits",
                           {}, dict(refine_iterations=20, level_set=True, region_tolerance=float("inf")), "hull"),
    "density_fields": ("convex hull, level set, adaptive, density fields",
                       FIELDS, dict(refine_iterations=20, level_set=True, region_tolerance=float("inf")), "hull"),
}


def load_crop(cache):
    d = pickle.load(open(cache, "rb"))
    THRESHOLD["value"] = float(d["threshold"])
    img = np.ascontiguousarray(d["img"][CROP])
    hull = np.ascontiguousarray((d["hulls"][CROP] == FEMUR).astype(np.uint8))
    lab = np.ascontiguousarray(d["labels"][CROP])
    return img, hull, lab, tuple(float(s) for s in d["spacing"])


def closing_start(cache, radius, out):
    """The femur's closing start, computed on the whole knee (so that the
    marrow is enclosed in every slice) and then cropped; cached in ``out``."""
    f = Path(out) / f"closing_{radius:g}.npy"
    if f.exists():
        return np.load(f)
    d = pickle.load(open(cache, "rb"))
    c = cbm.bone_hulls(d["labels"], d["spacing"], closing_radius=radius)
    start = np.ascontiguousarray((c[CROP] == FEMUR).astype(np.uint8))
    np.save(f, start)
    return start


def filled_bone(lab):
    """Thresholded femur with its marrow filled (per axial slice, then in 3D);
    the notch, open to the back, is not filled."""
    b = np.stack([ndimage.binary_fill_holes(s) for s in (lab == FEMUR)])
    return ndimage.binary_fill_holes(b)


def steepest_rise(x, prof):
    g = np.gradient(ndimage.gaussian_filter1d(prof.astype(float), 1.0, axis=1), x, axis=1)
    i = np.argmax(g, axis=1)
    return x[i], g[np.arange(len(i)), i]


def bridged_region(cache, hull, sp, out):
    """The concavities the convex hull bridges: inside the hull, more than
    1 mm outside the 4 mm closing (essentially the notch)."""
    c = closing_start(cache, CLOSING, out).astype(bool)
    return hull.astype(bool) & (ndimage.distance_transform_edt(~c, sampling=sp) > 1.0)


def measures(rr, img, hull, bone, sp, options, bridged):
    sp = np.array(sp)
    v = float(np.prod(sp))
    in_d = ndimage.distance_transform_edt(bone, sampling=sp)
    out_d = ndimage.distance_transform_edt(~bone, sampling=sp)
    m = {
        "bridged_cm3": round(float(bridged.sum() * v) / 1000, 2),
        "notch_still_inside_cm3": round(float((rr.periosteal_mask & bridged).sum() * v) / 1000, 2),
        "start_notch_inside_cm3": round(float((hull.astype(bool) & bridged).sum() * v) / 1000, 2),
        "bone_outside_cm3": round(float((~rr.periosteal_mask & (in_d > 2.0)).sum() * v) / 1000, 3),
    }
    # convex agreement: signed distance (+ = outside the thresholded bone)
    signed = np.where(bone, -in_d, out_d)
    hull_sd = np.where(hull.astype(bool), 0, 1)
    hull_surf_dist = ndimage.distance_transform_edt(hull_sd, sampling=sp) + ndimage.distance_transform_edt(1 - hull_sd, sampling=sp)
    verts = rr.result.surface_vertices
    q = np.clip(np.round(verts / sp).astype(int), 0, np.array(img.shape) - 1)
    # nodes whose nearest hull surface lies within 1 mm of the threshold surface
    near_hull = hull_surf_dist[q[:, 0], q[:, 1], q[:, 2]] < 1.5
    hull_ok = np.abs(signed[q[:, 0], q[:, 1], q[:, 2]]) < 2.0
    sel = near_hull & hull_ok
    sd = ndimage.map_coordinates(signed, (verts[sel] / sp).T, order=1)
    m["convex_nodes"] = int(sel.sum())
    m["convex_signed_dist_median_mm"] = round(float(np.median(sd)), 3)
    m["convex_abs_dist_median_mm"] = round(float(np.median(np.abs(sd))), 3)
    # edge agreement on the final fit (resampled with profiles kept)
    o2 = cbm.Options(**{**options, "keep_profiles": True})
    r = rr.result
    line = np.arange(-o2.outside, o2.inside + 1e-9, o2.resolution)
    pts = r.points[:, None, :] + line[None, :, None] * r.normals[:, None, :]
    prof = ndimage.map_coordinates(img.astype(np.float64), (pts.reshape(-1, 3) / sp).T, order=1).reshape(len(r.points), -1)
    ge, slope = steepest_rise(line, prof)
    edge = r.center - r.thickness / 2
    ok = slope > 150.0
    diff = edge[ok] - ge[ok]
    # threshold surface along each convex node's normal: the 267 HU crossing
    # nearest the node, linearly interpolated (+ = the node lies inside the
    # threshold surface, i.e. the crossing is outward of it)
    thr = THRESHOLD["value"]
    qn = np.clip(np.round(r.points / sp).astype(int), 0, np.array(img.shape) - 1)
    hull_d = ndimage.distance_transform_edt(hull.astype(bool), sampling=sp) + \
        ndimage.distance_transform_edt(~hull.astype(bool), sampling=sp)
    convex = (hull_d[qn[:, 0], qn[:, 1], qn[:, 2]] < 1.5) & ok
    above = prof >= thr
    cross = np.full(len(prof), np.nan)
    i0 = int(np.argmin(np.abs(line)))
    for k in np.nonzero(convex)[0]:
        a = above[k]
        idx = np.nonzero(a[1:] != a[:-1])[0]  # crossing between idx and idx+1
        if len(idx) == 0:
            continue
        j = idx[np.argmin(np.abs(idx + 0.5 - i0))]
        y0, y1 = prof[k, j], prof[k, j + 1]
        cross[k] = line[j] + (thr - y0) / (y1 - y0) * (line[j + 1] - line[j])
    c = cross[~np.isnan(cross)]
    m["convex_threshold_nodes"] = int(len(c))
    m["convex_node_to_threshold_mm"] = round(float(np.median(-c)), 3)
    m["edge_vs_rise_median_mm"] = round(float(np.median(diff)), 3)
    m["edge_vs_rise_abs_median_mm"] = round(float(np.median(np.abs(diff))), 3)
    m["edge_vs_rise_over_0.5mm"] = round(float(np.mean(np.abs(diff) > 0.5)), 3)
    m["sigma_median_mm"] = round(float(np.median(r.sigma)), 3)
    m["sigma_at_bound"] = round(float(np.mean(r.sigma > 0.999 * o2.upper[2])), 3)
    m["rho_b_p5_p50_p95"] = [round(float(np.percentile(r.rho_b, p))) for p in (5, 50, 95)]
    m["thickness_median_mm"] = round(float(np.median(r.thickness)), 3)
    return m


def run(args):
    img, hull, lab, sp = load_crop(args.cache)
    bone = filled_bone(lab)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bridged = bridged_region(args.cache, hull, sp, out)
    for name in args.variants or list(VARIANTS):
        desc, oo, ro, *st = VARIANTS[name]
        start_kind = st[0] if st else "hull"
        start = hull if start_kind == "hull" else closing_start(args.cache, start_kind, out)
        options = {**BASE, **oo}
        refine = cbm.RefineOptions(**{**BASE_REFINE, **ro, "label": name})
        t = time.time()
        rr = cbm.refine_segmentation(img, start, sp, options=cbm.Options(**options), refine=refine)
        seconds = time.time() - t
        s = dict(name=name, description=desc, start=start_kind, options=oo, refine=ro, seconds=round(seconds, 1),
                 converged=rr.converged, stalled=rr.stalled, message=rr.message,
                 moves=len(rr.surfaces) - 1, nodes=int(rr.result.n),
                 timings={k: round(v, 1) for k, v in rr.timings},
                 history=rr.history)
        s.update(measures(rr, img, start, bone, sp, options, bridged))
        pickle.dump(rr, open(out / f"{name}.pkl", "wb"))
        json.dump(s, open(out / f"{name}.json", "w"), indent=1, default=float)
        print(json.dumps({k: v for k, v in s.items() if k not in ("history", "timings")}, default=float), flush=True)


def remeasure(args):
    """Recompute the measures of saved results (no refitting)."""
    img, hull, lab, sp = load_crop(args.cache)
    bone = filled_bone(lab)
    out = Path(args.out)
    bridged = bridged_region(args.cache, hull, sp, out)
    for name in args.variants or list(VARIANTS):
        if not (out / f"{name}.json").exists():
            continue
        s = json.load(open(out / f"{name}.json"))
        rr = pickle.load(open(out / f"{name}.pkl", "rb"))
        desc, oo, ro, *st = VARIANTS[name]
        start_kind = st[0] if st else "hull"
        start = hull if start_kind == "hull" else closing_start(args.cache, start_kind, out)
        for k in ("notch_fill_excess_cm3", "hull_excess_cm3"):
            s.pop(k, None)
        s.update(measures(rr, img, start, bone, sp, {**BASE, **oo}, bridged))
        json.dump(s, open(out / f"{name}.json", "w"), indent=1, default=float)
        print(name, s["notch_still_inside_cm3"], flush=True)


CUTS = [(0, 40, "axial"), (0, 60, "axial"), (2, 70, "sagittal"), (1, 110, "coronal"), (2, 40, "sagittal")]


def figures(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    img, hull, lab, sp = load_crop(args.cache)
    # blank everything farther than 10 mm outside the hull (display only; real data)
    keep = ndimage.distance_transform_edt(hull == 0, sampling=sp) <= 10.0
    shown = np.where(keep, img, -1000)
    out = Path(args.out)
    names = [n for n in (args.variants or VARIANTS) if (out / f"{n}.pkl").exists()]
    fig, axes = plt.subplots(len(names), len(CUTS), figsize=(3.2 * len(CUTS), 3.3 * len(names)), squeeze=False)
    for row, n in zip(axes, names):
        rr = pickle.load(open(out / f"{n}.pkl", "rb"))
        s = json.load(open(out / f"{n}.json"))
        r = rr.result
        st = s.get("start", "hull")
        start = hull if st == "hull" else closing_start(args.cache, float(st), out)
        for ax, (axis, k, kind) in zip(row, CUTS):
            ax.imshow(np.take(shown, k, axis=axis), cmap="gray", vmin=-200, vmax=1400, origin="lower")
            ax.contour(np.take(start, k, axis=axis), [0.5], colors="y", linewidths=0.7, linestyles="--")
            for v, c, lw in ((r.surface_vertices, "r", 0.9), (r.displaced_surface("endosteal"), "c", 0.6)):
                seg = cbm.mesh_section(v, r.surface_faces, axis, k, sp)[:, :, ::-1]
                if len(seg):
                    ax.add_collection(LineCollection(seg, colors=c, linewidths=lw))
            ax.set_xticks([]), ax.set_yticks([])
            if n == names[0]:
                ax.set_title(f"{kind} {k}", fontsize=9)
        status = "converged" if s["converged"] else ("stalled" if s["stalled"] else "not converged")
        row[0].set_ylabel(f"{n}\n{status}, {s['moves']} moves, {s['seconds']:.0f} s", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / args.figure, dpi=70)
    plt.close(fig)
    print(IMAGES / args.figure)


MODELS = {
    "global densities": dict(global_params=("rho_s", "rho_b")),
    "density fields": dict(global_params=()),
    "density fields, global sigma": dict(global_params=("sigma",)),
}
# three nodes on the anterior surface where global densities misplace the edge
PROFILE_POINTS = [(40, 43, 70), (40, 46, 80), (40, 42, 90)]


def profiles(args):
    """One fit (no refinement) on the thresholded femur surface per model:
    fitted edge vs the steepest rise over the outer surface, and three
    profiles with their fitted models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, hull, lab, sp = load_crop(args.cache)
    bone = filled_bone(lab).astype(np.uint8)
    depth = ndimage.distance_transform_edt(hull, sampling=sp)
    fits, table = {}, {}
    for name, oo in MODELS.items():
        o = cbm.Options(**{**BASE, **oo, "keep_profiles": True})
        t = time.time()
        r = cbm.cortical_bone_mapping(img, bone, sp, options=o)
        seconds = time.time() - t
        ge, slope = steepest_rise(r.profile_x, r.profiles)
        edge = r.center - r.thickness / 2
        q = np.clip(np.round(r.points / np.array(sp)).astype(int), 0, np.array(img.shape) - 1)
        outer = (depth[q[:, 0], q[:, 1], q[:, 2]] < 1.5) & (slope > 150.0)
        diff = (edge - ge)[outer]
        table[name] = dict(
            seconds=round(seconds, 1), outer_nodes=int(outer.sum()), rho_c=round(float(r.rho_c)),
            edge_vs_rise_median_mm=round(float(np.median(diff)), 2),
            edge_vs_rise_over_0_5mm=round(float(np.mean(np.abs(diff) > 0.5)), 2),
            edge_inside_node_median_mm=round(float(np.median(edge[outer])), 2),
            sigma_median_mm=round(float(np.median(r.sigma[outer])), 2),
            sigma_at_bound=round(float(np.mean(r.sigma > 1.99)), 3),
            thickness_median_mm=round(float(np.median(r.thickness[outer])), 2),
            rho_b_p5_p50_p95=[round(float(np.percentile(r.rho_b, p))) for p in (5, 50, 95)],
        )
        print(name, json.dumps(table[name]), flush=True)
        fits[name] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(out / "profiles.json", "w"), indent=1)
    fig, axes = plt.subplots(1, len(PROFILE_POINTS), figsize=(5 * len(PROFILE_POINTS), 3.8))
    r0 = fits["global densities"]
    for ax, p in zip(axes, PROFILE_POINTS):
        target = np.array(p) * np.array(sp)
        for (name, r), style in zip(fits.items(), ("r-", "b--", "g:")):
            i = int(np.argmin(np.linalg.norm(r.points - target, axis=1)))
            if r is r0:
                ax.plot(r.profile_x, r.profiles[i], "k.-", ms=3, label="image")
            mp = np.array([r.center[i], r.thickness[i], r.sigma[i], r.rho_s[i], r.rho_b[i]])
            ax.plot(r.profile_x, cbm.profile_model(r.profile_x, mp, r.rho_c), style,
                    label=f"{name}: edge {mp[0] - mp[1] / 2:+.2f}, σ {mp[2]:.2f}, ρb {mp[4]:.0f}")
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlabel("position along the inward normal (mm)")
        ax.set_ylabel("HU")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(IMAGES / "notch_profiles.png", dpi=80)
    print(IMAGES / "notch_profiles.png")


def table(args):
    """Print a Markdown table of the variants' results (for the write-up)."""
    out = Path(args.out)
    names = [n for n in (args.variants or VARIANTS) if (out / f"{n}.json").exists()]
    print("| variant | start | result | moves | time | notch inside the surface (cm³) | bone outside (cm³) "
          "| convex: node depth below the threshold surface (mm) | edge − steepest rise (mm) | > 0.5 mm | σ (mm) | t (mm) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in names:
        s = json.load(open(out / f"{n}.json"))
        status = "converged" if s["converged"] else ("stalled" if s["stalled"] else "not converged")
        start = "hull" if s.get("start", "hull") == "hull" else f"closing {float(s['start']):g} mm"
        print(f"| `{n}` | {start} | {status} | {s['moves']} | {s['seconds']:.0f} s "
              f"| {s['notch_still_inside_cm3']:.1f} of {s['bridged_cm3']:.1f} | {s['bone_outside_cm3']:.2f} "
              f"| {s['convex_node_to_threshold_mm']:+.2f} | {s['edge_vs_rise_median_mm']:+.2f} "
              f"| {100 * s['edge_vs_rise_over_0.5mm']:.0f}% | {s['sigma_median_mm']:.2f} | {s['thickness_median_mm']:.2f} |")
    p = out / "profiles.json"
    if p.exists():
        print()
        print("| model | edge − steepest rise (mm) | > 0.5 mm | edge inside the node (mm) | σ (mm) | σ at bound | t (mm) | ρb 5–50–95% (HU) | time |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---:|")
        for n, s in json.load(open(p)).items():
            rb = "–".join(str(v) for v in s["rho_b_p5_p50_p95"])
            print(f"| {n} | {s['edge_vs_rise_median_mm']:+.2f} | {100 * s['edge_vs_rise_over_0_5mm']:.0f}% "
                  f"| {s['edge_inside_node_median_mm']:.2f} | {s['sigma_median_mm']:.2f} "
                  f"| {100 * s['sigma_at_bound']:.0f}% | {s['thickness_median_mm']:.2f} | {rb} | {s['seconds']:.0f} s |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="cache pickle written by ct_example.py --cache")
    ap.add_argument("--out", required=True, help="directory for the results (not the repository)")
    ap.add_argument("command", choices=("run", "figures", "profiles", "table", "remeasure"))
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--figure", default="notch_variants.png", help="file name in docs/images (figures)")
    a = ap.parse_args()
    {"run": run, "figures": figures, "profiles": profiles, "table": table, "remeasure": remeasure}[a.command](a)


if __name__ == "__main__":
    main()
