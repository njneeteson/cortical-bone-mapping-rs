"""Refinement on the site phantoms, every method, against the exact truth.

For each site phantom (``cbm.site_phantom``: distal radius and tibia at
HR-pQCT resolution, proximal femur at clinical resolution) every method
refines the same automatic start:

* radius, tibia: ``cbm.hull_segmentation`` (threshold 320, largest
  component, per-slice convex hull), which leaves out the ulna / fibula;
* femur: ``cbm.label_bones`` at the Otsu threshold and ``cbm.bone_hulls``
  with a 4 mm closing; the acetabulum's start is excluded while the femur is
  refined.

Writes docs/results/phantoms.json and the docs/images/phantom_* figures.

    python docs/scripts/phantoms.py            # compute and draw
    python docs/scripts/phantoms.py --redraw   # redraw from the last run, if the code is unchanged
"""

import sys
import time

import numpy as np

import cortical_bone_mapping_rs as cbm
from _common import (ENDO, LABELS, METHODS, PERI, RESULTS, START, TRUTH, draw_segments, load_run,
                     plotter, provenance, pv_mesh, save, save_run, screenshot, section_segments, style,
                     write_json)

HR = dict(outside=1.0, inside=2.5, normal_constraint="in_plane", normal_axis=0)
SITES = {
    "distal_radius": dict(title="Distal radius (HR-pQCT, 61 µm)", sampling=HR, k=500, tied=True,
                          spacings=(7.7696, 3.8848, 1.9424, 0.9712)),
    "distal_tibia": dict(title="Distal tibia (HR-pQCT, 61 µm)", sampling=HR, k=500, tied=True,
                         spacings=(7.7696, 3.8848, 1.9424, 0.9712)),
    "proximal_femur": dict(title="Proximal femur (clinical CT, 0.75 × 0.75 × 1 mm)",
                           sampling=dict(outside=2.5, inside=7.0), k=128, tied=False, spacings=(8.0, 4.0, 2.0)),
}
REFINE = dict(refine_iterations=50)
MARGIN = 3  # end slices of the shafts left out of the mask metrics
INPUTS = dict(sites=SITES, refine=REFINE, seed=1)


def method_options(s, method):
    kw = dict(s["sampling"], method=method, rho_c="estimate", progress_every=0, reg_sigma=1.9424,
              reg_neighbours=s["k"])
    if method == "multigrid":
        kw["multigrid_spacings"] = s["spacings"]
    if method == "volumetric" and s["tied"]:
        kw["vol_sigma_z"] = "tied"
    return kw


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def site_data(name):
    s = SITES[name]
    ph = cbm.site_phantom(name, seed=1)
    img, peri_t, endo_t = ph.generate()
    sp = tuple(float(v) for v in ph.spacing)
    femur = name == "proximal_femur"
    if femur:
        tp, te = peri_t == ph.FEMUR, endo_t == ph.FEMUR
        labels, _ = cbm.label_bones(img, "otsu", otsu_floor=-500.0, min_size=int(2000 / np.prod(sp)))
        starts = cbm.bone_hulls(labels, sp, closing_radius=4.0)
        ids = [b for b in np.unique(starts) if b]
        fl = max(ids, key=lambda b: (tp & (starts == b)).sum())
        start = starts == fl
        exclude = (starts > 0) & ~start
        view = ("coronal", 1, img.shape[1] // 2)
    else:
        tp, te = peri_t.astype(bool), endo_t.astype(bool)
        start = cbm.hull_segmentation(img)
        exclude = None
        view = ("axial", 0, img.shape[0] // 2)
    tc = tp & ~te
    keep = slice(None) if femur else slice(MARGIN, -MARGIN)
    out = dict(title=s["title"], spacing=sp, view=view, runs={})
    _, ax_i, k = view
    out["slice"] = dict(img=np.take(img, k, ax_i), tp=np.take(tp, k, ax_i), te=np.take(te, k, ax_i),
                        start=np.take(start, k, ax_i))
    for method in METHODS:
        t0 = time.perf_counter()
        rr = cbm.refine_segmentation(img, start, sp, exclude=exclude, verbose=False, **REFINE,
                                     **method_options(s, method))
        sec = time.perf_counter() - t0
        r = rr.result
        o = r.options
        edges = ph.edges_along(r.points, r.normals, -o.outside, o.inside)
        t_true = edges[:, 1] - edges[:, 0]
        if femur:
            t_true[edges[:, 2] != ph.FEMUR] = np.nan
        x_peri = np.einsum("ij,ij->i", r.periosteal_points - r.points, r.normals)
        err_t, err_p = r.thickness - t_true, x_peri - edges[:, 0]
        ok = np.isfinite(err_t) & np.isfinite(err_p)
        if not femur:
            z = r.points[:, 0] / sp[0]
            ok &= (z > MARGIN) & (z < img.shape[0] - 1 - MARGIN)
        row = dict(moves=len(rr.history) - 1, converged=bool(rr.converged), time_s=sec, nodes=int(r.n),
                   rho_c=float(r.rho_c), thickness_bias_mm=float(err_t[ok].mean()),
                   thickness_rmse_mm=float(np.sqrt(np.mean(err_t[ok] ** 2))),
                   periosteum_rmse_mm=float(np.sqrt(np.mean(err_p[ok] ** 2))),
                   cortex_dice=dice(rr.cortical_mask[keep], tc[keep]))
        print(name, method, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        err_t = np.where(ok, err_t, np.nan)
        out["runs"][method] = dict(
            row=row, peri=(r.surface_vertices, r.surface_faces), endo=(rr.endosteal_vertices, r.surface_faces),
            nodes=(r.points, r.faces), thickness=r.thickness, error=err_t)
    return out


# ------------------------------------------------------------------ figures

def zoom_centre(d, sl, ax_i, k, sp, margin_mm):
    """Where to zoom: the point of the local fit's endosteal section farthest
    from the true endosteum (where the methods differ most)."""
    from scipy import ndimage

    te = sl["te"]
    edge = te & ~ndimage.binary_erosion(te)
    dist = ndimage.distance_transform_edt(~edge)
    seg = section_segments(*d["runs"]["local"]["endo"], ax_i, k, sp).reshape(-1, 2)
    c = np.clip(np.round(seg).astype(int), 0, np.array(te.shape[::-1]) - 1)
    rows = np.nonzero(sl["tp"])[0]
    other = [a for a in range(3) if a != ax_i]
    m = margin_mm / sp[other[0]]
    ok = (c[:, 1] > rows.min() + m) & (c[:, 1] < rows.max() - m)  # not near a cut end
    c = c[ok]
    i = int(np.argmax(dist[c[:, 1], c[:, 0]]))
    return c[i, 1], c[i, 0]


def sections(name, d, plt):
    """One section per method, whole and zoomed: truth, start and refined surfaces."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    _, ax_i, k = d["view"]
    sl = d["slice"]
    sp = d["spacing"]
    other = [a for a in range(3) if a != ax_i]
    rows, cols = np.nonzero(sl["tp"])
    pad = 12
    shape = sl["tp"].shape
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, shape[0] - 1)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, shape[1] - 1)
    aspect = sp[other[0]] / sp[other[1]]
    zoom_mm = 24.0 if name == "proximal_femur" else 4.0  # about 30 and 65 voxels
    half = int(round(zoom_mm / 2 / sp[other[1]]))
    zr, zc = zoom_centre(d, sl, ax_i, k, sp, margin_mm=15.0 if name == "proximal_femur" else 0.5)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.6), layout="constrained")
    for c, method in enumerate(METHODS):
        run = d["runs"][method]
        peri = section_segments(*run["peri"], ax_i, k, sp)
        endo = section_segments(*run["endo"], ax_i, k, sp)
        for r, zoom in enumerate((False, True)):
            ax = axes[r, c]
            ax.imshow(sl["img"], cmap="gray", vmin=-200, vmax=1900 if name == "proximal_femur" else 1300, aspect=aspect)
            lw = 2.2 if zoom else 1.3
            for m in (sl["tp"], sl["te"]):
                ax.contour(m, [0.5], colors=TRUTH, linewidths=lw, linestyles="--")
            ax.contour(sl["start"], [0.5], colors=START, linewidths=lw, linestyles=":")
            draw_segments(ax, peri, PERI, lw=lw)
            draw_segments(ax, endo, ENDO, lw=lw)
            if zoom:
                hr = int(round(half * sp[other[1]] / sp[other[0]]))
                ax.set_xlim(zc - half, zc + half)
                ax.set_ylim(zr + hr, zr - hr)
            else:
                ax.set_xlim(c0, c1)
                ax.set_ylim(r1, r0)
                hr = int(round(half * sp[other[1]] / sp[other[0]]))
                ax.add_patch(Rectangle((zc - half, zr - hr), 2 * half, 2 * hr, fill=False, ec="white", lw=1.5))
                row = run["row"]
                ax.set_title(f"{LABELS[method]}\nCt.Th RMSE {1000 * row['thickness_rmse_mm']:.0f} µm")
            ax.set_xticks([])
            ax.set_yticks([])
    axes[1, 0].set_ylabel(f"zoom (white box), {zoom_mm:g} mm wide", fontsize=12)
    handles = [Line2D([], [], color=TRUTH, ls="--", lw=2.5, label="truth"),
               Line2D([], [], color=START, ls=":", lw=2.5, label="start"),
               Line2D([], [], color=PERI, lw=2.5, label="refined periosteal"),
               Line2D([], [], color=ENDO, lw=2.5, label="fitted endosteal")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, f"phantom_{name}_sections.png")


def unrolled(ax, run, key, cmap, lo, hi, sp):
    """A shaft's node values unrolled: angle around the shaft by height."""
    from scipy.stats import binned_statistic_2d

    pts = run["nodes"][0]
    v = run[key]
    ok = np.isfinite(v)
    cy, cx = pts[:, 1].mean(), pts[:, 2].mean()
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 2] - cx)) % 360
    z = pts[:, 0]
    zb = np.arange(z[ok].min(), z[ok].max() + sp[0], sp[0])
    img = binned_statistic_2d(z[ok], ang[ok], v[ok], "mean", bins=[zb, np.linspace(0, 360, 181)]).statistic
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                   extent=(0, 360, zb[0], zb[-1]))
    ax.set_facecolor("0.85")
    return im


def maps(name, d, plt):
    """Fitted thickness and its error, per method: unrolled for the shafts,
    on the 3D surface for the femur."""
    femur = name == "proximal_femur"
    tmax = float(np.nanpercentile(np.concatenate([d["runs"][m]["thickness"] for m in METHODS]), 98))
    emax = 0.5 if femur else 0.2
    specs = [("thickness", "viridis", 0.0, tmax, "fitted Ct.Th (mm)"),
             ("error", "RdBu_r", -emax, emax, "Ct.Th error,\nfitted − true (mm)")]
    if femur:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8.2), layout="constrained")
    else:
        fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), layout="constrained", sharex=True, sharey=True)
    for r, (key, cmap, lo, hi, label) in enumerate(specs):
        for c, method in enumerate(METHODS):
            run = d["runs"][method]
            ax = axes[r, c]
            if femur:
                p = plotter((700, 760))
                p.add_mesh(pv_mesh(*run["nodes"], run[key], key), scalars=key, cmap=cmap, clim=(lo, hi),
                           nan_color="lightgrey", show_scalar_bar=False, smooth_shading=True, ambient=0.35,
                           diffuse=0.7, specular=0.0)
                p.view_vector((-0.2, -1.0, 0.25), viewup=(0, 0, 1))
                p.camera.zoom(1.3)
                ax.imshow(screenshot(p))
                ax.set_axis_off()
            else:
                unrolled(ax, run, key, cmap, lo, hi, d["spacing"])
                if r == 1:
                    ax.set_xlabel("angle around the shaft (°)")
                    ax.set_xticks([0, 90, 180, 270, 360])
                if c == 0:
                    ax.set_ylabel("height (mm)")
            if r == 0:
                ax.set_title(LABELS[method])
        from matplotlib import cm, colors

        sm = cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap)
        fig.colorbar(sm, ax=axes[r, :], shrink=0.9, pad=0.01).set_label(label)
    save(fig, f"phantom_{name}_maps.png")


def draw(data):
    plt = style()
    for name, d in data.items():
        sections(name, d, plt)
        maps(name, d, plt)


def main():
    redraw = "--redraw" in sys.argv
    saved = load_run("phantoms", __file__, INPUTS) if redraw else None
    if saved is None:
        prov = provenance(__file__, INPUTS)
        data = {name: site_data(name) for name in SITES}
        write_json(RESULTS / "phantoms.json", prov,
                   {name: {m: d["runs"][m]["row"] for m in METHODS} for name, d in data.items()})
        save_run("phantoms", prov, data)
    else:
        data = saved["data"]
    draw(data)


if __name__ == "__main__":
    main()
