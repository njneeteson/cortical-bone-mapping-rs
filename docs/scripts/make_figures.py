"""Regenerate the figures in docs/images.

    python docs/scripts/make_figures.py [names...]

Names: profile, methods, noise, curvature, refine, benchmark (default: all).
Needs matplotlib (and pillow for the GIF).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import cortical_bone_mapping_rs as cbm  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]
IMG = DOCS / "images"
IMG.mkdir(exist_ok=True)
CYL = dict(outside=1.5, inside=2.5, normal_constraint="in_plane")
METHODS = {
    "local": dict(method="local"),
    "multigrid": dict(method="multigrid", multigrid_spacings=(4.0, 2.0, 1.0)),
    "regularized": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0),
    "curvature": dict(method="regularized", reg_lambda=0.0, curvature="change"),
    # a shaft: every normal is in-plane, so the axial blur is tied
    "volumetric": dict(method="volumetric", vol_sigma_z="tied", reg_neighbours=64, reg_sigma=1.0),
}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10})


def save(fig, name):
    fig.savefig(IMG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", IMG / name)


def theta_z(ph, pts):
    _, th = ph.polar(pts)
    return np.degrees(th), pts[:, 0]


# --------------------------------------------------------------------------
def fig_profile():
    ph = cbm.Phantom(noise=30.0, dims=(24, 96, 96))
    img, mask = ph.generate()
    r = cbm.cortical_bone_mapping(img, mask, ph.spacing, keep_profiles=True, **CYL)
    th = np.degrees(ph.polar(r.points)[1])
    i = int(np.argmin(np.abs(th - 30) + 10 * np.abs(r.points[:, 0] - r.points[:, 0].mean())))
    x = r.profile_x
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    k = int(round(r.points[i, 0] / ph.spacing[0]))
    a.imshow(img[k], cmap="gray", extent=[0, img.shape[2] * 0.1, img.shape[1] * 0.1, 0])
    a.contour(np.arange(img.shape[2]) * 0.1, np.arange(img.shape[1]) * 0.1, mask[k], [0.5], colors="tab:red", linewidths=0.8)
    p, n = r.points[i], r.normals[i]
    seg = np.array([p + x[0] * n, p + x[-1] * n])
    a.plot(seg[:, 2], seg[:, 1], color="tab:orange", lw=1.5)
    peri, endo = r.periosteal_points[i], r.endosteal_points[i]
    a.plot([peri[2], endo[2]], [peri[1], endo[1]], color="tab:cyan", lw=3)
    a.set_title("slice, mask (red), profile line, fitted cortex (cyan)")
    a.set_xlabel("axis 2 [mm]")
    a.set_ylabel("axis 1 [mm]")
    b.plot(x, r.profiles[i], ".", ms=3, color="0.4", label="sampled")
    xf = np.linspace(x[0], x[-1], 400)
    b.plot(xf, r.model(i, xf), color="tab:blue", label="fitted model")
    m, t, s, rs, rb = r.params[i]
    for e, lab in [(m - t / 2, "periosteal"), (m + t / 2, "endosteal")]:
        b.axvline(e, ls="--", color="tab:cyan", lw=1)
        b.text(e, b.get_ylim()[1] * 0.95, lab, rotation=90, va="top", ha="right", fontsize=7)
    b.axvline(0, color="tab:red", lw=0.8)
    true_t = ph.true_thickness(r.points[i:i + 1])[0]
    b.set_title(f"t = {t:.3f} mm (true {true_t:.3f}), σ = {s:.3f} mm, ρs = {rs:.0f}, ρb = {rb:.0f}")
    b.set_xlabel("distance along inward normal x [mm]")
    b.set_ylabel("density")
    b.legend(loc="center right", fontsize=8)
    save(fig, "profile_fit.png")


# --------------------------------------------------------------------------
def fig_methods():
    ph = cbm.Phantom(noise=120.0, seed=3, dims=(40, 96, 96))
    img, mask = ph.generate()
    res = {k: cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL, **v) for k, v in METHODS.items()}
    ref = res["local"]
    fig, axes = plt.subplots(2, len(res) + 1, figsize=(2.6 * (len(res) + 1), 5.2), gridspec_kw=dict(height_ratios=[1, 0.8]))
    th, z = theta_z(ph, ref.points)
    kw = dict(s=2, cmap="viridis", vmin=0.4, vmax=1.2, rasterized=True)
    axes[0, 0].scatter(th, z, c=ph.true_thickness(ref.points), **kw)
    axes[0, 0].set_title("truth")
    for ax, (name, r) in zip(axes[0, 1:], res.items()):
        th, z = theta_z(ph, r.points)
        sc = ax.scatter(th, z, c=r.thickness, **kw)
        e = r.thickness - ph.true_thickness(r.points)
        ax.set_title(f"{name}\nRMSE {np.sqrt((e**2).mean())*1000:.0f} µm, {r.info['timings'].get(name + ' fit', sum(r.info['timings'].values())):.2f} s"
                     if name != "curvature" else f"{name} (λ=0)\nRMSE {np.sqrt((e**2).mean())*1000:.0f} µm")
    for ax in axes[0]:
        ax.set_xlabel("θ [°]")
    axes[0, 0].set_ylabel("axial position [mm]")
    fig.colorbar(sc, ax=axes[0].tolist(), label="thickness [mm]", shrink=0.8)
    # profiles of thickness around the circumference at the middle slice
    mid = ref.points[:, 0].mean()
    for ax, name in zip(axes[1, 1:], res):
        r = res[name]
        sel = np.abs(r.points[:, 0] - mid) < 0.15
        th = np.degrees(ph.polar(r.points[sel])[1])
        o = np.argsort(th)
        ax.plot(th[o], r.thickness[sel][o], ".", ms=2, color="tab:blue")
        tt = np.linspace(-180, 180, 361)
        ax.plot(tt, ph.thickness(np.radians(tt)), color="k", lw=1)
        ax.set_ylim(0.3, 1.3)
        ax.set_xlabel("θ [°]")
    axes[1, 1].set_ylabel("thickness [mm]")
    e = {k: r.thickness - ph.true_thickness(r.points) for k, r in res.items()}
    print("methods, noise 120: RMSE [um]", {k: round(float(np.sqrt((v ** 2).mean())) * 1000, 1) for k, v in e.items()})
    axes[1, 0].hist(list(e.values()), bins=40, range=(-0.2, 0.2), histtype="step", label=list(e))
    axes[1, 0].set_xlabel("thickness error [mm]")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Cylinder phantom, noise σ = 120 (cortex 1000, soft tissue 50, marrow 250), PSF 0.12 mm, 0.1 mm voxels")
    save(fig, "methods_comparison.png")


# --------------------------------------------------------------------------
def fig_noise():
    noises = [10, 30, 60, 120, 200]
    out = {k: [] for k in METHODS}
    for nz in noises:
        ph = cbm.Phantom(noise=float(nz), seed=11)
        img, mask = ph.generate()
        for k, v in METHODS.items():
            r = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL, **v)
            e = r.thickness - ph.true_thickness(r.points)
            out[k].append((e.mean(), np.sqrt((e**2).mean())))
    for k, v in out.items():
        print(f"noise {noises}: {k:12} bias [um] {[round(b * 1000, 1) for b, _ in v]} RMSE [um] "
              f"{[round(r * 1000, 1) for _, r in v]}")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for k, v in out.items():
        v = np.array(v)
        a.plot(noises, v[:, 1] * 1000, "o-", label=k)
        b.plot(noises, v[:, 0] * 1000, "o-", label=k)
    a.set_xlabel("noise standard deviation")
    a.set_ylabel("thickness RMSE [µm]")
    b.set_xlabel("noise standard deviation")
    b.set_ylabel("thickness bias [µm]")
    b.axhline(0, color="k", lw=0.6)
    a.legend()
    fig.suptitle("Accuracy vs noise (voxel 100 µm, mean thickness 800 µm)")
    save(fig, "accuracy_vs_noise.png")


# --------------------------------------------------------------------------
def fig_curvature():
    ph = cbm.Phantom(noise=150.0, seed=4)
    img, mask = ph.generate()
    loc = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL)
    e0 = np.sqrt(((loc.thickness - ph.true_thickness(loc.points)) ** 2).mean())
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ws = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for scale in [0.15, 0.25, 0.5, 1.0]:
        rm = []
        for wgt in ws:
            r = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL, method="regularized", reg_lambda=0.0,
                                          curvature="change", curvature_weight=wgt, curvature_scale=scale)
            rm.append(np.sqrt(((r.thickness - ph.true_thickness(r.points)) ** 2).mean()))
        ax.plot(ws, np.array(rm) * 1000, "o-", label=f"scale {scale} mm")
        print(f"curvature, scale {scale} mm, weights {ws}: RMSE [um] {[round(v * 1000, 1) for v in rm]}")
    lam = [0.1, 0.3, 1, 3, 10, 30]
    rn = []
    for l in lam:
        r = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL, method="regularized", reg_lambda=l,
                                      reg_neighbours=64, reg_sigma=1.0)
        rn.append(np.sqrt(((r.thickness - ph.true_thickness(r.points)) ** 2).mean()))
    ax.plot(lam, np.array(rn) * 1000, "k--", marker="s", label="Neeteson λ (k=64)")
    print(f"smoothing, lambda {lam}: RMSE [um] {[round(v * 1000, 1) for v in rn]}")
    rv = []
    for wgt in ws:
        r = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL, **{**METHODS["volumetric"], "vol_curvature_weight": wgt})
        rv.append(np.sqrt(((r.thickness - ph.true_thickness(r.points)) ** 2).mean()))
    ax.plot(ws, np.array(rv) * 1000, "k:", marker="^", label="volumetric μ (with its λ = 1)")
    print(f"volumetric curvature weight {ws}: RMSE [um] {[round(v * 1000, 1) for v in rv]}; local {e0 * 1000:.1f}")
    ax.axhline(e0 * 1000, color="gray", lw=1, label="local")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("curvature weight μ (or λ)")
    ax.set_ylabel("thickness RMSE [µm]")
    ax.legend(fontsize=7)
    ax.set_title("Curvature penalty ('change' mode, λ = 0), noise σ = 150")
    save(fig, "curvature_sweep.png")


# --------------------------------------------------------------------------
def fig_refine():
    ph = cbm.Phantom(dims=(30, 128, 128), radius=4.0, radius_bump=0.5, radius_lobes=3,
                     thickness_mean=0.8, thickness_amplitude=0.3, mask_offset=0.45,
                     mask_bump=0.35, mask_lobes=7, noise=40.0)
    img, bad = ph.generate()
    truth = ph.true_mask()
    rr = cbm.refine_segmentation(img, bad, ph.spacing, **CYL, method="regularized", reg_neighbours=64,
                                 reg_sigma=1.0, keep_history=True, tolerance=0.002, refine_iterations=12)
    h = 0.1
    k = img.shape[0] // 2
    # with the level set every surface has its own mesh
    masks = [cbm.voxelize(v, f, img.shape, ph.spacing) for v, f in zip(rr.surfaces, rr.surfaces_faces)]
    dices = [2 * (m & (truth > 0)).sum() / (m.sum() + truth.sum()) for m in masks]
    # faces away from the caps, coloured by signed radial error
    zlim = (0, (img.shape[0] - 1) * h)

    def face_data(v, faces):
        tri = v[faces]
        keep = (tri[:, :, 0] > zlim[0]).all(1) & (tri[:, :, 0] < zlim[1]).all(1)
        err = ph.radial_offset(tri[keep].mean(1))
        return tri[keep], err

    fig = plt.figure(figsize=(9.6, 4.4))
    a = fig.add_subplot(1, 2, 1)
    b = fig.add_subplot(1, 2, 2, projection="3d")
    ext = [0, img.shape[2] * h, img.shape[1] * h, 0]
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.6, vmax=0.6)
    yy, xx = np.arange(img.shape[1]) * h, np.arange(img.shape[2]) * h

    def draw(it):
        a.clear()
        b.clear()
        a.imshow(img[k], cmap="gray", extent=ext)
        a.contour(xx, yy, truth[k], [0.5], colors="lime", linewidths=1.0, linestyles="--")
        a.contour(xx, yy, masks[it][k], [0.5], colors="red", linewidths=1.2)
        a.set_title(f"iteration {it}: Dice {dices[it]:.4f}" + ("  (input)" if it == 0 else ""))
        a.set_xlim(1, ext[1] - 1)
        a.set_ylim(ext[2] - 1, 1)
        a.set_xticks([])
        a.set_yticks([])
        a.text(1.3, ext[2] - 1.4, "red: segmentation   green: true periosteum", color="w", fontsize=7)
        tri, err = face_data(rr.surfaces[it], rr.surfaces_faces[it])
        cols = plt.cm.RdBu_r(norm(err))
        pc = Poly3DCollection(tri[:, :, ::-1], facecolors=cols, edgecolor="none")
        b.add_collection3d(pc)
        c = ph.center
        b.set_xlim(c[2] - 5.6, c[2] + 5.6)
        b.set_ylim(c[1] - 5.6, c[1] + 5.6)
        b.set_zlim(zlim[0] - 0.5, zlim[1] + 0.5)
        b.set_box_aspect((1, 1, 0.45), zoom=1.35)
        b.view_init(elev=28, azim=-60 + 4 * it)
        b.set_axis_off()
        rms = np.sqrt((err**2).mean())
        b.set_title(f"surface, radial error (RMS {rms*1000:.0f} µm)")

    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    fig.colorbar(sm, ax=b, shrink=0.6, label="distance outside true periosteum [mm]")
    frames = list(range(len(rr.surfaces))) + [len(rr.surfaces) - 1] * 4
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=600)
    anim.save(IMG / "refinement.gif", writer=animation.PillowWriter(fps=1.6))
    print("wrote", IMG / "refinement.gif")
    draw(len(rr.surfaces) - 1)
    save(fig, "refinement_final.png")
    # convergence, with the volumetric model as the inner fit for comparison
    rv = cbm.refine_segmentation(img, bad, ph.spacing, **CYL, **METHODS["volumetric"], keep_history=True,
                                 tolerance=0.002, refine_iterations=12)
    dices_v = [2 * (m & (truth > 0)).sum() / (m.sum() + truth.sum())
               for m in (cbm.voxelize(v, f, img.shape, ph.spacing) for v, f in zip(rv.surfaces, rv.surfaces_faces))]
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(8, 3))
    for hist, dc, lab in ((rr.history, dices, "regularized"), (rv.history, dices_v, "volumetric")):
        rms = [hh["rms_displacement"] for hh in hist]
        c1.semilogy(range(1, len(rms) + 1), rms, "o-", label=lab)
        c2.plot(range(len(dc)), dc, "o-", label=lab)
    c1.set_xlabel("iteration")
    c1.set_ylabel("RMS vertex displacement [mm]")
    c1.set_title("geometric convergence (α = 0.5)")
    c1.legend(fontsize=7)
    c2.set_xlabel("iteration")
    c2.set_ylabel("Dice with true segmentation")
    save(fig, "refinement_convergence.png")
    print("refinement: regularized Dice %.4f, volumetric Dice %.4f, iterations %d / %d"
          % (dices[-1], dices_v[-1], len(rr.history), len(rv.history)))
    for lab, hist, dc in (("regularized", rr.history, dices), ("volumetric", rv.history, dices_v)):
        print(f"{lab}: RMS displacement {[round(hh['rms_displacement'], 4) for hh in hist]}; "
              f"Dice {[round(float(d), 4) for d in dc]}; converged {(rr if lab == 'regularized' else rv).converged}")


# --------------------------------------------------------------------------
def fig_benchmark():
    rows = json.loads((DOCS / "results" / "benchmark_results.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.3))
    for name in dict.fromkeys(r["method"] for r in rows):
        rr = [r for r in rows if r["method"] == name]
        n = [r["nodes"] for r in rr]
        a.loglog(n, [r["time"] for r in rr], "o-", label=name)
        b.loglog(n, [r["peak_bytes"] / 1e6 for r in rr], "o-", label=name)
    a.set_xlabel("surface nodes")
    a.set_ylabel("wall time [s]")
    b.set_xlabel("surface nodes")
    b.set_ylabel("peak heap [MB]")
    a.legend(fontsize=7)
    fig.suptitle("Full pipeline (surface → fit), 67 samples per profile, 16 threads")
    save(fig, "benchmark.png")


def fig_bonelab():
    path = DOCS / "results" / "bonelab_comparison.json"
    if not path.exists():
        print("skipping bonelab figure (run compare_bonelab.py first)")
        return
    rows = [r for r in json.loads(path.read_text()) if r["nodes"] == max(x["nodes"] for x in json.loads(path.read_text()))]
    names = [r["method"] for r in rows]
    y = np.arange(len(rows))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.8))
    a.barh(y + 0.2, [r["bonelab_time"] for r in rows], 0.4, label="Bonelab (Python)", color="tab:gray")
    a.barh(y - 0.2, [r["rust_fit_time"] for r in rows], 0.4, label="this crate", color="tab:blue")
    for yi, r in zip(y, rows):
        a.text(r["bonelab_time"] * 1.1, yi, f"{r['bonelab_time'] / r['rust_fit_time']:.0f}×", va="center", fontsize=8)
    a.set_xscale("log")
    a.set_yticks(y, names)
    a.set_xlabel("fit time [s]")
    a.legend(fontsize=7, loc="upper right")
    b.barh(y + 0.2, [r["bonelab_rmse"] * 1000 for r in rows], 0.4, color="tab:gray")
    b.barh(y - 0.2, [r["rust_rmse"] * 1000 for r in rows], 0.4, color="tab:blue")
    b.set_yticks(y, [])
    b.set_xlabel("thickness RMSE [µm]")
    fig.suptitle(f"Same {rows[0]['nodes']} profiles fitted by Bonelab and by this crate (16 threads)")
    save(fig, "bonelab_comparison.png")


FIGS = dict(profile=fig_profile, methods=fig_methods, noise=fig_noise, curvature=fig_curvature,
            refine=fig_refine, benchmark=fig_benchmark, bonelab=fig_bonelab)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
