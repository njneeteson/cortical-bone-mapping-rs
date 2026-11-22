"""Cortical bone mapping of a CT image, from the image alone.

    python docs/scripts/ct_example.py DICOM_FOLDER [--crop Z0:Z1,Y0:Y1,X0:X1]
        [--threshold otsu|VALUE] [--otsu-floor HU] [--min-volume MM3] [--max-bones N]
        [--landmark NAME=Z,Y,X ...] [--closing MM] [--progress N] [--slices Z,Y,X]
        [--figures PREFIX] [--vtk OUTDIR] [--cache FILE]

Every processing step is a library call; this script only reads the DICOM
series (or a .npy volume with ``--spacing``), crops it, and draws figures.

1. Threshold: ``--threshold VALUE`` in the image's units, or ``otsu`` (the
   default): the two-class Otsu threshold of the voxels above
   ``--otsu-floor`` (``cbm.otsu_thresholds``), for images without a known
   bone threshold such as non-calibrated CT.
2. Bones: ``cbm.label_bones``, numbered by volume (1 = largest). With
   landmarks (``--landmark femur=250,240,123``, voxel indices of the full
   image), ``cbm.separate_bones`` names the bones and, where two landmarks
   fall on one bone, raises the threshold within that bone until they split.
3. Starting segmentation: ``cbm.bone_hulls`` with ``closing_radius``: each
   bone's morphological closing by a ball of ``--closing`` mm (default 4),
   with its voids filled, the bones kept apart. Concavities narrower than
   about twice the radius are filled, wider ones (the femoral intercondylar
   notch) are kept. ``--closing 0`` uses the per-slice convex hull instead.
4. ``cbm.refine_bones``: CBM segmentation refinement, one bone (one closed
   surface) at a time, from the smallest to the largest: per-node densities,
   the surface carried as a level set (remeshed every iteration), inexact
   inner fits and regional convergence (see docs/refinement_experiments.md).
   A progress line every ``--progress`` solver iterations and after every
   fit.

Figures: ``PREFIX_threshold.png``, ``PREFIX_segmentation.png``,
``PREFIX_refinement.gif``, ``PREFIX_thickness.png``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import cortical_bone_mapping_rs as cbm

DOCS = Path(__file__).resolve().parents[1]

# Options for clinical CT (~0.5-1 mm voxels): profiles 4 mm out / 8 mm in sampled
# every 0.25 mm; regularisation over ~3 mm; per-node (regularised) soft-tissue and
# trabecular densities, since the trabecular density varies over a large field of
# view; the cortical density is estimated per bone; bounds in mm and image units (HU).
OPTIONS = cbm.Options(
    outside=4.0, inside=8.0, resolution=0.25,
    taubin_iterations=50, taubin_passband=0.1,
    method="regularized", reg_sigma=3.0, reg_neighbours=500,
    global_params=(),
    rho_c="estimate",
    init=(0.0, 2.0, 0.5, 0.0, 100.0),
    lower=(-4.0, 0.05, 0.05, -300.0, -300.0),
    upper=(8.0, 8.0, 2.0, 400.0, 1200.0),
)
REFINE = cbm.RefineOptions(refine_iterations=150, keep_history=True, level_set=True, freeze=True)


def read_dicom(folder):
    """The largest axial CT series in ``folder``: image (z, y, x) in the
    stored units after the rescale slope/intercept (HU for CT), spacing."""
    import pydicom

    series = {}
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file() or f.name.startswith("._") or f.name == "DICOMDIR":
            continue
        try:
            d = pydicom.dcmread(f)
        except Exception:
            continue
        if "ImagePositionPatient" not in d or "AXIAL" not in list(d.get("ImageType", ["AXIAL"])):
            continue
        series.setdefault(d.SeriesInstanceUID, []).append(d)
    if not series:
        raise SystemExit(f"no axial image series in {folder}")
    ds = max(series.values(), key=len)
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    img = np.stack([d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1))
                    + float(d.get("RescaleIntercept", 0)) for d in ds])
    py, px = (float(v) for v in ds[0].PixelSpacing)
    return img, (float(np.median(np.diff(z))), py, px)


def parse_crop(text, shape):
    if not text:
        return tuple(slice(0, n) for n in shape)
    parts = text.split(",")
    if len(parts) != 3:
        raise SystemExit("--crop needs three ranges, e.g. :,:,0:256")
    out = []
    for p, n in zip(parts, shape):
        a, _, b = p.partition(":")
        out.append(slice(*slice(int(a) if a else None, int(b) if b else None).indices(n)[:2]))
    return tuple(out)


def run(a):
    if a.image.endswith(".npy"):
        image = np.load(a.image).astype(np.float32)
        spacing = tuple(float(v) for v in a.spacing.split(","))
    else:
        image, spacing = read_dicom(a.image)
    print(f"image {image.shape}, spacing {tuple(round(s, 4) for s in spacing)} mm", flush=True)
    crop = parse_crop(a.crop, image.shape)
    img = np.ascontiguousarray(image[crop])
    origin = tuple(s.start * h for s, h in zip(crop, spacing))  # physical coordinates of the full image
    thr = a.threshold
    if thr == "otsu":
        thr = cbm.otsu_thresholds(img, floor=a.otsu_floor)[0]
        print(f"Otsu threshold of the voxels above {a.otsu_floor:g}: {thr:.1f}", flush=True)
    else:
        thr = float(thr)
    min_size = max(1, int(a.min_volume / np.prod(spacing)))
    landmarks = {}
    for spec in a.landmark or []:
        name, _, ijk = spec.partition("=")
        landmarks[name] = np.array([float(v) for v in ijk.split(",")]) * np.array(spacing)
    if landmarks:
        labels, names, used = cbm.separate_bones(img, thr, landmarks, spacing, origin, min_size=min_size,
                                                 max_bones=a.max_bones)
    else:
        labels, _ = cbm.label_bones(img, thr, min_size=min_size, max_bones=a.max_bones)
        names, used = {}, {}
    hulls = cbm.bone_hulls(labels, spacing)  # per-slice convex hulls (kept for comparison)
    starts = cbm.bone_hulls(labels, spacing, closing_radius=a.closing) if a.closing > 0 else hulls
    o = OPTIONS
    t0 = time.perf_counter()
    results = cbm.refine_bones(img, starts, spacing, options=o, refine=REFINE, names=names,
                               progress_every=a.progress, origin=origin)
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
    for b, rr in sorted(results.items()):
        r = rr.result
        print(f"bone {b}{' (' + names[b] + ')' if b in names else ''}: {r.n} nodes, {len(rr.history) - 1} moves, "
              f"converged {rr.converged}, rho_c {r.rho_c:.0f}, thickness median {np.median(r.thickness):.2f} mm "
              f"(5-95%: {np.percentile(r.thickness, 5):.2f}-{np.percentile(r.thickness, 95):.2f})", flush=True)
    body = img[img >= a.otsu_floor]
    hist = np.histogram(body, bins=np.linspace(a.otsu_floor, float(body.max()), 440))
    return dict(img=img, spacing=spacing, origin=origin, threshold=thr, labels=labels, hulls=hulls,
                starts=starts, closing=a.closing,
                names=names, used=used, results=results, hist=hist)


# ---------------------------------------------------------------- figures

def colour(b):
    import matplotlib.pyplot as plt

    return plt.cm.tab10((b - 1) % 10)


def bone_label(d, b):
    n = d["names"].get(b)
    vol = (d["labels"] == b).sum() * np.prod(d["spacing"]) / 1000
    return f"bone {b}" + (f" ({n})" if n else "") + f", {vol:.0f} cm³"


def sections(d, slices):
    if slices:
        z, y, x = (int(v) for v in slices.split(","))
    else:
        idx = np.argwhere(d["labels"] > 0)
        z, y, x = ((idx.min(0) + idx.max(0)) // 2).tolist()
    return [("coronal", y), ("sagittal", x), ("axial", z)]


def cut(a, kind, k):
    return {"axial": lambda: a[k], "coronal": lambda: a[:, k, :], "sagittal": lambda: a[:, :, k]}[kind]()


BONE_MARGIN = 3.0  # mm of context around the bones in the slice figures


def show(ax, d, kind, k):
    """A section of the image, cropped to the bones (plus BONE_MARGIN) so
    that the figures show bone, not the body outline."""
    sp = d["spacing"]
    aspect = {"axial": sp[1] / sp[2], "coronal": sp[0] / sp[2], "sagittal": sp[0] / sp[1]}[kind]
    if "near_bone" not in d:  # everything further than BONE_MARGIN from the bone hulls is blanked
        from scipy import ndimage

        d["near_bone"] = ndimage.distance_transform_edt(d.get("starts", d["hulls"]) == 0, sampling=sp) <= BONE_MARGIN
    sl = np.where(cut(d["near_bone"], kind, k), cut(d["img"], kind, k), -1000.0)
    ax.imshow(sl, cmap="gray", vmin=-200, vmax=1400, origin="upper" if kind == "axial" else "lower", aspect=aspect)
    idx = np.argwhere(d.get("starts", d["hulls"]) > 0)
    m = np.ceil(BONE_MARGIN / np.asarray(sp)).astype(int)
    lo, hi = idx.min(0) - m, idx.max(0) + m
    rows, cols = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[kind]
    ax.set_xlim(lo[cols], hi[cols])
    ax.set_ylim((hi[rows], lo[rows]) if kind == "axial" else (lo[rows], hi[rows]))
    ax.set_xticks([])
    ax.set_yticks([])


def fig_threshold(d, plt, prefix):
    h, e = d["hist"]
    c = 0.5 * (e[1:] + e[:-1])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(c, np.maximum(h, 1), color="k", lw=1)
    ax.axvline(d["threshold"], color="tab:red", ls="--")
    ax.text(d["threshold"] + 20, h.max() / 3, f"threshold {d['threshold']:.0f}", color="tab:red")
    ax.set_xlabel("image value")
    ax.set_ylabel("voxels")
    ax.set_title("Image histogram and the bone threshold")
    fig.tight_layout()
    fig.savefig(f"{prefix}_threshold.png", dpi=110)
    plt.close(fig)


def fig_segmentation(d, plt, prefix, slices):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    for ax, (kind, k) in zip(axes, sections(d, slices)):
        show(ax, d, kind, k)
        for b, rr in d["results"].items():
            ax.contour(cut(d.get("starts", d["hulls"]) == b, kind, k), [0.5], colors="yellow", linewidths=0.7,
                       linestyles="--")
            ax.contour(cut(rr.periosteal_mask, kind, k), [0.5], colors=[colour(b)], linewidths=1.1)
            ax.contour(cut(rr.endosteal_mask, kind, k), [0.5], colors=[colour(b)], linewidths=0.8, linestyles=":")
        ax.set_title(f"{kind} {k}")
    handles = [Line2D([], [], color="yellow", ls="--", label="start")]
    handles += [Line2D([], [], color=colour(b), label=bone_label(d, b)) for b in sorted(d["results"])]
    handles += [Line2D([], [], color="0.6", ls=":", label="fitted endosteal")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False)
    start = (f"{d['closing']:g} mm closing" if d.get("closing", 0) > 0 else "Hull")
    fig.suptitle(f"{start} at threshold {d['threshold']:.0f} → CBM refinement (solid: periosteal), one bone at a time")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(f"{prefix}_segmentation.png", dpi=90)
    plt.close(fig)


def fig_refinement_gif(d, plt, prefix, slices):
    from matplotlib import animation

    shape, sp, org = d["img"].shape, d["spacing"], d["origin"]
    n_frames = max(len(rr.surfaces) for rr in d["results"].values())
    views = sections(d, slices)[:2]
    frames = []
    for it in range(n_frames):
        m = {}
        for b, rr in d["results"].items():
            j = min(it, len(rr.surfaces) - 1)
            m[b] = cbm.voxelize(rr.surfaces[j], rr.surfaces_faces[j], shape, sp, org)
        frames.append(m)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    def draw(it):
        for ax, (kind, k) in zip(axes, views):
            ax.clear()
            show(ax, d, kind, k)
            for b, m in frames[it].items():
                ax.contour(cut(m, kind, k), [0.5], colors=[colour(b)], linewidths=1.0)
            ax.set_title(f"{kind}: iteration {it}" + (" (start)" if it == 0 else ""))
        fig.tight_layout()

    seq = list(range(n_frames)) + [n_frames - 1] * 4
    animation.FuncAnimation(fig, draw, frames=seq, interval=600).save(
        f"{prefix}_refinement.gif", writer=animation.PillowWriter(fps=1.6))
    plt.close(fig)


def fig_thickness(d, plt, prefix):
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    res = {b: rr.result for b, rr in d["results"].items()}
    allt = np.concatenate([r.thickness for r in res.values()])
    norm = Normalize(0.0, float(np.percentile(allt, 98)))
    cmap = plt.cm.viridis
    light = np.array([0.5, 0.6, 0.6])
    light /= np.linalg.norm(light)
    # plot axes: X = patient left (+x), Y = anterior (−y), Z = superior (+z) for standard axial DICOM
    to_plot = lambda p: p[:, ::-1] * np.array([1.0, -1.0, 1.0])  # noqa: E731
    views = [("anterior", 90), ("patient left", 0), ("posterior", -90), ("patient right", 180)]
    allp = np.concatenate([to_plot(r.points) for r in res.values()])
    lo, hi = allp.min(0), allp.max(0)
    fig = plt.figure(figsize=(16, 9))
    for i, (title, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        for r in res.values():
            tri = to_plot(r.points)[r.faces]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            col = cmap(norm(r.thickness[r.faces].mean(1)))
            col[:, :3] *= (0.55 + 0.45 * np.abs(nrm @ light))[:, None]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo), zoom=1.6)
        ax.view_init(elev=5, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.12, wspace=0)
    cax = fig.add_axes((0.3, 0.06, 0.4, 0.025))
    fig.colorbar(sm, cax=cax, orientation="horizontal", label="cortical thickness [mm]")
    fig.suptitle("Fitted cortical thickness on the refined periosteal surfaces")
    fig.savefig(f"{prefix}_thickness.png", dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="DICOM folder, or a .npy volume (with --spacing)")
    ap.add_argument("--spacing", help="voxel size z,y,x in mm for a .npy volume")
    ap.add_argument("--crop", help="voxel ranges z0:z1,y0:y1,x0:x1 (empty = all), e.g. :,:,0:256")
    ap.add_argument("--threshold", default="otsu", help="bone threshold in image units, or 'otsu' (default)")
    ap.add_argument("--otsu-floor", type=float, default=-500.0, help="Otsu: only voxels above this (default -500)")
    ap.add_argument("--min-volume", type=float, default=2000.0, help="smallest bone in mm^3 (default 2000)")
    ap.add_argument("--max-bones", type=int, help="keep at most this many bones (the largest)")
    ap.add_argument("--landmark", action="append",
                    help="NAME=Z,Y,X: a voxel (full-image indices) in that bone; repeat for several bones")
    ap.add_argument("--closing", type=float, default=4.0,
                    help="start from each bone's closing by a ball of this radius in mm "
                         "(default 4; 0 = per-slice convex hull)")
    ap.add_argument("--progress", type=int, default=1,
                    help="print the fit solver state every N iterations (0 = one line per refinement iteration)")
    ap.add_argument("--slices", help="Z,Y,X (cropped indices) of the sections in the figures")
    ap.add_argument("--figures", default=str(DOCS / "images" / "ct"), help="figure file prefix")
    ap.add_argument("--vtk", help="write each bone's thickness map as VTK into this folder")
    ap.add_argument("--cache", help="pickle the results here and reuse them on the next run (figures only)")
    a = ap.parse_args(argv)
    if a.cache and Path(a.cache).exists():
        d = pickle.loads(Path(a.cache).read_bytes())
    else:
        d = run(a)
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps(d))
    if a.vtk:
        Path(a.vtk).mkdir(parents=True, exist_ok=True)
        for b, rr in d["results"].items():
            rr.result.save_vtk(Path(a.vtk) / f"bone_{b}{'_' + d['names'][b] if b in d['names'] else ''}.vtk")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_threshold(d, plt, a.figures)
    fig_segmentation(d, plt, a.figures, a.slices)
    fig_thickness(d, plt, a.figures)
    fig_refinement_gif(d, plt, a.figures, a.slices)
    print("wrote", ", ".join(f"{a.figures}_{s}" for s in ("threshold.png", "segmentation.png", "thickness.png",
                                                           "refinement.gif")))


if __name__ == "__main__":
    sys.exit(main())
