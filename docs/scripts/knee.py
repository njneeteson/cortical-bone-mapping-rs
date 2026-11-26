"""Every method on a clinical knee CT: four bones segmented and mapped from
the image alone.

The CT (GE Revolution GSI, 0.65 × 0.65 mm pixels, 0.625 mm slices, both
knees) is not part of the repository; it is used with the subject's
permission. The left knee had an injury in which bone broke off the femur
and the patella; the right knee is uninjured. All eight bones are mapped
together (they share the scanner's estimated parameters).

Pipeline (the same start for every method):

1. bone threshold: two-class Otsu of the voxels above -500 HU;
2. bones: ``cbm.separate_bones`` with a landmark in each bone, then
   ``cbm.grow_bones`` into connected voxels above the soft-tissue peak + 4 SD;
3. start: ``cbm.bone_hulls`` with a 4 mm closing;
4. ``cbm.refine_bones`` with each method; the bones share an estimated
   ``rho_c`` (and, for the volumetric model, the PSF), and every other bone
   is excluded while one is refined.

Writes docs/results/knee.json and the docs/images/knee_* figures, including
the refinement GIF of the default method.

    python docs/scripts/knee.py /path/to/DICOM            # compute and draw
    python docs/scripts/knee.py /path/to/DICOM --redraw   # redraw from the last run, if the code is unchanged
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm
from _common import (BONE_COLOURS, LABELS, METHODS, RESULTS, START, draw_segments, load_run, plotter,
                     provenance, pv_image, pv_mesh, save, save_run, screenshot, section_segments, style,
                     write_json, IMAGES)

LANDMARKS = {  # a voxel inside each bone; image columns 0-255 are the right knee
    "right femur": (250, 240, 123), "right tibia": (60, 250, 110), "right patella": (211, 167, 114),
    "right fibula": (60, 296, 43),
    "left femur": (215, 250, 400), "left tibia": (92, 232, 408), "left patella": (235, 172, 421),
    "left fibula": (77, 308, 473),
}
BASE = dict(outside=4.0, inside=8.0, resolution=0.25, taubin_iterations=50, taubin_passband=0.1,
            global_params=(), rho_c="estimate", init=(0.0, 2.0, 0.5, 0.0, 100.0),
            lower=(-4.0, 0.05, 0.05, -300.0, -300.0), upper=(8.0, 8.0, 2.0, 400.0, 1200.0),
            reg_sigma=3.0, reg_neighbours=500, progress_every=0)
METHOD_OPTIONS = {
    "local": dict(method="local"),
    "multigrid": dict(method="multigrid", multigrid_spacings=(16.0, 8.0, 4.0, 2.0)),
    "regularized": dict(method="regularized"),
    "volumetric": dict(method="volumetric", vol_scanner_axis=0),
}
DEFAULT = "volumetric"  # the method whose refinement is animated
REFINE = dict(refine_iterations=150)
BLANK_MM = 10.0  # display only: soft tissue further than this outside the starts is blanked
INPUTS = dict(landmarks=LANDMARKS, base=BASE, methods=METHOD_OPTIONS, refine=REFINE)


def read_dicom(folder):
    """The largest axial CT series in ``folder``, in HU, and its spacing."""
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


def segment(img, sp):
    """Threshold, bones and starts (the same for every method)."""
    thr = cbm.otsu_thresholds(img, floor=-500.0)[0]
    marks = {n: np.array(v, float) * np.array(sp) for n, v in LANDMARKS.items()}
    labels, names, _ = cbm.separate_bones(img, thr, marks, sp, min_size=int(2000 / np.prod(sp)))
    labels = np.where(np.isin(labels, list(names)), labels, 0).astype(labels.dtype)  # the eight named bones only
    low = cbm.soft_tissue_threshold(img, thr, floor=-500.0)
    labels = cbm.grow_bones(img, labels, low)
    starts = cbm.bone_hulls(labels, sp, closing_radius=4.0)
    print(f"threshold {thr:.0f} HU, low threshold {low:.0f} HU, bones {names}", flush=True)
    return thr, low, labels, starts, names


def bone_outside(labels, b, peri, sp):
    """Volume (cm³) of the bone's grown label more than 1 mm outside its refined surface."""
    d = ndimage.distance_transform_edt(~peri, sampling=sp)
    return float(((labels == b) & (d > 1.0)).sum() * np.prod(sp) / 1000)


def run_method(method, img, sp, labels, starts, names):
    keep = method == DEFAULT
    t0 = time.perf_counter()
    # options as an Options object: explicit keyword arguments would override the
    # scanner parameters (rho_c, PSF) that refine_bones estimates once for all bones
    options = cbm.Options(spacing=sp, **BASE, **METHOD_OPTIONS[method])
    res = cbm.refine_bones(img, starts, sp, options=options, names=names, keep_history=keep, **REFINE)
    total = time.perf_counter() - t0
    out = {}
    for b, rr in res.items():
        r = rr.result
        t = r.thickness
        row = dict(moves=len(rr.history) - 1, converged=bool(rr.converged), nodes=int(r.n), rho_c=float(r.rho_c),
                   thickness_median_mm=float(np.median(t)), thickness_p5_mm=float(np.percentile(t, 5)),
                   thickness_p95_mm=float(np.percentile(t, 95)),
                   bone_outside_cm3=bone_outside(labels, b, rr.periosteal_mask, sp))
        if r.volumetric is not None:
            row["psf_mm"] = [float(r.volumetric["sigma_xy"]), float(r.volumetric["sigma_z"])]
        bone = dict(row=row, peri=(r.surface_vertices, r.surface_faces), endo=(rr.endosteal_vertices, r.surface_faces),
                    nodes=(r.points, r.faces), thickness=t)
        if keep:
            bone["history"] = list(zip(rr.surfaces, rr.surfaces_faces))
        out[names[b]] = bone
        print(method, names[b], {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return dict(bones=out, time_s=total)


# ------------------------------------------------------------------ figures

SIDES = ("right", "left")
ENDO_COLOUR = "#ff4fd8"  # the fitted endosteal surfaces, distinct from every bone colour


def colour(name):
    return BONE_COLOURS[name.split()[-1]]


def display_image(d):
    """The image with soft tissue far from every start blanked (display only)."""
    far = ndimage.distance_transform_edt(d["starts"] == 0, sampling=d["spacing"]) > BLANK_MM
    return np.where(far, -1000.0, d["img"]).astype(np.float32)


def side_columns(side):
    return slice(0, 256) if side == "right" else slice(256, None)


def knee_planes(d, side):
    """A knee's sections, from its own anatomy: coronal and sagittal through
    its femur and patella landmarks, axial through the femoral condyles
    (15 mm above the femur start's lowest point)."""
    sp = d["spacing"]
    fem = [b for b, n in d["names"].items() if n == f"{side} femur"][0]
    z_low = np.argwhere(d["starts"] == fem)[:, 0].min()
    return (("coronal", 1, LANDMARKS[f"{side} femur"][1]),
            ("sagittal", 2, LANDMARKS[f"{side} patella"][2]),
            ("axial", 0, int(z_low + round(15.0 / sp[0]))))


def plane_window(d, side, ax_i, k, pad=8):
    """A section's window: the knee's starts it cuts, padded, within the image."""
    other = [a for a in range(3) if a != ax_i]
    st = np.zeros(d["starts"].shape, bool)
    st[:, :, side_columns(side)] = d["starts"][:, :, side_columns(side)] > 0
    idx = np.argwhere(np.take(st, k, ax_i))
    shape = np.array(d["starts"].shape)[other]
    return other, np.maximum(idx.min(0) - pad, 0), np.minimum(idx.max(0) + pad, shape - 1)


def draw_plane(ax, d, disp, side, plane, surfaces, endo=None, lw=1.8):
    """One section of one knee: the image, the starts (dashed) and each
    bone's surfaces; head up in coronal and sagittal sections."""
    _, ax_i, k = plane
    sp = d["spacing"]
    other, lo, hi = plane_window(d, side, ax_i, k)
    ax.imshow(np.take(disp, k, ax_i), cmap="gray", vmin=-200, vmax=1300, aspect=sp[other[0]] / sp[other[1]])
    ax.contour(np.take(d["starts"] > 0, k, ax_i), [0.5], colors=START, linewidths=1.0, linestyles="--")
    mine = [b for b in surfaces if b.startswith(side)]
    for bone in mine if endo else []:  # under the periosteal surfaces
        draw_segments(ax, section_segments(*endo[bone], ax_i, k, sp), ENDO_COLOUR, lw=0.9, ls=":")
    for bone in mine:
        draw_segments(ax, section_segments(*surfaces[bone], ax_i, k, sp), colour(bone), lw=lw)
    ax.set_xlim(lo[1], hi[1])
    if other[0] == 0:
        ax.set_ylim(lo[0], hi[0])  # axis 0 (foot to head) upwards
    else:
        ax.set_ylim(hi[0], lo[0])
    ax.set_xticks([])
    ax.set_yticks([])


def legend(fig, endo=True):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=START, ls="--", lw=2.5, label="start")]
    h += [Line2D([], [], color=c, lw=3, label=b) for b, c in BONE_COLOURS.items()]
    if endo:
        h.append(Line2D([], [], color=ENDO_COLOUR, ls=":", lw=2.5, label="fitted endosteal (all bones)"))
    fig.legend(handles=h, loc="outside lower center", ncol=len(h), frameon=False)


def widths(d, side, planes):
    """Width over height of each section's window, in physical units."""
    out = []
    for _, ax_i, k in planes:
        other, lo, hi = plane_window(d, side, ax_i, k)
        sp = d["spacing"]
        out.append((hi[1] - lo[1]) * sp[other[1]] / ((hi[0] - lo[0]) * sp[other[0]]))
    return out


def fig_sections(d, plt, side):
    disp = display_image(d)
    planes = knee_planes(d, side)
    w = widths(d, side, planes)
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(4.6 * sum(w) + 0.8, 4.6 * len(METHODS)),
                             width_ratios=w, layout="constrained")
    for r, method in enumerate(METHODS):
        bones = d["runs"][method]["bones"]
        peri = {b: v["peri"] for b, v in bones.items()}
        endo = {b: v["endo"] for b, v in bones.items()}
        for c, plane in enumerate(planes):
            draw_plane(axes[r, c], d, disp, side, plane, peri, endo=endo)
            if r == 0:
                axes[r, c].set_title(f"{plane[0]} section")
        axes[r, 0].set_ylabel(LABELS[method], fontsize=15)
    legend(fig)
    save(fig, f"knee_{side}_sections.png")


def render_bones(meshes, scalars_key=None, cmap=None, clim=None, size=(800, 900), image=None):
    p = plotter(size)
    for bone, m in meshes.items():
        if scalars_key:
            p.add_mesh(m, scalars=scalars_key, cmap=cmap, clim=clim, show_scalar_bar=False, smooth_shading=True,
                       ambient=0.3, diffuse=0.75, specular=0.0)
        else:
            p.add_mesh(m, color=colour(bone), smooth_shading=True, ambient=0.25, specular=0.1)
    if image is not None:
        p.add_mesh(image, scalars="intensity", cmap="gray", clim=(-200, 1300), show_scalar_bar=False,
                   lighting=False)
    return p


def fig_thickness(d, plt):
    """Fitted thickness on every bone of each knee, from the front and back."""
    from matplotlib import cm, colors

    clim = (0.0, 3.0)
    views = [(side, view, f"{side} knee, {name}") for side in SIDES
             for view, name in (((0.0, -1.0, 0.0), "anterior"), ((0.0, 1.0, 0.0), "posterior"))]
    fig, axes = plt.subplots(len(METHODS), len(views), figsize=(15, 5.6 * len(METHODS)), layout="constrained")
    for r, method in enumerate(METHODS):
        bones = d["runs"][method]["bones"]
        for c, (side, view, title) in enumerate(views):
            meshes = {b: pv_mesh(*v["nodes"], v["thickness"], "t") for b, v in bones.items() if b.startswith(side)}
            p = render_bones(meshes, "t", "viridis", clim, size=(560, 820))
            p.view_vector(view, viewup=(0, 0, 1))
            p.camera.zoom(1.8)
            ax = axes[r, c]
            ax.imshow(screenshot(p))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(LABELS[method], fontsize=15)
    sm = cm.ScalarMappable(norm=colors.Normalize(*clim), cmap="viridis")
    fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.01).set_label("fitted cortical thickness (mm)")
    save(fig, "knee_thickness.png")


def knee_box(d, side, pad_mm=10.0):
    """A knee's region (voxel slices): its starts' bounding box, padded."""
    sp = np.array(d["spacing"])
    sub = np.zeros(d["starts"].shape, bool)
    sub[:, :, side_columns(side)] = d["starts"][:, :, side_columns(side)] > 0
    idx = np.argwhere(sub)
    pad = np.ceil(pad_mm / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, d["starts"].shape)
    return tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))


def render_3d(d, surf, side, plane, size=(700, 900), opacity=0.3):
    """One knee in 3D: the whole sagittal section of the image, and each
    bone's surface, translucent, with its section drawn on the image."""
    sp = np.array(d["spacing"])
    box = knee_box(d, side)
    grid = pv_image(d["img"][box], sp, np.array([s.start for s in box]) * sp)
    x0 = plane[2] * sp[2]
    toward = -1.0 if side == "right" else 1.0  # camera on the lateral side
    p = plotter(size)
    p.add_mesh(grid.slice(normal="x", origin=(x0, 0.0, 0.0)), scalars="intensity", cmap="gray",
               clim=(-200, 1300), show_scalar_bar=False, lighting=False)
    for b, s in surf.items():
        if not b.startswith(side):
            continue
        mesh = pv_mesh(*s)
        p.add_mesh(mesh, color=colour(b), smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.2,
                   opacity=opacity)
        # the section, nudged towards the camera so that it is drawn over the image
        line = mesh.slice(normal="x", origin=(x0, 0, 0))
        if line.n_points:
            p.add_mesh(line.translate((0.3 * toward, 0, 0)), color=colour(b), line_width=4)
    p.view_vector((toward, -0.85, 0.3), viewup=(0, 0, 1))
    p.camera.zoom(1.45)
    return screenshot(p)


def crop_white(img, pad=6):
    """A rendering cropped to its content (the background is white)."""
    idx = np.argwhere((img[..., :3] < 250).any(-1))
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = idx.max(0) + pad + 1
    return img[lo[0]:hi[0], lo[1]:hi[1]]


def gif_palette():
    """A fixed GIF palette: greys for the image, shades of each bone colour
    and of the start colour (so that thin coloured lines keep their colour),
    and each bone colour blended over greys (the translucent 3D surfaces)."""
    from matplotlib import colors
    from PIL import Image

    pal = [(v, v, v) for v in np.linspace(0, 255, 96).round().astype(int)]
    for c in list(BONE_COLOURS.values()) + [START]:
        rgb = np.array(colors.to_rgb(c))
        for f in np.linspace(0.45, 1.0, 8):
            pal.append(tuple(int(round(255 * x)) for x in rgb * f))
        pal.append(tuple(int(round(255 * x)) for x in rgb + (1 - rgb) * 0.5))
        pal.append(tuple(int(round(255 * x)) for x in rgb + (1 - rgb) * 0.25))
    for c in BONE_COLOURS.values():
        rgb = np.array(colors.to_rgb(c))
        for g in np.linspace(0.0, 1.0, 26):
            pal.append(tuple(int(round(255 * x)) for x in 0.35 * rgb + 0.65 * g))
    pal = pal[:256] + [(0, 0, 0)] * (256 - len(pal))
    img = Image.new("P", (1, 1))
    img.putpalette([v for c in pal for v in c])
    return img


def fig_gif(d, plt):
    """The default method's refinement of every bone, one frame per move:
    each knee's three sections, and a 3D view of its translucent surfaces
    over the whole sagittal section."""
    from PIL import Image

    bones = d["runs"][DEFAULT]["bones"]
    hist = {b: v["history"] for b, v in bones.items()}
    n = max(len(h) for h in hist.values())
    disp = display_image(d)
    planes = {side: knee_planes(d, side) for side in SIDES}
    w = {side: widths(d, side, planes[side]) for side in SIDES}
    frames = []
    for k in range(n):
        surf = {b: h[min(k, len(h) - 1)] for b, h in hist.items()}
        fig = plt.figure(figsize=(16, 9.6), layout="constrained")
        sub = fig.subfigures(2, 1)
        for side, sf in zip(SIDES, sub):
            axes = sf.subplots(1, 4, width_ratios=w[side] + [0.75])
            for ax, plane in zip(axes, planes[side]):
                draw_plane(ax, d, disp, side, plane, surf, lw=2.4)
                ax.set_title(f"{plane[0]} section")
            axes[3].imshow(crop_white(render_3d(d, surf, side, planes[side][1])))
            axes[3].set_axis_off()
            axes[3].set_title("3D, with the sagittal section")
            sf.supylabel(f"{side} knee", fontsize=16)
        moving = [b for b, h in hist.items() if k < len(h) - 1]
        status = "start" if k == 0 else f"move {k}: " + (f"{len(moving)} of {len(hist)} bones still moving"
                                                          if moving else "every bone has converged")
        fig.suptitle(f"{LABELS[DEFAULT].capitalize()} refinement, {status}", fontsize=17)
        legend(fig, endo=False)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    pal = gif_palette()
    frames = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    durations = ([1600] + [900] * (n - 2) + [3500]) if n > 1 else [3500]
    out = IMAGES / "knee_refinement.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print("wrote", out)


def draw(d):
    plt = style()
    for side in SIDES:
        fig_sections(d, plt, side)
    fig_thickness(d, plt)
    fig_gif(d, plt)


# ------------------------------------------------------------------ export
# File formats are the user's business; the library takes and returns arrays.
# These two helpers only show how the masks go to and from a segmentation
# editor (ITK-SNAP, MITK, 3D Slicer) as NIfTI, on the DICOM's own grid.

LABEL_ORDER = ["right femur", "right tibia", "right patella", "right fibula",
               "left femur", "left tibia", "left patella", "left fibula"]  # NIfTI label values 1-8


def dicom_affine(folder):
    """The NIfTI (RAS) affine of the series ``read_dicom`` reads, for an
    array transposed to (column, row, slice)."""
    import pydicom

    series = {}
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file() or f.name.startswith("._") or f.name == "DICOMDIR":
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if "ImagePositionPatient" in ds and "AXIAL" in list(ds.get("ImageType", ["AXIAL"])):
            series.setdefault(ds.SeriesInstanceUID, []).append(ds)
    ds = max(series.values(), key=len)
    ds.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    iop = np.array(ds[0].ImageOrientationPatient, float)
    dr, dc = (float(v) for v in ds[0].PixelSpacing)
    p0 = np.array(ds[0].ImagePositionPatient, float)
    lps = np.eye(4)
    lps[:3, 0] = iop[:3] * dc  # along a row: increasing column
    lps[:3, 1] = iop[3:] * dr  # down a column: increasing row
    lps[:3, 2] = (np.array(ds[-1].ImagePositionPatient, float) - p0) / (len(ds) - 1)
    lps[:3, 3] = p0
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps  # DICOM LPS to NIfTI RAS


def read_labels(path):
    """A NIfTI label map as an array in (slice, row, column) order, and its affine."""
    import nibabel as nib

    im = nib.load(path)
    return np.ascontiguousarray(np.asarray(im.dataobj).transpose(2, 1, 0)), im.affine


def export_labels(d, dicom, out_dir):
    """The default method's refined periosteal masks as one NIfTI label map
    (values in ``LABEL_ORDER``) with an ITK-SNAP label description file."""
    import nibabel as nib
    from matplotlib import colors

    labels = np.zeros(d["img"].shape, np.uint8)
    for i, name in enumerate(LABEL_ORDER, start=1):
        v, f = d["runs"][DEFAULT]["bones"][name]["peri"]
        m = cbm.voxelize(v, f, labels.shape, d["spacing"], (0.0, 0.0, 0.0))
        labels[m & (labels == 0)] = i
    affine = dicom_affine(dicom)
    im = nib.Nifti1Image(np.ascontiguousarray(labels.transpose(2, 1, 0)), affine)
    im.set_qform(affine, code=1)
    im.set_sform(affine, code=1)
    out = Path(out_dir) / f"knee_{DEFAULT}_labels.nii.gz"
    nib.save(im, out)
    rgb = {b: tuple(int(round(255 * x)) for x in colors.to_rgb(c)) for b, c in BONE_COLOURS.items()}
    lines = ['    0     0    0    0        0  0  0    "Clear Label"']
    lines += [f'{i:5d} {r:5d} {g:4d} {b:4d}        1  1  1    "{name}"'
              for i, name in enumerate(LABEL_ORDER, start=1) for r, g, b in [rgb[name.split()[-1]]]]
    (Path(out_dir) / "knee_labels.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, "and", Path(out_dir) / "knee_labels.txt")


def main():
    """``knee.py DICOM_DIR [--redraw] [--export OUT_DIR]``"""
    args = sys.argv[1:]
    export = args[args.index("--export") + 1] if "--export" in args else None
    args = [a for a in args if not a.startswith("--") and a != export]
    if not args:
        raise SystemExit(__doc__)
    inputs = dict(INPUTS, dicom=Path(args[0]).name)
    saved = load_run("knee", __file__, inputs) if "--redraw" in sys.argv else None
    if saved is None:
        img, sp = read_dicom(args[0])
        print(f"image {img.shape}, spacing {tuple(round(s, 4) for s in sp)} mm", flush=True)
        thr, low, labels, starts, names = segment(img, sp)
        prov = provenance(__file__, inputs)
        runs = {m: run_method(m, img, sp, labels, starts, names) for m in METHODS}
        d = dict(img=img, spacing=sp, labels=labels, starts=starts, names=names, runs=runs)
        write_json(RESULTS / "knee.json", prov, dict(
            threshold_hu=float(thr), low_threshold_hu=float(low), spacing_mm=list(sp),
            methods={m: dict(time_s=v["time_s"], bones={b: x["row"] for b, x in v["bones"].items()})
                     for m, v in runs.items()}))
        save_run("knee", prov, d)
    else:
        d = saved["data"]
    draw(d)
    if export:
        export_labels(d, args[0], export)


if __name__ == "__main__":
    main()
