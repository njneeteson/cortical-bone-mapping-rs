"""The periarticular region of an HR-pQCT proximal tibia: the 1D global
regularised method refined from our own start, against the provided cortical
and trabecular masks. (The local method diverges here, and the volumetric
method is left out until its refinement on HR-pQCT is tuned: each of its
moves rebuilds a band of about 345 million voxels on this slab.)

The scan (XtremeCT II, 60.7 µm voxels; TRIKNEE, participant 22, right tibia,
scan 1) is not part of the repository; it is used with the participant's
permission. Its provided masks: ``*_CORT_MASK.AIM`` (cortex),
``*_TRAB_MASK.AIM`` (trabecular bone) and ``*_ROI3?_MASK.AIM`` (the
periarticular analysis regions below the medial and lateral contact
surfaces; their union is the region studied here).

Pipeline:

1. the slab: every slice from ``inside + 1 mm`` below the regions to
   ``outside + 1 mm`` above the top of the bone over them, cropped in-plane
   to the bone of those slices padded by ``outside + 1 mm``;
2. our start: bones labelled at 320 mg HA/cm³ (``cbm.label_bones``), the
   tibia the label overlapping the provided bone mask most,
   ``cbm.bone_hulls`` with a 4 mm closing;
3. ``cbm.refine_bones`` on the tibia (the other bones excluded) with the 1D
   global regularised method (HR-pQCT settings below), at most 15 moves (beyond about 5 the moves are below half a voxel and the masks
   barely change);
4. comparison in the region: each column of the regions from their bottom
   through the bone's top (the articular surface) and 1 mm above it.
   Labels: cortex, trabecular (bone that is not cortex), background; Dice
   per class, and the mean local thickness of each cortical mask
   (local-thickness-rs) in the region.

Writes examples/results/hrpqct_periarticular.json and the
examples/images/hrpqct_periarticular_* figures: per method, six sections
(three coronal, three sagittal) where the masks
disagree most, each as the provided masks, the refined masks and the
disagreement over the image; and the refined surfaces with those sections
in 3D.

    python examples/hrpqct_periarticular.py /path/to/TRIKNEE_folder TRIKNEE_22_TR_1
    python examples/hrpqct_periarticular.py ... --redraw   # from the last run, if the code is unchanged
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import cortical_bone_mapping_rs as cbm
from _common import RESULTS, load_run, plotter, provenance, pv_image, pv_mesh, save, save_run, screenshot, style, write_json
from aim import read_aim

VOXEL = 0.0607  # mm
BONE_THRESHOLD = 320.0  # mg HA/cm³, for our start's bone labels
CLOSING_MM = 4.0
SIGMA_VOXELS = 25  # σ_reg in voxels
METHODS = ("regularized",)
SETTINGS = dict(rho_c="estimate", normal_constraint="none", progress_every=0, grid_step=4, inside=5.0,
                reg_sigma=SIGMA_VOXELS * VOXEL, reg_cutoff=3.0, reliability=False)
METHOD_SETTINGS = {"regularized": {}}
REFINE = dict(refine_iterations=15)
REGION_ABOVE_MM = 1.0  # the comparison region reaches this far above the bone's top
INPUTS = dict(voxel=VOXEL, threshold=BONE_THRESHOLD, closing=CLOSING_MM, settings=SETTINGS,
              methods=METHOD_SETTINGS, refine=REFINE, region_above=REGION_ABOVE_MM)
LOG = Path(tempfile.gettempdir()) / "cbm_hrpqct_periarticular.log"


def read_scan(folder, name):
    """The image (mg HA/cm³), the provided masks and the union of the
    analysis regions, on the image's grid, and the spacing."""
    base = str(Path(folder) / name)
    raw, _, sp, (a, b) = read_aim(base + ".AIM")
    cort = read_aim(base + "_CORT_MASK.AIM")[0] > 0
    trab = read_aim(base + "_TRAB_MASK.AIM")[0] > 0
    roi = np.zeros(cort.shape, bool)
    for f in sorted(Path(folder).glob(name + "_ROI3?_MASK.AIM")):
        roi |= read_aim(str(f))[0] > 0
    return raw, (a, b), cort, trab, roi, tuple(float(s) for s in sp)


def slab_box(cort, trab, roi, sp):
    """The slab (voxel slices) around the regions (module documentation)."""
    bone = cort | trab
    pad_in = int(np.ceil((SETTINGS["inside"] + 1.0) / sp[0]))
    pad_out = [int(np.ceil((5.0 + 1.0) / s)) for s in sp]  # the preset's 5 mm outside, + 1 mm
    z = np.flatnonzero(roi.any((1, 2)))
    cols = roi.any(0)
    top = int(np.flatnonzero(bone[:, cols].any(1)).max())
    z0, z1 = max(int(z.min()) - pad_in, 0), min(top + pad_out[0] + 1, bone.shape[0])
    sub = bone[z0:z1]
    y = np.flatnonzero(sub.any((0, 2)))
    x = np.flatnonzero(sub.any((0, 1)))
    return (slice(z0, z1),
            slice(max(int(y.min()) - pad_out[1], 0), min(int(y.max()) + pad_out[1] + 1, bone.shape[1])),
            slice(max(int(x.min()) - pad_out[2], 0), min(int(x.max()) + pad_out[2] + 1, bone.shape[2])))


def comparison_region(roi, bone, sp):
    """Each column of the regions from their bottom to 1 mm above the bone's
    top in that column (module documentation)."""
    nz = roi.shape[0]
    cols = roi.any(0)
    bottom = np.where(cols, roi.argmax(0), nz)
    has_bone = bone.any(0)
    top = np.where(has_bone, nz - 1 - bone[::-1].argmax(0), -1)
    upper = np.minimum(top + int(np.ceil(REGION_ABOVE_MM / sp[0])), nz - 1)
    zz = np.arange(nz)[:, None, None]
    return cols[None] & has_bone[None] & (zz >= bottom[None]) & (zz <= upper[None])


def labels_of(cortex, bone):
    """0 background, 1 cortex, 2 trabecular (bone that is not cortex)."""
    lab = np.zeros(bone.shape, np.uint8)
    lab[bone] = 2
    lab[cortex] = 1
    return lab


def start(img, ref_bone, sp):
    """Our start: bone labels, the tibia's label, the starting masks."""
    labels, _ = cbm.label_bones(img, BONE_THRESHOLD, min_size=100000)
    overlap = np.bincount(labels[ref_bone].ravel())
    overlap[0] = 0
    tib = int(overlap.argmax())
    starts = cbm.bone_hulls(labels, sp, closing_radius=CLOSING_MM)
    return labels, tib, starts


def method_options(method, starts, tib, sp):
    """The HR-pQCT options, with the neighbour count reaching 3 σ_reg at the
    node spacing of the starting surface."""
    base = cbm.Options.preset("knee", voxel=VOXEL, spacing=sp, method=method, **SETTINGS,
                              **METHOD_SETTINGS[method])
    v, f, _ = cbm.mask_surface(starts == tib, sp, options=base)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    k = int(np.ceil(np.pi * (3 * SETTINGS["reg_sigma"]) ** 2 / (area / len(v))))
    return cbm.Options(**{**base.__dict__, "reg_neighbours": k})


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def region_box(region, sp, pad_mm=1.0):
    """The region's bounding box (voxel slices), padded."""
    idx = np.argwhere(region.any(1).any(1))[:, 0], np.argwhere(region.any(0).any(1))[:, 0], np.argwhere(region.any(0).any(0))[:, 0]
    pad = [int(np.ceil(pad_mm / s)) for s in sp]
    return tuple(slice(max(int(i.min()) - p, 0), min(int(i.max()) + p + 1, n))
                 for i, p, n in zip(idx, pad, region.shape))


def mean_thickness(cortex, region, sp):
    """Mean local thickness (mm) of a cortical mask over its voxels in the
    region (computed on the region's box padded by 1 mm, more than the
    cortex's thickness)."""
    import localthickness_rs as lt

    box = region_box(region, sp)
    c, r = cortex[box], region[box]
    if not (c & r).any():
        return float("nan")
    return float(lt.local_thickness(c, spacing=sp)[c & r].mean())


def for_figures(img, sp, ref, region, runs):
    """What the figures need, on the region's box padded by 1 mm (surfaces
    moved to its frame)."""
    box = region_box(region, sp)
    shift = np.array([s.start for s in box]) * np.array(sp)
    return dict(img=img[box], spacing=sp, ref=ref[box], region=region[box],
                runs={m: dict(labels=v["labels"][box], surface=(v["surface"][0] - shift, v["surface"][1]))
                      for m, v in runs.items()})


def run_method(method, img, starts, tib, sp, ref_lab, region):
    t0 = time.perf_counter()
    options = method_options(method, starts, tib, sp)
    rr = cbm.refine_bones(img, starts, sp, bones=[tib], options=options, verbose=False, masks=True, **REFINE)[tib]
    elapsed = time.perf_counter() - t0
    lab = labels_of(rr.cortical_mask, rr.periosteal_mask)
    row = dict(time_s=round(elapsed, 1), moves=len(rr.history) - 1, converged=bool(rr.converged),
               nodes=int(rr.result.n), rho_c=float(rr.result.rho_c),
               dice_bone=dice((lab > 0) & region, (ref_lab > 0) & region),
               dice_cortex=dice((lab == 1) & region, (ref_lab == 1) & region),
               dice_trabecular=dice((lab == 2) & region, (ref_lab == 2) & region),
               ct_th_mm=mean_thickness(lab == 1, region, sp))
    print(method, row, flush=True)
    return dict(row=row, labels=lab, surface=(rr.result.surface_vertices, rr.result.surface_faces))


# ------------------------------------------------------------------ figures

CORTEX, TRAB, DIFF = (0.1, 0.85, 0.2), (0.15, 0.45, 1.0), (1.0, 0.1, 0.1)
WINDOW_MM = 10.0  # the side of each section's window
MIN_APART_MM = 3.0  # between sections of the same plane
CLIM = (-200.0, 1400.0)  # mg HA/cm³


def sections(d):
    """The six sections (three coronal, three sagittal) where the masks
    disagree most, summed over the methods, at least MIN_APART_MM apart; each
    with its window (centred on the densest disagreement in the plane)."""
    from scipy import ndimage

    sp = d["spacing"]
    dis = np.zeros(d["region"].shape, np.uint8)
    for m in METHODS:
        dis += (d["runs"][m]["labels"] != d["ref"]) & d["region"]
    half = [int(round(WINDOW_MM / 2 / s)) for s in sp]
    out = []
    for axis, name in ((1, "coronal"), (2, "sagittal")):
        counts = dis.sum(axis=tuple(a for a in range(3) if a != axis)).astype(float)
        apart = int(round(MIN_APART_MM / sp[axis]))
        for _ in range(3):
            k = int(counts.argmax())
            counts[max(k - apart, 0):k + apart + 1] = -1
            plane = np.take(dis, k, axis=axis).astype(float)
            dense = ndimage.uniform_filter(plane, size=(2 * half[0] + 1, 2 * half[2 if axis == 1 else 1] + 1))
            r, c = np.unravel_index(int(dense.argmax()), dense.shape)
            hc = half[2 if axis == 1 else 1]
            r0, c0 = max(r - half[0], 0), max(c - hc, 0)
            r1, c1 = min(r0 + 2 * half[0] + 1, plane.shape[0]), min(c0 + 2 * hc + 1, plane.shape[1])
            out.append(dict(axis=axis, name=name, index=k, rows=(r0, r1), cols=(c0, c1)))
    return out


def plane_of(a, s):
    return np.take(a, s["index"], axis=s["axis"])[s["rows"][0]:s["rows"][1], s["cols"][0]:s["cols"][1]]


def overlay(ax, image, layers):
    """The image in grey, each (mask, colour, alpha) layer over it; axis 0
    (the scanner axis, up the tibia) points up."""
    ax.imshow(image, cmap="gray", vmin=CLIM[0], vmax=CLIM[1], origin="lower")
    for mask, colour, alpha in layers:
        rgba = np.zeros(mask.shape + (4,))
        rgba[mask] = (*colour, alpha)
        ax.imshow(rgba, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])


def fig_sections(d, method, secs, plt):
    from matplotlib.patches import Patch

    sp = d["spacing"]
    lab = d["runs"][method]["labels"]
    fig, axes = plt.subplots(len(secs), 3, figsize=(12, 4.1 * len(secs)), layout="constrained")
    for r, s in enumerate(secs):
        image = plane_of(d["img"], s)
        ref, ours = plane_of(d["ref"], s), plane_of(lab, s)
        region = plane_of(d["region"], s)
        cells = [
            [(ref == 1, CORTEX, 0.35), (ref == 2, TRAB, 0.3)],
            [(ours == 1, CORTEX, 0.35), (ours == 2, TRAB, 0.3)],
            [((ref != ours) & region, DIFF, 0.55)],
        ]
        for c, layers in enumerate(cells):
            overlay(axes[r, c], image, layers)
        pos = s["index"] * sp[s["axis"]]
        axes[r, 0].set_ylabel(f"{r + 1}: {s['name']} {s['index']} ({pos:.1f} mm)")
    for c, title in enumerate(("provided masks", f"refined ({METHOD_LABELS[method]})", "disagreement (in the region)")):
        axes[0, c].set_title(title)
    fig.legend(handles=[Patch(color=CORTEX, alpha=0.6, label="cortex"), Patch(color=TRAB, alpha=0.6, label="trabecular"),
                        Patch(color=DIFF, alpha=0.7, label="labels differ")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, f"hrpqct_periarticular_{method}.png")


def section_quad(d, s):
    """A section's window as a thin image slab in the slab's physical frame."""
    sp = np.array(d["spacing"])
    sl = [slice(None)] * 3
    sl[s["axis"]] = slice(s["index"], s["index"] + 1)
    sl[0] = slice(*s["rows"])
    other = 2 if s["axis"] == 1 else 1
    sl[other] = slice(*s["cols"])
    lo = np.array([x.start for x in sl], float)
    return pv_image(d["img"][tuple(sl)], sp, lo * sp)


def render_context(d, method, secs):
    """The refined surface over the region, with the sections in place and numbered."""
    import pyvista as pv

    sp = np.array(d["spacing"])
    idx = np.argwhere(d["region"])
    lo, hi = idx.min(0) * sp - 2.0, idx.max(0) * sp + 2.0
    mesh = pv_mesh(*d["runs"][method]["surface"]).clip_box([lo[2], hi[2], lo[1], hi[1], lo[0], hi[0]], invert=False)
    p = plotter((900, 700))
    p.add_mesh(mesh, color="#d9c7a3", smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.1, opacity=0.55)
    for r, s in enumerate(secs):
        quad = section_quad(d, s)
        p.add_mesh(quad, scalars="intensity", cmap="gray", clim=CLIM, show_scalar_bar=False, lighting=False)
        p.add_point_labels(pv.PolyData(np.array([quad.center])), [str(r + 1)], font_size=22, text_color="red",
                           shape_opacity=0.0, show_points=False, always_visible=True)
    p.view_vector((0.4, -1.0, 0.9), viewup=(0, 0, 1))
    p.camera.zoom(1.3)
    return screenshot(p)


def fig_context(d, secs, plt):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(8 * len(METHODS), 6.5), layout="constrained")
    for ax, method in zip(np.atleast_1d(axes), METHODS):
        ax.imshow(render_context(d, method, secs))
        ax.set_title(f"refined surface ({METHOD_LABELS[method]}) and the sections")
        ax.axis("off")
    save(fig, "hrpqct_periarticular_context.png")


METHOD_LABELS = {"regularized": "1D regularised"}


def draw(d):
    plt = style()
    secs = sections(d)
    for method in METHODS:
        fig_sections(d, method, secs, plt)
    fig_context(d, secs, plt)


def main():
    """``hrpqct_periarticular.py FOLDER NAME [--redraw]``"""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    inputs = dict(INPUTS, scan=args[1])
    saved = load_run("hrpqct_periarticular", __file__, inputs) if "--redraw" in sys.argv else None
    if saved is None:
        cbm.log_to(LOG, level="DEBUG")
        raw, (a, b), cort, trab, roi, sp = read_scan(*args)
        box = slab_box(cort, trab, roi, sp)
        img = raw[box].astype(np.float32) * np.float32(a) + np.float32(b)
        del raw
        cort, trab, roi = cort[box], trab[box], roi[box]
        ref = labels_of(cort, cort | trab)
        region = comparison_region(roi, ref > 0, sp)
        del cort, trab, roi
        print(f"slab {tuple(s.stop - s.start for s in box)} at {tuple(s.start for s in box)}, "
              f"region {int(region.sum())} voxels; log: {LOG}", flush=True)
        t0 = time.perf_counter()
        labels, tib, starts = start(img, ref > 0, sp)
        start_s = time.perf_counter() - t0
        start_bone = starts == tib
        del labels
        prov = provenance(__file__, inputs)
        runs = {m: run_method(m, img, starts, tib, sp, ref, region) for m in METHODS}
        write_json(RESULTS / "hrpqct_periarticular.json", prov, dict(
            spacing_mm=list(sp), slab_shape=[s.stop - s.start for s in box], slab_start=[s.start for s in box],
            region_voxels=int(region.sum()), start_s=round(start_s, 1),
            start_dice_bone=dice(start_bone & region, (ref > 0) & region),
            ref_ct_th_mm=mean_thickness(ref == 1, region, sp),
            methods={m: v["row"] for m, v in runs.items()}))
        d = for_figures(img, sp, ref, region, runs)
        save_run("hrpqct_periarticular", prov, d)
    else:
        d = saved["data"]
    draw(d)


if __name__ == "__main__":
    main()
