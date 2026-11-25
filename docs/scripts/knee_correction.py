"""Correcting a segmentation by hand and refining it again.

The volumetric refinement of the clinical knee (``knee.py``) was exported as a
NIfTI label map (``knee.py DICOM_DIR --redraw --export DIR``), corrected by
hand in ITK-SNAP, and saved. This script refines the corrected bones again
from the corrected labels, with the scan's cortical density and PSF held at
the values the knee run estimated (``docs/results/knee.json``) and every
other bone excluded.

Writes docs/results/knee_correction.json and, per corrected bone, the
docs/images/knee_correction_* figure and GIF.

    python docs/scripts/knee_correction.py DICOM_DIR AUTOMATIC.nii.gz CORRECTED.nii.gz [--redraw]
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm
import knee
from _common import (BONE_COLOURS, RESULTS, START, draw_segments, load_run, provenance, save, save_run,
                     section_segments, style, write_json, IMAGES)

METHOD = "volumetric"
REFINE = dict(refine_iterations=150)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def scanner(method):
    """The scan's parameters from the knee run: its shared rho_c and PSF."""
    row = next(iter(json.loads((RESULTS / "knee.json").read_text())["methods"][method]["bones"].values()))
    out = dict(rho_c=row["rho_c"])
    if "psf_mm" in row:
        out.update(vol_sigma_xy=row["psf_mm"][0], vol_sigma_z=row["psf_mm"][1])
    return out


def automatic_surfaces(dicom):
    """The automatic refined surfaces of the knee run (``knee.py``), which
    must be current."""
    saved = load_run("knee", knee.__file__, dict(knee.INPUTS, dicom=Path(dicom).name))
    if saved is None:
        raise SystemExit("run knee.py first: its saved results are missing or stale")
    return {b: v["peri"] for b, v in saved["data"]["runs"][METHOD]["bones"].items()}


def compute(dicom, automatic, corrected):
    from scipy.spatial import cKDTree

    before = automatic_surfaces(dicom)
    img, sp = knee.read_dicom(dicom)
    auto, a_aff = knee.read_labels(automatic)
    corr, c_aff = knee.read_labels(corrected)
    affine = knee.dicom_affine(dicom)
    if not (np.allclose(a_aff, affine, atol=1e-3) and np.allclose(c_aff, affine, atol=1e-3)):
        raise SystemExit("the label maps are not on the DICOM's grid")
    if auto.shape != img.shape or corr.shape != img.shape:
        raise SystemExit(f"shapes differ: image {img.shape}, labels {auto.shape}, {corr.shape}")
    names = {i: n for i, n in enumerate(knee.LABEL_ORDER, start=1)}
    edited = [i for i in names if ((auto == i) != (corr == i)).any()]
    print("edited bones:", [names[i] for i in edited], flush=True)
    options = cbm.Options(spacing=sp, **dict(knee.BASE, **scanner(METHOD)), **knee.METHOD_OPTIONS[METHOD])
    t0 = time.perf_counter()
    res = cbm.refine_bones(img, corr.astype(np.uint16), sp, bones=edited, options=options, names=names,
                           keep_history=True, shared=False, **REFINE)
    sec = time.perf_counter() - t0
    v = float(np.prod(sp)) / 1000
    rows, bones = {}, {}
    for i in edited:
        rr = res[i]
        painted = (corr == i) & (auto != i)
        # where the refined surface lies against the automatic one: the distance
        # outside the automatic mask of its vertices over the painted region
        outside = ndimage.distance_transform_edt(auto != i, sampling=sp)
        vert = rr.result.surface_vertices
        vox = np.clip(np.round(vert / np.array(sp)).astype(int), 0, np.array(img.shape) - 1)
        near = ndimage.distance_transform_edt(~painted, sampling=sp)[tuple(vox.T)] <= 1.0
        moved = outside[tuple(vox.T)][near]
        rows[names[i]] = dict(
            painted_cm3=float(painted.sum() * v), erased_cm3=float(((auto == i) & (corr != i)).sum() * v),
            moves=len(rr.history) - 1, converged=bool(rr.converged), time_s=sec,
            painted_kept_cm3=float((painted & rr.periosteal_mask).sum() * v),
            surface_nodes_in_edit=int(near.sum()),
            outside_automatic_median_mm=float(np.median(moved)), outside_automatic_p95_mm=float(np.percentile(moved, 95)),
            thickness_median_mm=float(np.median(rr.result.thickness)))
        # how far the refined surface is from the automatic one, by distance from the edit
        dist = cKDTree(before[names[i]][0]).query(vert)[0]
        from_edit = ndimage.distance_transform_edt(~painted, sampling=sp)[tuple(vox.T)]
        for lo, hi in ((0.0, 1.0), (1.0, 5.0), (5.0, np.inf)):
            sel = (from_edit >= lo) & (from_edit < hi)
            rows[names[i]][f"change_{lo:g}_to_{hi:g}_mm_from_edit"] = dict(
                nodes=int(sel.sum()), median_mm=float(np.median(dist[sel])),
                p95_mm=float(np.percentile(dist[sel], 95)), max_mm=float(dist[sel].max()))
        print(names[i], {k: round(x, 3) if isinstance(x, float) else x for k, x in rows[names[i]].items()},
              flush=True)
        bones[names[i]] = dict(peri=(rr.result.surface_vertices, rr.result.surface_faces),
                               history=list(zip(rr.surfaces, rr.surfaces_faces)), label=i)
    return dict(img=img, spacing=sp, auto=auto, corr=corr, bones=bones), rows


# ------------------------------------------------------------------ figures

PAINT = "#ff3b30"

def views(d, name):
    """Three sections through the middle of the edit (two axial, one sagittal)
    and a window around it."""
    i = d["bones"][name]["label"]
    idx = np.argwhere((d["corr"] == i) != (d["auto"] == i))
    lo, hi, c = idx.min(0), idx.max(0), idx.mean(0).round().astype(int)
    q = lambda f: int(round(lo[0] + f * (hi[0] - lo[0])))  # noqa: E731
    pad = 22
    return [("axial", 0, q(0.3), (c[1] - pad - 10, c[1] + pad + 10), (c[2] - pad - 10, c[2] + pad + 10)),
            ("axial", 0, q(0.7), (c[1] - pad - 10, c[1] + pad + 10), (c[2] - pad - 10, c[2] + pad + 10)),
            ("sagittal", 2, int(c[2]), (lo[0] - pad, hi[0] + pad), (c[1] - pad - 25, c[1] + pad + 25))]


def section(ax, d, view, layers):
    """One section with mask contours and surfaces: ``layers`` is a list of
    ("mask", array, colour, style) or ("surface", (vertices, faces), colour, style)."""
    kind, ax_i, k, rows, cols = view
    sp = d["spacing"]
    other = [a for a in range(3) if a != ax_i]
    ax.imshow(np.take(d["img"], k, ax_i), cmap="gray", vmin=-150, vmax=1100, aspect=sp[other[0]] / sp[other[1]])
    for what, obj, colour, ls in layers:
        if what == "mask":
            ax.contour(np.take(obj, k, ax_i), [0.5], colors=colour, linewidths=2.0, linestyles=ls)
        elif what == "fill":
            from matplotlib.colors import ListedColormap

            m = np.take(obj, k, ax_i)
            ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([PAINT]), alpha=0.5, vmin=0, vmax=1,
                      aspect=sp[other[0]] / sp[other[1]])
        else:
            draw_segments(ax, section_segments(*obj, ax_i, k, sp), colour, lw=2.4, ls=ls)
    ax.set_xlim(*cols)
    if other[0] == 0:
        ax.set_ylim(*rows)  # head up
    else:
        ax.set_ylim(rows[1], rows[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{kind} {k}", fontsize=12)


def fig_steps(d, plt, name):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    i = d["bones"][name]["label"]
    auto, corr = d["auto"] == i, d["corr"] == i
    col = BONE_COLOURS[name.split()[-1]]
    vs = views(d, name)
    steps = [("1. automatic (volumetric refinement)", [("mask", auto, col, "-")]),
             ("2. corrected by hand (painted in red)", [("fill", corr & ~auto, None, None),
                                                        ("mask", corr, START, "--")]),
             ("3. refined from the correction", [("mask", corr, START, "--"),
                                                 ("surface", d["bones"][name]["peri"], col, "-"),
                                                 ("mask", auto, "white", ":")])]
    fig, axes = plt.subplots(len(steps), len(vs), figsize=(4.6 * len(vs), 4.4 * len(steps)), layout="constrained")
    for r, (title, layers) in enumerate(steps):
        for c, v in enumerate(vs):
            section(axes[r, c], d, v, layers)
        axes[r, 0].set_ylabel(title, fontsize=13)
    h = [Line2D([], [], color=col, lw=2.5, label=f"{name}, refined surface"),
         Patch(color=PAINT, alpha=0.5, label="painted by hand"),
         Line2D([], [], color=START, ls="--", lw=2.5, label="corrected mask (the new start)"),
         Line2D([], [], color="white", ls=":", lw=2.5, label="automatic result, for comparison")]
    leg = fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=True)
    leg.get_frame().set_facecolor("0.85")
    save(fig, f"knee_correction_{name.replace(' ', '_')}.png")


def fig_gif(d, plt, name):
    from PIL import Image

    i = d["bones"][name]["label"]
    corr, auto = d["corr"] == i, d["auto"] == i
    col = BONE_COLOURS[name.split()[-1]]
    hist = d["bones"][name]["history"]
    vs = views(d, name)
    frames = []
    for k, surf in enumerate(hist):
        fig, axes = plt.subplots(1, len(vs), figsize=(4.6 * len(vs), 5.2), layout="constrained")
        for ax, v in zip(axes, vs):
            section(ax, d, v, [("mask", corr, START, "--"), ("mask", auto, "white", ":"), ("surface", surf, col, "-")])
        status = "the corrected mask" if k == 0 else f"move {k}" + (", converged" if k == len(hist) - 1 else "")
        fig.suptitle(f"{name}, refinement from the hand correction: {status}", fontsize=15)
        from matplotlib.lines import Line2D

        h = [Line2D([], [], color=col, lw=2.5, label="current surface"),
             Line2D([], [], color=START, ls="--", lw=2.5, label="corrected mask (the start)"),
             Line2D([], [], color="white", ls=":", lw=2.5, label="automatic result")]
        leg = fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=True)
        leg.get_frame().set_facecolor("0.85")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    durations = [1800] + [900] * (len(frames) - 2) + [3500]
    out = IMAGES / f"knee_correction_{name.replace(' ', '_')}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print("wrote", out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        raise SystemExit(__doc__)
    dicom, automatic, corrected = args
    knee_prov = json.loads((RESULTS / "knee.json").read_text())["provenance"]["source_hash"]
    inputs = dict(dicom=Path(dicom).name, automatic=sha256(automatic), corrected=sha256(corrected),
                  knee_results=knee_prov, method=METHOD, refine=REFINE)
    saved = load_run("knee_correction", __file__, inputs) if "--redraw" in sys.argv else None
    if saved is None:
        prov = provenance(__file__, inputs)
        d, rows = compute(dicom, automatic, corrected)
        write_json(RESULTS / "knee_correction.json", prov, dict(bones=rows))
        save_run("knee_correction", prov, d)
    else:
        d = saved["data"]
    plt = style()
    for name in d["bones"]:
        fig_steps(d, plt, name)
        fig_gif(d, plt, name)


if __name__ == "__main__":
    main()
