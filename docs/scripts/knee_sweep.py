"""Compare whole-knee runs of ct_example.py (docs/clinical_ct.md): the
cortical density sweep and the half-width inside limit.

    python docs/scripts/ct_example.py DICOM ... --cache runs/base.pkl
    python docs/scripts/ct_example.py DICOM ... --half-width --cache runs/half_width.pkl
    python docs/scripts/ct_example.py DICOM ... --half-width --rho-c 1000 --cache runs/half_width_rc1000.pkl
    ...
    python docs/scripts/knee_sweep.py runs/base.pkl runs/half_width.pkl ... [--figures docs/images/knee_sweep]

Per run and bone: convergence, moves, time, thickness, the depth of the
refined surface below the bone threshold's isosurface (along each node's
normal, interpolated), the fitted edge against the steepest rise of each
profile, and the bone (above the hysteresis threshold, connected to a bone)
left more than 1 mm outside the refined surface. Prints Markdown tables and
draws ``PREFIX_rho_c.png`` and ``PREFIX_sections.png`` (blanked beyond 10 mm
from the starts; display only).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm

COLOURS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}


def label(d):
    rc = d.get("rho_c_option", "estimate")
    parts = ["ρc " + ("estimated" if rc == "estimate" else f"{float(rc):g} HU")]
    parts.append("half-width" if d.get("half_width") else "full inside")
    return ", ".join(parts)


def node_measures(d, rr):
    """Depth below the threshold isosurface and edge − steepest rise, per node."""
    img, sp, org = d["img"].astype(np.float64), np.array(d["spacing"]), np.array(d["origin"])
    r = rr.result
    x = np.arange(-4.0, 4.0 + 1e-9, 0.125)
    pts = r.points[:, None, :] + x[None, :, None] * r.normals[:, None, :]
    prof = ndimage.map_coordinates(img, ((pts.reshape(-1, 3) - org) / sp).T, order=1).reshape(len(r.points), -1)
    thr = d["threshold"]
    above = prof >= thr
    i0 = int(np.argmin(np.abs(x)))
    depth = np.full(len(prof), np.nan)
    for k in range(len(prof)):
        idx = np.nonzero(above[k, 1:] != above[k, :-1])[0]
        if len(idx) == 0:
            continue
        j = idx[np.argmin(np.abs(idx + 0.5 - i0))]
        y0, y1 = prof[k, j], prof[k, j + 1]
        depth[k] = -(x[j] + (thr - y0) / (y1 - y0) * (x[j + 1] - x[j]))
    g = np.gradient(ndimage.gaussian_filter1d(prof, 1.0, axis=1), x, axis=1)
    jr = np.argmax(g, axis=1)
    ok = g[np.arange(len(jr)), jr] > 150.0
    edge = r.center - r.thickness / 2
    return depth, np.where(ok, edge - x[jr], np.nan)


def missed(d, rr, b, ref):
    sp = np.array(d["spacing"])
    out = ndimage.distance_transform_edt(~rr.periosteal_mask, sampling=sp) > 1.0
    return float(((ref == b) & out).sum() * np.prod(sp)) / 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="caches written by ct_example.py --cache")
    ap.add_argument("--figures", help="figure file prefix")
    a = ap.parse_args()
    runs = {Path(p).stem: pickle.load(open(p, "rb")) for p in a.runs}
    d0 = next(iter(runs.values()))
    low = d0.get("low_threshold") or 146.0
    ref = cbm.grow_bones(d0["img"], d0.get("seeds", d0["labels"]), low)
    names = d0["names"]
    rows, summary = [], {}
    for n, d in runs.items():
        for b, rr in sorted(d["results"].items()):
            depth, er = node_measures(d, rr)
            t = rr.result.thickness
            total = dict(rr.timings).get("total", float("nan"))
            s = dict(conv=rr.converged, moves=len(rr.history) - 1, time=total, rho_c=rr.result.rho_c,
                     t50=np.median(t), t5=np.percentile(t, 5), t95=np.percentile(t, 95),
                     depth=np.nanmedian(depth), er=np.nanmedian(er), missed=missed(d, rr, b, ref))
            summary[(n, b)] = s
            rows.append(f"| {label(d)} | {names.get(b, b)} | {'yes' if s['conv'] else '**no**'} | {s['moves']} "
                        f"| {s['time']:.0f} s | {s['rho_c']:.0f} | {s['t50']:.2f} ({s['t5']:.2f}–{s['t95']:.2f}) "
                        f"| {s['depth']:+.2f} | {s['er']:+.2f} | {s['missed']:.2f} |")
    print("| run | bone | converged | moves | time | ρc (HU) | thickness, median (5–95%) (mm) "
          f"| depth below {d0['threshold']:.0f} HU surface (mm) | edge − steepest rise (mm) | missed bone (cm³) |")
    print("|---|---|---|---:|---:|---:|---|---:|---:|---:|")
    print("\n".join(rows))
    if a.figures:
        figures(runs, summary, names, a.figures)


def figures(runs, summary, names, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # thickness and depth against rho_c (half-width runs with a fixed rho_c, plus the estimate)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for b in sorted(names):
        pts = sorted((float(d["rho_c_option"]), summary[(n, b)]) for n, d in runs.items()
                     if d.get("half_width") and d.get("rho_c_option", "estimate") != "estimate")
        est = [summary[(n, b)] for n, d in runs.items() if d.get("half_width") and d.get("rho_c_option", "estimate") == "estimate"]
        if not pts:
            continue
        rc = [p[0] for p in pts]
        for ax, key in zip(axes, ("t50", "depth")):
            ax.plot(rc, [p[1][key] for p in pts], "o-", color=COLOURS[b], label=names[b])
            for e in est:
                ax.plot([e["rho_c"]], [e[key]], "*", ms=12, color=COLOURS[b])
    axes[0].set_ylabel("median thickness (mm)")
    axes[1].set_ylabel("depth below the bone-threshold surface (mm)")
    for ax in axes:
        ax.set_xlabel("ρc (HU)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cortical density sweep (half-width inside limit); stars: ρc estimated per bone", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}_rho_c.png", dpi=90)
    plt.close(fig)

    # sections at the spots that needed attention
    d0 = next(iter(runs.values()))
    img, sp, org = d0["img"], np.array(d0["spacing"]), np.array(d0["origin"])
    starts = d0.get("starts", d0["hulls"])
    shown = np.where(ndimage.distance_transform_edt(starts == 0, sampling=sp) <= 10.0, img, -1000)

    def cut(v, f, axis, k):
        q = (v - org) / sp
        t = q[f]
        dv = t[:, :, axis] - k
        segs = []
        for tri, dd in zip(t, dv):
            p = [tri[i] + dd[i] / (dd[i] - dd[j]) * (tri[j] - tri[i]) for i, j in ((0, 1), (1, 2), (2, 0))
                 if (dd[i] < 0) != (dd[j] < 0)]
            if len(p) == 2:
                segs.append(p)
        o = [i for i in range(3) if i != axis]
        return np.array(segs)[:, :, o][:, :, ::-1] if segs else np.zeros((0, 2, 2))

    views = [(0, 210, (60, 150), (200, 140), "axial 210: patella"),
             (2, 105, (190, 300), (85, 160), "sagittal 105: anterior tibia"),
             (1, 255, (70, 160), (115, 160), "coronal 255: tibial spines")]
    fig, axes = plt.subplots(len(runs), len(views), figsize=(5.2 * len(views), 3.9 * len(runs)), squeeze=False)
    for row, (n, d) in zip(axes, runs.items()):
        for ax, (axis, k, xl, yl, title) in zip(row, views):
            ax.imshow(np.take(shown, k, axis=axis), cmap="gray", vmin=-200, vmax=1400,
                      origin="upper" if axis == 0 else "lower")
            for b, rr in d["results"].items():
                seg = cut(rr.result.surface_vertices, rr.result.surface_faces, axis, k)
                if len(seg):
                    ax.add_collection(LineCollection(seg, colors=COLOURS[b], linewidths=0.9))
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label(d)}: {title}", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_sections.png", dpi=70)
    plt.close(fig)
    print(f"wrote {prefix}_rho_c.png, {prefix}_sections.png")


if __name__ == "__main__":
    main()
