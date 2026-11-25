"""Microarchitecture of the HR-pQCT site phantoms, and Ct.Th by local
thickness against Ct.Th from the cortical bone mapping thickness fields.

For the distal radius and tibia phantoms (``cbm.site_phantom``):

* **truth**: the phantom's exact structure at the voxel centres (rendered
  without blur or noise), in its true compartments;
* **image**: the rendered scan segmented as in the second-generation
  HR-pQCT standard analysis (Whittier et al. 2020): Gaussian filter
  (sigma 0.8 voxels, support 1 voxel), trabecular bone above 320 and
  cortical pores below 450 mg HA/cm3, within the true compartments (the
  dual-threshold compartment extraction is not repeated here).

Tb.Th and Tb.Sp are the mean local thickness of the trabecular bone and of
the marrow space within the trabecular compartment, Ct.Th the mean local
thickness of the cortical compartment (Hildebrand & Rueegsegger 1997, with
local-thickness-rs); BV/TV and Ct.Po are volume fractions. Ct.Th is then
compared with the mean CBM thickness of the local, regularised and
volumetric fits on the same scan (from the true periosteal mask), and with
the true thickness along CBM's normals.

Run from the repository root with the package and ``localthickness_rs``
installed; writes ``docs/results/site_microarchitecture.json``.
"""

import json
import os
import sys
from dataclasses import replace

os.environ.setdefault("RAYON_NUM_THREADS", "4")

import numpy as np
from scipy import ndimage

import cortical_bone_mapping_rs as cbm
import localthickness_rs as lt

THREADS = int(os.environ["RAYON_NUM_THREADS"])
MEMORY = 1 << 30
SITES = ("distal_radius", "distal_tibia")
SEEDS = (1, 2, 3)
CBM = dict(outside=1.0, inside=2.5, rho_c="estimate", normal_constraint="in_plane", normal_axis=0,
           progress_every=0)
METHODS = {
    "local": dict(method="local"),
    "regularized": dict(method="regularized", reg_neighbours=64, reg_sigma=0.5),
    "volumetric": dict(method="volumetric", vol_sigma_z="tied", reg_neighbours=64, reg_sigma=0.5),
}


def thickness(mask, spacing, roi=None):
    m = lt.local_thickness(mask, spacing, roi=roi, threads=THREADS, memory_limit=MEMORY)
    return float(m[mask].mean())


def separation(bone, spacing, roi):
    m = lt.local_separation(bone, spacing, roi=roi, threads=THREADS, memory_limit=MEMORY)
    return float(m[roi & ~bone].mean())


def structure(bone, pores, trab, cort, spacing):
    """BV/TV, Tb.Th, Tb.Sp, Ct.Th, Ct.Po of a segmentation."""
    tb = bone & trab
    return {
        "BV/TV": float(tb.sum() / trab.sum()),
        "Tb.Th": thickness(tb, spacing, roi=trab),
        "Tb.Sp": separation(tb, spacing, trab),
        "Ct.Th": thickness(cort, spacing),
        "Ct.Po": float((pores & cort).sum() / cort.sum()),
    }


def site(name, seed):
    ph = cbm.site_phantom(name, seed=seed)
    sp = tuple(ph.spacing)
    img, peri, endo = ph.generate()
    trab, cort = endo, peri & ~endo
    exact = replace(ph, noise=0.0, blur=(0.0, 0.0, 0.0), supersample=1).generate()[0]
    mid = 0.5 * (ph.rho_bone + ph.rho_marrow)
    out = {"truth": structure(exact > mid, exact < mid, trab, cort, sp)}
    filt = ndimage.gaussian_filter(img, 0.8, truncate=1.0 / 0.8)
    out["image"] = structure(filt > 320.0, filt < 450.0, trab, cort, sp)
    del exact, filt
    fits = {}
    for m, kw in METHODS.items():
        r = cbm.cortical_bone_mapping(img, peri, sp, **CBM, **kw)
        true_t = ph.true_thickness(r)
        ok = np.isfinite(r.thickness) & np.isfinite(true_t)
        fits[m] = {
            "Ct.Th": float(r.thickness[ok].mean()),
            "true along normals": float(true_t[ok].mean()),
            "rmse": float(np.sqrt(((r.thickness - true_t)[ok] ** 2).mean())),
            "nodes": int(ok.sum()),
        }
        print(f"  {name} seed {seed} {m}: {fits[m]}", flush=True)
    out["cbm"] = fits
    return out


def main():
    results = {}
    for name in SITES:
        results[name] = [site(name, s) for s in SEEDS]
        for k in ("truth", "image"):
            print(name, k, {q: round(float(np.mean([r[k][q] for r in results[name]])), 4)
                            for q in results[name][0][k]}, flush=True)
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/results/site_microarchitecture.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
