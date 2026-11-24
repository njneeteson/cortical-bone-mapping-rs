"""Validation of the volumetric model against the 1D methods on the site
phantoms (``cbm.site_phantom``): accuracy and precision of cortical
thickness and of the periosteal surface position.

Each site's geometry is rendered once without noise; the scans are that
image plus independent Gaussian noise of the site's level (which is exactly
how the phantoms add noise), so the fits of different realisations share
the surface and the nodes. Every method fits from the true periosteal
mask, with the same sampling range, ``rho_c="estimate"``, and its defaults
otherwise; at the clinical sites, where the cortex is thin against the PSF,
every method is also run with ``rho_c`` fixed at the phantom's true
cortical density (tissue and pores), to separate the error of the
estimated ``rho_c`` from that of the method. The regularised method uses the smoothing of our 2025 paper
(sigma 1.94 mm; 500 neighbours at HR-pQCT, 128 at clinical resolution, where
the nodes are about ten times sparser), and the volumetric model the same
operator for its density fields. Per node and method:

* the thickness error against the truth along the method's own normal, and
  the error of the periosteal position along it;
* over the realisations, the mean error (accuracy) and the standard
  deviation (precision).

Results are summarised per site and method, and binned by the true
thickness relative to the in-plane PSF (``t / sigma_xy``; the thin-cortex
regime is ``t < 2 sigma``). Writes ``docs/results/volumetric_validation.json``
(sites given as arguments are updated in an existing file).

    python docs/scripts/volumetric_validation.py [OUT.json] [SITE ...]
"""

import json
import os
import sys
from dataclasses import replace

os.environ.setdefault("RAYON_NUM_THREADS", "4")

import numpy as np

import cortical_bone_mapping_rs as cbm

SEEDS = (11, 12, 13, 14, 15)
BINS = (0.0, 1.0, 2.0, 4.0, np.inf)

# per site: the bones fitted (label, name), sampling and neighbourhoods
SITES = {
    "distal_radius": dict(bones=((None, "radius"),), outside=1.0, inside=2.5, sigma=1.9424, k=500,
                          extra=dict(normal_constraint="in_plane", normal_axis=0), tied=True, true_rho_c=False),
    "distal_tibia": dict(bones=((None, "tibia"),), outside=1.0, inside=2.5, sigma=1.9424, k=500,
                         extra=dict(normal_constraint="in_plane", normal_axis=0), tied=True, true_rho_c=False),
    "proximal_femur": dict(bones=((1, "femur"),), outside=2.5, inside=7.0, sigma=1.9424, k=128, extra={},
                           tied=False, true_rho_c=True),
    "knee": dict(bones=((1, "tibia"), (2, "femur")), outside=3.0, inside=6.0, sigma=1.9424, k=128, extra={},
                 tied=False, true_rho_c=True),
}


def methods(s):
    reg = dict(reg_neighbours=s["k"], reg_sigma=s["sigma"])
    vol = dict(method="volumetric", **reg)
    if s["tied"]:
        vol["vol_sigma_z"] = "tied"
    return {
        "local": dict(method="local"),
        "local, estimated PSF": dict(method="local", sigma="estimate_psf", psf_model="axial", psf_axis=0),
        "regularized": dict(method="regularized", **reg),
        "volumetric": vol,
    }


def truth(ph, r, label):
    """True (periosteal offset, thickness) along each node's line."""
    o = r.options
    e = ph.edges_along(r.points, r.normals, -o.outside, o.inside)
    xp, t = e[:, 0], e[:, 1] - e[:, 0]
    if e.shape[1] == 3 and label is not None:
        wrong = e[:, 2] != label
        xp[wrong], t[wrong] = np.nan, np.nan
    return xp, t


def fitted_offset(r):
    return np.einsum("ij,ij->i", r.periosteal_points - r.points, r.normals)


def summarise(err_t, err_p, t_true, sigma):
    """Accuracy and precision over nodes (rows) and realisations (columns)."""
    ok = np.isfinite(err_t).all(1) & np.isfinite(err_p).all(1) & np.isfinite(t_true)
    mt, sd_t = err_t[ok].mean(1), err_t[ok].std(1, ddof=1)
    mp, sd_p = err_p[ok].mean(1), err_p[ok].std(1, ddof=1)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")
    row = lambda sel: {
        "nodes": int(sel.sum()),
        "t bias": float(mt[sel].mean()) if sel.any() else float("nan"),
        "t rmse": rms(err_t[ok][sel]),
        "t precision (rms sd)": rms(sd_t[sel]),
        "periosteum bias": float(mp[sel].mean()) if sel.any() else float("nan"),
        "periosteum rmse": rms(err_p[ok][sel]),
        "periosteum precision (rms sd)": rms(sd_p[sel]),
    }
    ratio = t_true[ok] / sigma
    out = {"all": row(np.ones(ok.sum(), bool))}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        out[f"t/sigma {lo:g}-{hi:g}"] = row((ratio >= lo) & (ratio < hi))
    return out


def true_rho_c(ph):
    """The phantom's mean cortical density: tissue and pores."""
    return (1.0 - ph.porosity) * ph.rho_bone + ph.porosity * ph.rho_marrow


def site(name):
    s = SITES[name]
    ph = cbm.site_phantom(name, seed=1)
    base, peri, _ = replace(ph, noise=0.0).generate()
    sp = tuple(ph.spacing)
    sigma_xy = ph.blur[1]
    rho_cs = [("estimate", "")] + ([(true_rho_c(ph), ", true rho_c")] if s["true_rho_c"] else [])
    out = {}
    runs = [(b, r, mk) for b in s["bones"] for r in rho_cs for mk in methods(s).items()]
    for (label, bone), (rho_c, tag), (m, kw) in runs:
        mask = peri if label is None else peri == label
        m = m + tag
        errs_t, errs_p, t_true, psf = [], [], None, []
        for seed in SEEDS:
            img = base + np.float32(ph.noise) * np.random.default_rng(seed).standard_normal(base.shape, np.float32)
            r = cbm.cortical_bone_mapping(img, mask, sp, outside=s["outside"], inside=s["inside"],
                                          rho_c=rho_c, progress_every=0, **s["extra"], **kw)
            xp, t = truth(ph, r, label)
            errs_t.append(r.thickness - t)
            errs_p.append(fitted_offset(r) - xp)
            t_true = t
            if r.volumetric is not None:
                psf.append((r.volumetric["sigma_xy"], r.volumetric["sigma_z"]))
        res = summarise(np.stack(errs_t, 1), np.stack(errs_p, 1), t_true, sigma_xy)
        if psf:
            res["psf"] = np.mean(psf, 0).tolist()
        out[f"{bone}: {m}"] = res
        a = res["all"]
        print(f"{name:15} {bone:7} {m:36} t bias {a['t bias']:+.4f} rmse {a['t rmse']:.4f} "
              f"sd {a['t precision (rms sd)']:.4f} | edge bias {a['periosteum bias']:+.4f} "
              f"rmse {a['periosteum rmse']:.4f} sd {a['periosteum precision (rms sd)']:.4f}"
              + (f" | psf {res['psf']}" if psf else ""), flush=True)
    return out


def main():
    names = sys.argv[2:] or list(SITES)
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/results/volumetric_validation.json"
    results = {"phantom psf": {}, "seeds": list(SEEDS), "sites": {}}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    for name in names:
        ph = cbm.site_phantom(name, seed=1)
        results["phantom psf"][name] = list(ph.blur)
        results["sites"][name] = site(name)
        with open(path, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
