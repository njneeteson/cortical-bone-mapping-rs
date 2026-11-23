"""The volumetric model against the regularised 1D fit on the clinical knee
CT of ``clinical_ct.md`` (used with the subject's permission; the image is
not part of the repository).

    python docs/scripts/knee_volumetric.py CACHE [OUT.json]

``CACHE`` is the ``--cache`` file of ``ct_example.py``. The fits start from
each bone's refined periosteal surface, with the options of
``ct_example.py`` (the volumetric model takes the same sampling range,
bounds, density smoothing and ``rho_c`` rule):

* **regularised**: each bone on its own, the other bones excluded (a single
  fit: ``refine_iterations=0``), with its own estimated ``rho_c``;
* **volumetric, joint**: all bones in one fit (the union of the masks; each
  band voxel belongs to its nearest surface, and surface-distance
  neighbourhoods keep the bones apart), so they share one PSF, as they
  share the scanner, and one estimated ``rho_c``;
* **volumetric, matched**: each bone on its own like the regularised fit,
  with the PSF held at the joint fit's and ``rho_c`` at the regularised
  fit's, so that the difference is that of the methods alone.

With no truth, the methods are compared with each other: per node, the
difference of thickness and of the periosteal position; and the volumetric
model's PSF, noise and reliability outputs. Writes
``docs/knee_volumetric.json``.
"""

import json
import os
import pickle
import sys
from dataclasses import replace

os.environ.setdefault("RAYON_NUM_THREADS", "4")

import numpy as np
from scipy.spatial import cKDTree

import cortical_bone_mapping_rs as cbm

sys.path.insert(0, os.path.dirname(__file__))
from ct_example import OPTIONS  # noqa: E402

VOLUMETRIC = replace(OPTIONS, method="volumetric", vol_scanner_axis=0)


def fit(d, label, options):
    """One bone's fit on its refined surface, the other bones excluded."""
    img, labels = d["img"], d["labels"]
    mask = d["results"][label].periosteal_mask
    others = (labels > 0) & (labels != label)
    rr = cbm.refine_segmentation(img, mask, d["spacing"], exclude=others, options=options,
                                 refine=cbm.RefineOptions(refine_iterations=0))
    return rr.result


def joint_fit(d):
    """All bones in one volumetric fit."""
    union = np.zeros(d["img"].shape, bool)
    for label in d["names"]:
        union |= d["results"][label].periosteal_mask
    return cbm.cortical_bone_mapping(d["img"], union, d["spacing"], options=VOLUMETRIC)


def rho_c_of(r):
    return float(np.nanmedian(r.rho_c_node)) if r.rho_c_node is not None else float(r.rho_c)


def offset(r, idx=slice(None)):
    return np.einsum("ij,ij->i", r.periosteal_points[idx] - r.points[idx], r.normals[idx])


def q(v):
    return [float(x) for x in np.percentile(v, (5, 50, 95))]


def compare(reg, vol, idx):
    """Volumetric (nodes ``idx`` of ``vol``) against regularised, per node."""
    t_v, info = vol.thickness[idx], vol.volumetric
    dt = t_v - reg.thickness
    dp = offset(vol, idx) - offset(reg)
    ok = np.isfinite(dt) & np.isfinite(dp)
    se = info["std_error_t"][idx]
    return {
        "nodes": int(ok.sum()),
        "thickness (5/50/95%)": q(t_v[ok]),
        "thickness difference mean, sd": [float(dt[ok].mean()), float(dt[ok].std())],
        "periosteum difference mean, sd": [float(dp[ok].mean()), float(dp[ok].std())],
        "rho_c": rho_c_of(vol),
        "psf": [info["sigma_xy"], info["sigma_z"]],
        "noise sd": info["noise_sd"],
        "noise correlation": list(info["noise_correlation"]),
        "std error t (5/50/95%)": q(se[ok & np.isfinite(se)]),
        "data share m, t (median)": [float(np.nanmedian(info["data_share"][idx][ok, k])) for k in (0, 1)],
        "misfit (5/50/95%)": q(info["misfit"][idx][ok]),
    }


def main():
    d = pickle.loads(open(sys.argv[1], "rb").read())
    path = sys.argv[2] if len(sys.argv) > 2 else "docs/knee_volumetric.json"
    joint = joint_fit(d)
    psf = (joint.volumetric["sigma_xy"], joint.volumetric["sigma_z"])
    print("joint", psf, rho_c_of(joint), flush=True)
    tree = cKDTree(joint.points)
    out = {"joint psf": list(psf), "joint rho_c": rho_c_of(joint), "bones": {}}
    for label, name in d["names"].items():
        reg = fit(d, label, OPTIONS)
        dist, idx = tree.query(reg.points)
        assert (dist < 1e-6).all(), "the joint surface has the bone's nodes"
        matched = fit(d, label, replace(VOLUMETRIC, rho_c=rho_c_of(reg), vol_sigma_xy=psf[0], vol_sigma_z=psf[1]))
        out["bones"][name] = {
            "regularized": {"nodes": len(reg.points), "thickness (5/50/95%)": q(reg.thickness[np.isfinite(reg.thickness)]),
                            "rho_c": rho_c_of(reg)},
            "volumetric, joint": compare(reg, joint, idx),
            "volumetric, matched": compare(reg, matched, slice(None)),
        }
        print(name, json.dumps(out["bones"][name]), flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
