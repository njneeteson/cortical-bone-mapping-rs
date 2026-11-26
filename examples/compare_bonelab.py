"""Benchmark against the reference Python implementation in the Bonelab
package (``bonelab.util.cortical_thickness``, Neeteson & Boyd 2025).

Both implementations fit the *same* profiles: the profiles are sampled once
by this crate (``keep_profiles=True``) and handed to Bonelab's
``LocalTreeceMinimization``, ``GlobalControlPointTreeceMinimization`` and
``GlobalRegularizationTreeceMinimization`` classes. Bonelab's own surface
extraction and sampling (PyVista) are not part of the comparison; for this
crate both the fit-only time and the full-pipeline time are reported.

Bonelab is not a dependency; point the script at a checkout:

    git clone https://github.com/Bonelab/Bonelab /path/to/Bonelab
    pip install tqdm
    python examples/compare_bonelab.py /path/to/Bonelab   # prints the rows

``examples/benchmarks.py`` runs it with the other benchmarks.

Memory: for Bonelab, the peak of Python-heap allocations during the fit
(tracemalloc, which includes NumPy/SciPy buffers). Rust allocations are
invisible to tracemalloc; the Rust peak heap is measured with a counting
allocator by ``examples/benchmark.rs`` on the same phantoms.
"""

import importlib
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np

import cortical_bone_mapping_rs as cbm

# Table 1 (radius & tibia) of Neeteson & Boyd (2025), except the trabecular initial
# guess: Bonelab's CLI default of 200 (with 0, its L-BFGS-B initial global fit
# stalls at the bounds m = 6, sigma = 2 on these phantoms)
COMMON = dict(
    rho_c=1000.0, residual_boost_factor=10.0, t_initial_guess=1.0, rho_s_initial_guess=0.0,
    rho_b_initial_guess=200.0, sigma_initial_guess=0.1, x_bounds=(-3.0, 6.0), t_bounds=(0.0607, 12.0),
    rho_s_bounds=(-200.0, 400.0), rho_b_bounds=(-200.0, 400.0), sigma_bounds=(0.01, 2.0), silent=True,
    max_iterations=400, f_tol=1e-6, g_tol=1e-6,
)
RUST = dict(outside=1.0, inside=3.0, normal_constraint="in_plane", residual_boost=10.0,
            max_iterations=400, ftol=1e-8, gtol=1e-8)


def load_bonelab(root: Path):
    """Import bonelab.util.cortical_thickness without running bonelab/__init__
    (which needs the package to be pip-installed for its version string)."""
    pkg = types.ModuleType("bonelab")
    pkg.__path__ = [str(root / "bonelab")]
    sys.modules["bonelab"] = pkg
    util = types.ModuleType("bonelab.util")
    util.__path__ = [str(root / "bonelab" / "util")]
    sys.modules["bonelab.util"] = util
    ct = "bonelab.util.cortical_thickness."
    return (
        importlib.import_module(ct + "LocalTreeceMinimization").LocalTreeceMinimization,
        importlib.import_module(ct + "GlobalControlPointTreeceMinimization").GlobalControlPointTreeceMinimization,
        importlib.import_module(ct + "GlobalRegularizationTreeceMinimization").GlobalRegularizationTreeceMinimization,
    )


def phantom(slices):
    ph = cbm.Phantom(dims=(slices, 300, 300), spacing=0.0607, radius=8.0, thickness_mean=1.2,
                     thickness_amplitude=0.5, blur=0.1, noise=60.0, supersample=1)
    img, mask = ph.generate()
    return ph, img, mask


def timed(fn):
    tracemalloc.start()
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, peak


def rmse(t, truth):
    return float(np.sqrt(((np.asarray(t) - truth) ** 2).mean()))


def fit_time(res):
    tt = res.info["timings"]
    return sum(v for k, v in tt.items() if k not in ("surface", "sampling"))


def main(root):
    """The comparison rows, one per method and phantom size."""
    Local, ControlPoint, Regularization = load_bonelab(Path(root))
    rows = []
    for slices, local_subset in [(8, 400), (32, 400)]:
        ph, img, mask = phantom(slices)
        # sample once (local fit also gives the profiles)
        loc, t_loc, _ = timed(lambda: cbm.cortical_bone_mapping(img, mask, ph.spacing, method="local",
                                                               keep_profiles=True, **RUST))
        f_ij, x_j, pts, n = loc.profiles.astype(np.float64), loc.profile_x, loc.points, loc.n
        truth = ph.true_thickness(pts)
        print(f"--- {n} nodes, {len(x_j)} samples per profile", flush=True)

        # ---- local (Bonelab: per-node least_squares; extrapolated from a subset)
        sub = np.linspace(0, n - 1, local_subset).astype(int)
        bl = Local(f_ij=f_ij[sub], x_j=x_j, **COMMON)
        (m, t, *_), dt, peak = timed(bl.fit)
        rows.append(dict(method="local", nodes=n, bonelab_time=dt * n / local_subset,
                         bonelab_time_extrapolated_from=local_subset, bonelab_peak_bytes=peak,
                         bonelab_rmse=rmse(t, truth[sub]), rust_fit_time=fit_time(loc),
                         rust_total_time=t_loc, rust_rmse=rmse(loc.thickness, truth)))
        print(rows[-1], flush=True)

        # ---- multigrid / control points (spacings in mm = 128..16 voxels of 60.7 um)
        sp = [7.77, 3.885, 1.942, 0.971]
        bl = ControlPoint(points=pts, control_point_separations=sp, interpolation_neighbours=5,
                          use_rbf_spline=False, rbf_smooth=0.0, rbf_degree=1, f_ij=f_ij, x_j=x_j, **COMMON)
        (m, t, *_), dt, peak = timed(bl.fit)
        rr, t_rr, _ = timed(lambda: cbm.cortical_bone_mapping(img, mask, ph.spacing, method="multigrid",
                                                             multigrid_spacings=sp, **RUST))
        rows.append(dict(method="multigrid", nodes=n, bonelab_time=dt, bonelab_peak_bytes=peak,
                         bonelab_rmse=rmse(t, truth), rust_fit_time=fit_time(rr), rust_total_time=t_rr,
                         rust_rmse=rmse(rr.thickness, truth)))
        print(rows[-1], flush=True)

        # ---- regularized
        for k in (64, 500):
            if k == 500 and n > 10000:
                continue  # Bonelab's dense (n, k) intermediates get large; see docs
            bl = Regularization(points=pts, neighbours_regularization=k, sigma_regularization=1.9424,
                                lambda_regularization=1.0, f_ij=f_ij, x_j=x_j, **COMMON)
            (m, t, *_), dt, peak = timed(bl.fit)
            rr, t_rr, _ = timed(lambda: cbm.cortical_bone_mapping(
                img, mask, ph.spacing, method="regularized", reg_neighbours=k, reg_sigma=1.9424, **RUST))
            rows.append(dict(method=f"regularized k={k}", nodes=n, bonelab_time=dt, bonelab_peak_bytes=peak,
                             bonelab_rmse=rmse(t, truth), rust_fit_time=fit_time(rr), rust_total_time=t_rr,
                             rust_rmse=rmse(rr.thickness, truth)))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
