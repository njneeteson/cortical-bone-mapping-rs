"""Compare the Rust solvers with NumPy/SciPy implementations of the same
methods, on the same sampled profiles.

* local:       ``scipy.optimize.least_squares`` (bounded TRF) per node.
* regularized: the global regularised problem of Neeteson & Boyd solved with
  ``scipy.optimize.minimize(method="L-BFGS-B")`` and an analytic gradient,
  with the sparse ``omega`` matrix from ``scipy.spatial.cKDTree``, i.e. the
  formulation of the paper's reference implementation.

Reports wall time, peak Python-heap memory (tracemalloc, which includes NumPy
buffers) for SciPy and the crate's own peak (from the Rust benchmark,
separately), final cost and thickness error against the phantom truth.

    python examples/compare_scipy.py            # prints the rows

``examples/benchmarks.py`` runs it with the other benchmarks.
"""

import sys
import time
import tracemalloc

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.special import erf

import cortical_bone_mapping_rs as cbm

RHO_C = 1000.0
LOWER = np.array([-3.0, 0.0607, 0.01, -200.0, -200.0])
UPPER = np.array([6.0, 12.0, 2.0, 400.0, 400.0])


def phantom(slices):
    ph = cbm.Phantom(dims=(slices, 300, 300), spacing=0.0607, radius=8.0, thickness_mean=1.2,
                     thickness_amplitude=0.5, blur=0.1, noise=60.0, supersample=1)
    img, mask = ph.generate()
    return ph, img, mask


OPTS = dict(outside=1.0, inside=3.0, normal_constraint="in_plane", keep_profiles=True)


def model_and_grad(x, m, t, s, rs, rb):
    """f and df/d(m, t, s, rs, rb) for (N,1) params and (M,) x."""
    u1 = (x - m + t / 2) / s
    u2 = (x - m - t / 2) / s
    e1, e2 = erf(u1 / np.sqrt(2)), erf(u2 / np.sqrt(2))
    p1, p2 = np.exp(-0.5 * u1**2) / np.sqrt(2 * np.pi), np.exp(-0.5 * u2**2) / np.sqrt(2 * np.pi)
    a1, a2 = RHO_C - rs, rb - RHO_C
    f = rs + 0.5 * a1 * (1 + e1) + 0.5 * a2 * (1 + e2)
    g1, g2 = a1 * p1 / s, a2 * p2 / s
    return f, (-g1 - g2, 0.5 * (g1 - g2), -(g1 * u1 + g2 * u2), 0.5 * (1 - e1), 0.5 * (1 + e2))


def scipy_local(profiles, x, w, init, n_max):
    sw = np.sqrt(w / w.sum())
    xs = np.array([0.1, 0.1, 0.05, 50.0, 50.0])
    out = []
    for y in profiles[:n_max]:
        def res(p):
            return sw * (model_and_grad(x, *p)[0] - y)

        def jac(p):
            _, g = model_and_grad(x, *p)
            return np.stack(g, -1) * sw[:, None] if np.ndim(g[3]) else np.column_stack(
                [g[0], g[1], g[2], np.broadcast_to(g[3], x.shape), np.broadcast_to(g[4], x.shape)]) * sw[:, None]

        r = least_squares(res, np.clip(init, LOWER, UPPER), jac=jac, bounds=(LOWER, UPPER), x_scale=xs,
                          ftol=1e-8, xtol=1e-10, gtol=1e-8, max_nfev=100)
        out.append(r.x)
    return np.array(out)


def scipy_regularized(points, profiles, x, w, gp, lam, sigma_reg, k):
    """Neeteson & Boyd regularised method with L-BFGS-B (per-node m,t,sigma; global rho_s, rho_b)."""
    n, M = profiles.shape
    tree = cKDTree(points)
    d, idx = tree.query(points, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    g = np.exp(-(d**2 - d[:, :1] ** 2) / (2 * sigma_reg**2))
    g /= g.sum(1, keepdims=True)
    rows = np.repeat(np.arange(n), k)
    W = sparse.eye(n, format="csr") - sparse.csr_matrix((g.ravel(), (rows, idx.ravel())), shape=(n, n))
    WtW = (W.T @ W).tocsr()
    t0, s0 = gp[1], gp[2]
    coef = np.array([lam * RHO_C**2 / (n * t0**2)] * 2 + [lam * RHO_C**2 / (n * s0**2)])
    ww = (w / (n * M))[None, :]
    y = profiles.astype(np.float64)
    # scale variables to O(1) for L-BFGS-B
    scale = np.array([0.1, 0.1, 0.05])
    gscale = np.array([50.0, 50.0])

    def unpack(z):
        f = z[: 3 * n].reshape(3, n) * scale[:, None]
        return f[0], f[1], f[2], z[3 * n] * gscale[0], z[3 * n + 1] * gscale[1]

    def fun(z):
        m, t, s, rs, rb = unpack(z)
        f, grads = model_and_grad(x[None, :], m[:, None], t[:, None], s[:, None], rs, rb)
        r = f - y
        cost = 0.5 * (ww * r * r).sum()
        gr = [(ww * r * gi).sum(1) for gi in grads[:3]]
        out = np.empty_like(z)
        for c, (v, gc) in enumerate(zip((m, t, s), gr)):
            u = W @ v
            cost += 0.5 * coef[c] * (u @ u)
            out[c * n:(c + 1) * n] = (gc + coef[c] * (WtW @ v)) * scale[c]
        out[3 * n] = (ww * r * grads[3]).sum() * gscale[0]
        out[3 * n + 1] = (ww * r * grads[4]).sum() * gscale[1]
        return cost, out

    z0 = np.concatenate([np.full(n, gp[0]) / scale[0], np.full(n, gp[1]) / scale[1],
                         np.full(n, gp[2]) / scale[2], [gp[3] / gscale[0], gp[4] / gscale[1]]])
    lo = np.concatenate([np.full(n, LOWER[c] / scale[c]) for c in range(3)] + [LOWER[3:] / gscale])
    hi = np.concatenate([np.full(n, UPPER[c] / scale[c]) for c in range(3)] + [UPPER[3:] / gscale])
    r = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options=dict(maxiter=400, ftol=1e-10, gtol=1e-8))
    m, t, s, rs, rb = unpack(r.x)
    return t, r.fun, r.nit


def main():
    """The comparison rows, one per method and phantom size."""
    rows = []
    for slices in [8, 32]:
        ph, img, mask = phantom(slices)
        # Rust: local
        t = time.perf_counter()
        loc = cbm.cortical_bone_mapping(img, mask, ph.spacing, method="local", **OPTS)
        t_rust_local = time.perf_counter() - t
        x, prof, n = loc.profile_x, loc.profiles, loc.n
        w = np.maximum(10 + (1 - 10) * np.abs(x) / np.abs(x).max(), 0)
        truth = ph.true_thickness(loc.points)
        # SciPy: local on a subset, extrapolated
        n_sub = min(n, 1000)
        t = time.perf_counter()
        sp = scipy_local(prof, x, w, np.array(loc.global_params), n_sub)
        t_sp_local = (time.perf_counter() - t) * n / n_sub
        agree = np.abs(sp[:, 1] - loc.thickness[:n_sub])
        rows.append(dict(method="local", nodes=n, rust_time=t_rust_local, scipy_time=t_sp_local,
                         scipy_time_extrapolated_from=n_sub,
                         rust_rmse=float(np.sqrt(((loc.thickness - truth) ** 2).mean())),
                         scipy_rmse=float(np.sqrt(((sp[:, 1] - truth[:n_sub]) ** 2).mean())),
                         median_abs_difference=float(np.median(agree))))
        print(rows[-1], flush=True)
        # Rust vs SciPy: regularized (k=64 so the SciPy matrix fits in memory)
        k = 64
        t = time.perf_counter()
        reg = cbm.cortical_bone_mapping(img, mask, ph.spacing, method="regularized", reg_neighbours=k, **OPTS)
        t_rust_reg = time.perf_counter() - t
        tracemalloc.start()
        t = time.perf_counter()
        t_sp, cost_sp, nit = scipy_regularized(reg.points, prof, x, w, reg.global_params, 1.0, 1.9424, k)
        t_sp_reg = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rows.append(dict(method=f"regularized k={k}", nodes=n, rust_time=t_rust_reg, scipy_time=t_sp_reg,
                         scipy_peak_bytes=peak, scipy_iterations=nit,
                         rust_cost=reg.info["cost"], scipy_cost=float(cost_sp),
                         rust_rmse=float(np.sqrt(((reg.thickness - truth) ** 2).mean())),
                         scipy_rmse=float(np.sqrt(((t_sp - truth) ** 2).mean()))))
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    main()
