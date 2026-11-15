# Performance

All timings: Windows 11, 16 hardware threads, release build. Phantoms are
cylinders with an HR-pQCT-like geometry: 60.7 µm voxels, outer radius 8 mm,
cortex 1.2 ± 0.5 mm, PSF 0.1 mm, noise σ = 60, with 8, 32 and 128 slices.
Profiles run from 1 mm outside to 3 mm inside (67 samples), normals in the
transverse plane. RMSE is thickness error against the phantom truth.

![Benchmark](images/benchmark.png)

## This crate: time and peak memory

`cargo run --release --example benchmark -p cortical-bone-mapping`.

Times cover the full pipeline: surface extraction and smoothing, sampling,
global initialisation and the fit. Peak heap is the most memory allocated
*during* the call, above the input image and mask, measured with a counting
global allocator.

| method | nodes | time [s] | peak heap [MB] | bytes/node | RMSE [µm] | outer it. | CG it. |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 7392 | 0.31 | 19 | 2524 | 19.8 | 12 | 0 |
| multigrid | 7392 | 0.92 | 19 | 2598 | 8.9 | 32 | 425 |
| regularized k=64 | 7392 | 0.48 | 26 | 3570 | 15.2 | 7 | 50 |
| regularized k=500 | 7392 | 0.73 | 52 | 7058 | 13.9 | 7 | 51 |
| reg+curvature | 7392 | 1.38 | 33 | 4495 | 7.7 | 7 | 329 |
| local | 32736 | 0.58 | 42 | 1290 | 19.3 | 13 | 0 |
| multigrid | 32736 | 1.52 | 47 | 1447 | 11.8 | 30 | 443 |
| regularized k=64 | 32736 | 0.94 | 81 | 2469 | 15.3 | 7 | 53 |
| regularized k=500 | 32736 | 1.89 | 195 | 5957 | 14.2 | 7 | 50 |
| reg+curvature | 32736 | 2.66 | 113 | 3460 | 7.8 | 7 | 266 |
| local | 134112 | 1.74 | 117 | 872 | 19.1 | 13 | 0 |
| multigrid | 134112 | 4.32 | 138 | 1030 | 12.6 | 32 | 454 |
| regularized k=64 | 134112 | 3.16 | 284 | 2115 | 15.2 | 7 | 55 |
| regularized k=500 | 134112 | 7.36 | 752 | 5603 | 14.1 | 7 | 50 |
| reg+curvature | 134112 | 17.25 | 418 | 3118 | 7.6 | 7 | 271 |

(`reg+curvature`: `λ = 1`, k = 64, plus the default `"change"` curvature
penalty, scale 0.25 mm, `μ = 1`.)

Where the memory goes at 134k nodes:

* profiles: `N × 67 × 4 B` = 36 MB;
* Gauss–Newton blocks and work vectors: about 25 doubles per node;
* the smoothing operator `ω`: 8 B per neighbour, so 536 MB for k = 500 and
  69 MB for k = 64. It is the dominant cost at the paper's k = 500. Building
  it in place (fixed stride, then compaction) brought the k = 500 peak down
  from 1166 MB to 751 MB;
* transposed sparse products: 16 fixed buffers of N doubles (17 MB).

Design changes driven by profiling:

* **The curvature Laplacian.** With the `k` *nearest* neighbours, 64
  neighbours at 60 µm spacing span only ~0.27 mm, so the operator was stiff at
  the node scale (1447 CG iterations, 65 s at 134k nodes). Importance
  sampling `k` neighbours from the Gaussian at the requested scale gives the
  same accuracy with about a third of the CG work.
* **Inexact Newton.** Solving each Gauss–Newton step only as accurately as
  the current gradient warrants (forcing term `min(0.1, √g)`) halved the CG
  work (k = 500: 90 → 50 iterations; curvature: 440 → 271) and the time of
  the largest fits (10.0 → 7.4 s, 21.6 → 17.2 s) with identical RMSEs.
* **A work budget.** Every trial step counts against `max_iterations`, and
  CG iterations against `cg_max_total` (2000), so a badly posed problem (a
  segmentation that has leaked into the marrow) stops in bounded time, flagged,
  instead of running for hours.
* **Deterministic reductions.** All parallel sums use fixed chunks in a fixed
  order, so results are bitwise identical for any thread count. This costs
  nothing measurable.

## Against Bonelab (the original Python implementation)

`python docs/scripts/compare_bonelab.py /path/to/Bonelab`. Bonelab's
`LocalTreeceMinimization`, `GlobalControlPointTreeceMinimization` and
`GlobalRegularizationTreeceMinimization` fit **the same profiles**, sampled
once by this crate. Bonelab's PyVista surface and sampling stages are not
timed. "This crate fit" is the global initialisation plus the method fit.
Bonelab's local fit is timed on 400 nodes and extrapolated (marked *).
Bonelab's memory is its Python-heap peak during the fit (tracemalloc).
Options follow Table 1 of the paper (k = 500 only at the smaller size).

![Bonelab comparison](images/bonelab_comparison.png)

| method | nodes | Bonelab fit [s] | this crate fit [s] | speed-up | Bonelab heap [MB] | RMSE Bonelab [µm] | RMSE this crate [µm] |
|---|---:|---:|---:|---:|---:|---:|---:|
| local* | 7392 | 47.4 | 0.26 | 180× | 0.4 | 20.1 | 19.8 |
| multigrid | 7392 | 11.8 | 1.03 | 11× | 52 | 14.1 | 8.9 |
| regularized k=64 | 7392 | 4.2 | 0.54 | 8× | 62 | 18.6 | 15.2 |
| regularized k=500 | 7392 | 4.3 | 0.80 | 5× | 83 | 17.7 | 13.9 |
| local* | 32736 | 204.4 | 0.48 | 424× | 0.1 | 18.5 | 19.3 |
| multigrid | 32736 | 63.1 | 1.39 | 46× | 229 | 16.3 | 11.8 |
| regularized k=64 | 32736 | 10.0 | 0.83 | 12× | 275 | 18.7 | 15.3 |

For comparison, the whole Rust pipeline peaks at 81 MB for regularised k=64
at 32736 nodes, against Bonelab's 275 MB for the fit alone.

Notes:

* **Local.** Both find the same per-node optima: the RMSEs agree, and so does
  the SciPy check below, to a median of 4e-8 mm. The speed-up comes from
  analytic Jacobians, 5×5 normal equations instead of SciPy's TRF machinery
  with finite-difference Jacobians, compiled code, and all cores.
* **Global methods.** This crate is both faster and more accurate, because
  Bonelab's L-BFGS-B runs stop before convergence. At 7,392 nodes
  (regularised, k = 64), the thickness RMSE is:

  | Bonelab configuration | RMSE | time |
  |---|---|---|
  | default tolerances (1e-6) | 18.65 µm | 2.6 s |
  | tight tolerances (`f_tol = 1e-14`, `g_tol = 1e-12`) | 16.30 µm | 4.4 s |
  | tight tolerances + consistent loss | 15.70 µm | 10.7 s |
  | this crate | 15.18 µm | 0.47 s |

  The regularised loss and gradient also disagree: the loss scales the m/t
  penalty by `ρc²/t0` and the gradient by `ρc²/t0²`, and the σ penalty is
  missing from the loss. Fixing that only matters once the solve runs to
  convergence. The regularised speed-ups in the table above are therefore
  against an *unconverged* solve. Against converged solves the factor is
  ~10–20× (Bonelab, tight tolerances) and ~70× (SciPy L-BFGS-B on the
  identical objective, below).
* **Robustness.** With the trabecular initial guess at 0, Bonelab's initial
  one-model L-BFGS-B fit stalls on the bounds (`m = 6`, `σ = 2`, `t ≈ 11 mm`)
  on these phantoms, and every global method inherits that (RMSE ≈ 8–9 mm).
  The comparison therefore uses Bonelab's CLI default of 200. The
  Gauss–Newton initial fit in this crate converges from 0.

## Against SciPy on the same objective

`python docs/scripts/compare_scipy.py` implements this crate's objective in
NumPy/SciPy:

* local: `least_squares` (bounded TRF, analytic Jacobian, same `√γ`
  weighting);
* regularised: L-BFGS-B, analytic gradient, sparse `ω` from `cKDTree`, all
  variables scaled to O(1), run to convergence.

| method | nodes | SciPy [s] | this crate [s] | speed-up | SciPy heap [MB] | final cost SciPy / this crate | RMSE [µm] SciPy / this crate |
|---|---:|---:|---:|---:|---:|---|---|
| local* | 7392 | 10.0 | 0.42 | 24× | | same optima (median Δt 4e-08 mm) | 20.1 / 19.8 |
| regularized k=64 | 7392 | 9.9 | 0.67 | 15× | 108 | 2920.54528 / 2920.54527 | 15.18 / 15.18 |
| local* | 32736 | 30.2 | 0.58 | 52× | | same optima (median Δt 4e-08 mm) | 20.1 / 19.3 |
| regularized k=64 | 32736 | 65.2 | 0.89 | 73× | 502 | 2842.38576 / 2842.38571 | 15.32 / 15.32 |

(This crate's times here are the full pipeline. SciPy's local fit is
extrapolated from 1000 nodes.)

The regularised fits reach the same optimum: the costs agree to 7–8
significant digits (this crate's slightly lower) and the thickness RMSEs to
0.001 µm. The Gauss–Newton/PCG engine needs 7 outer iterations (~50 CG
products) where L-BFGS-B needs 214–330. The gap widens with N.
