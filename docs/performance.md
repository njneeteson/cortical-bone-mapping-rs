# Performance

This page reports the run time and memory of each method, explains where
they go, and compares the crate with the original Python implementation
(Bonelab) and with SciPy on the same objective.

## Setup

Windows 11, 16 hardware threads, release build. The phantoms are cylinders
with an HR-pQCT-like geometry: 60.7 µm voxels, outer radius 8 mm, cortex
1.2 ± 0.5 mm, PSF 0.1 mm, noise σ = 60, and 8, 32 or 128 slices. Profiles run
from 1 mm outside to 3 mm inside (67 samples) with normals in the transverse
plane. RMSE is the thickness error against the phantom truth.

`cargo run --release --example benchmark -p cortical-bone-mapping` times the
full pipeline (surface extraction and smoothing, sampling, global
initialisation and fit). Peak heap is the most memory allocated during the
call above the input image and mask, measured with a counting global
allocator. `reg+curvature` is `λ = 1`, k = 64, plus the default `"change"`
curvature penalty (four node spacings, about 0.25 mm; `μ = 1`); `volumetric` is the
[volumetric model](method.md#volumetric-model) with its defaults (the
paper's smoothing of `m`, `t` and of the density fields, `λ = 1` over 500
neighbours, plus the `"change"` curvature penalty at weight 0.3) and its
axial blur tied to the in-plane one.

## Results

![Benchmark](images/benchmark.png)

| method | nodes | time [s] | peak heap [MB] | bytes/node | RMSE [µm] | outer it. | CG it. |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 7392 | 0.40 | 19 | 2524 | 19.8 | 12 | 0 |
| multigrid | 7392 | 1.07 | 19 | 2524 | 8.9 | 32 | 425 |
| regularized k=64 | 7392 | 0.62 | 21 | 2797 | 15.2 | 7 | 50 |
| regularized k=500 | 7392 | 1.04 | 46 | 6285 | 13.9 | 7 | 51 |
| reg+curvature | 7392 | 1.69 | 27 | 3662 | 7.8 | 7 | 328 |
| volumetric | 7392 | 28.95 | 92 | 12414 | 31.6 | 33 | 1645 |
| local | 32736 | 0.61 | 34 | 1041 | 19.3 | 13 | 0 |
| multigrid | 32736 | 1.57 | 43 | 1316 | 11.8 | 30 | 443 |
| regularized k=64 | 32736 | 0.98 | 80 | 2449 | 15.3 | 7 | 53 |
| regularized k=500 | 32736 | 1.78 | 194 | 5937 | 14.2 | 7 | 50 |
| reg+curvature | 32736 | 3.18 | 107 | 3284 | 7.8 | 7 | 280 |
| volumetric | 32736 | 11.84 | 404 | 12349 | 8.2 | 30 | 197 |
| local | 134112 | 1.82 | 103 | 768 | 19.1 | 13 | 0 |
| multigrid | 134112 | 4.69 | 140 | 1044 | 12.6 | 32 | 454 |
| regularized k=64 | 134112 | 3.35 | 318 | 2370 | 15.2 | 7 | 55 |
| regularized k=500 | 134112 | 5.67 | 786 | 5858 | 14.1 | 7 | 50 |
| reg+curvature | 134112 | 15.91 | 429 | 3200 | 7.6 | 7 | 285 |
| volumetric | 134112 | 50.91 | 1997 | 14889 | 6.6 | 30 | 205 |

The 1D methods scale close to linearly in the number of nodes. Their memory
at 134k nodes goes to the profiles (`N × 67 × 4 B` = 36 MB), the
Gauss–Newton blocks and work vectors (about 25 doubles per node), the
transposed sparse products (16 buffers of N doubles, 17 MB) and, above all,
the smoothing operator `ω` at 8 B per neighbour: 536 MB at the paper's
k = 500 and 69 MB at k = 64.

The volumetric model is the most accurate on the two longer slabs (8.2 and
6.6 µm), at 7–9 times the time and 2.5 times the memory of the regularised
fit with the same smoothing. On the 8-slice slab (0.49 mm long) it is the
least accurate and uses most of its CG budget: in a slab only a few voxels
long, most band voxels have their closest surface point on the rim where the
image cuts the bone and are dropped (16 slices: 9.2 µm; 32: 8.4 µm). Real
scans are far longer (168 slices for a standard HR-pQCT stack).

### Where the volumetric model's cost goes

`--volumetric` runs the volumetric model at 134k nodes with its smoothing
over 64 neighbours instead of 500, with one global density each instead of
fields, and without the curvature penalty (the smoothing of `m` and `t`
then regularises the edges alone):

| variant | time [s] | peak heap [MB] | RMSE [µm] | CG it. |
|---|---:|---:|---:|---:|
| volumetric | 51.3 | 1997 | 6.6 | 205 |
| volumetric k=64 | 33.8 | 1529 | 6.7 | 196 |
| volumetric, global densities | 27.6 | 926 | 6.5 | 179 |
| volumetric, no curvature | 21.4 | 1884 | 13.2 | 57 |

Of the default fit's 51 s, building the band (the closest surface point of
about 8 million voxels) takes 8 s, the global initialisation 3 s, the field
fit 38 s (32 s of it Gauss–Newton products) and the reliability outputs
2 s. The 500-neighbour operator (about 540 MB) and the larger per-face
Gauss–Newton blocks of the density fields cost a third of the time and half
the memory; on this phantom, whose densities are uniform, they buy nothing,
on real bone, whose trabecular density varies, they matter, and
`reg_neighbours=64` or `vol_global_densities=True` are the cheaper settings
where they do not. Without the curvature penalty the fit takes 40% of the
time and its error doubles: the curvature penalty needs about four times the
CG iterations of the smoothing alone. Voxels farther from the voxelised
surface than the sampling range plus two voxels are skipped before the
closest-point search, which leaves the band unchanged and saved about 1 s.

### Design changes driven by profiling

- The curvature Laplacian first used the `k` nearest neighbours, which at
  60 µm spacing span only about 0.27 mm, so the operator was stiff at the
  node scale (1447 CG iterations, 65 s at 134k nodes). Sampling `k`
  neighbours from the Gaussian at the requested scale gives the same
  accuracy with a third of the CG work.
- Solving each Gauss–Newton step only as accurately as the gradient warrants
  (inexact Newton, forcing term `min(0.1, √g)`) halved the CG work (k = 500:
  90 → 50 iterations; curvature: 440 → 271) with identical RMSEs.
- Building `ω` in place (fixed stride, then compaction) brought the k = 500
  peak down from 1166 MB to 751 MB.
- Every trial step counts against `max_iterations` and every CG iteration
  against `cg_max_total` (2000), so a badly posed problem (a segmentation
  that has leaked into the marrow) stops in bounded time, flagged, instead
  of running for hours.
- All parallel sums use fixed chunks in a fixed order, so results are
  bitwise identical for any thread count, at no measurable cost.

## Comparison with Bonelab

`python docs/scripts/compare_bonelab.py /path/to/Bonelab` runs Bonelab's
`LocalTreeceMinimization`, `GlobalControlPointTreeceMinimization` and
`GlobalRegularizationTreeceMinimization` on the same profiles, sampled once by
this crate; Bonelab's surface and sampling stages are not timed. "This crate
fit" is the global initialisation plus the method fit. Bonelab's local fit
is timed on 400 nodes and extrapolated (*). Its memory is the Python-heap
peak during the fit (tracemalloc). Options follow Table 1 of our 2025 paper
(k = 500 only at the smaller size). The volumetric model has no counterpart
in Bonelab.

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

The whole Rust pipeline peaks at 81 MB for regularised k = 64 at 32 736
nodes, against Bonelab's 275 MB for the fit alone.

Both local fits find the same per-node optima (the RMSEs agree, and so does
the SciPy check below, to a median of 4e-8 mm). The speed-up comes from
analytic Jacobians and 5×5 normal equations instead of SciPy's TRF with
finite differences, compiled code, and all cores.

The global methods are both faster and more accurate here, because
Bonelab's L-BFGS-B runs stop before convergence. At 7392 nodes (regularised,
k = 64):

| Bonelab configuration | RMSE | time |
|---|---|---|
| default tolerances (1e-6) | 18.65 µm | 2.6 s |
| tight tolerances (`f_tol = 1e-14`, `g_tol = 1e-12`) | 16.30 µm | 4.4 s |
| tight tolerances + consistent loss | 15.70 µm | 10.7 s |
| this crate | 15.18 µm | 0.47 s |

Bonelab's regularised loss and gradient also disagree (the loss scales the
m/t penalty by `ρc²/t0` and the gradient by `ρc²/t0²`, and the σ penalty is
missing from the loss), which matters only once the solve runs to
convergence. The regularised speed-ups above are therefore against
unconverged solves; against converged ones the factor is about 10–20
(Bonelab, tight tolerances) and about 70 (SciPy, below).

Bonelab's initial one-model fit stalls on the bounds when the trabecular
initial guess is 0 (`m = 6`, `σ = 2`, `t ≈ 11 mm`), and every global method
inherits that (RMSE ≈ 8–9 mm), so the comparison uses its CLI default of
200. This crate's Gauss–Newton initial fit converges from 0.

## Comparison with SciPy on the same objective

`python docs/scripts/compare_scipy.py` implements this crate's objective in
NumPy/SciPy: `least_squares` (bounded TRF, analytic Jacobian, the same
weighting) for the local fit, and L-BFGS-B (analytic gradient, sparse `ω`
from `cKDTree`, variables scaled to O(1), run to convergence) for the
regularised fit. This crate's times are the full pipeline; SciPy's local fit
is extrapolated from 1000 nodes.

| method | nodes | SciPy [s] | this crate [s] | speed-up | SciPy heap [MB] | final cost SciPy / this crate | RMSE [µm] SciPy / this crate |
|---|---:|---:|---:|---:|---:|---|---|
| local* | 7392 | 10.0 | 0.42 | 24× | | same optima (median Δt 4e-08 mm) | 20.1 / 19.8 |
| regularized k=64 | 7392 | 9.9 | 0.67 | 15× | 108 | 2920.54528 / 2920.54527 | 15.18 / 15.18 |
| local* | 32736 | 30.2 | 0.58 | 52× | | same optima (median Δt 4e-08 mm) | 20.1 / 19.3 |
| regularized k=64 | 32736 | 65.2 | 0.89 | 73× | 502 | 2842.38576 / 2842.38571 | 15.32 / 15.32 |

The regularised fits reach the same optimum: the costs agree to 7–8
significant digits (this crate's slightly lower) and the RMSEs to 0.001 µm.
The Gauss–Newton/PCG engine needs 7 outer iterations (about 50 CG products)
where L-BFGS-B needs 214–330, and the gap widens with N.
