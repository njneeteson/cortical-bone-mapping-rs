# Method

This page describes what is computed and how, including where this
implementation deliberately differs from the original implementations. The
mathematical notation follows our paper (Neeteson & Boyd 2025).

* [Pipeline](#pipeline)
* [The profile model](#the-profile-model)
* [Residual weighting](#residual-weighting)
* [Local method (Treece et al.)](#local-method)
* [Blur: free, fixed, PSF, estimated PSF](#blur)
* [Cortical density](#cortical-density)
* [The global engine](#the-global-engine)
* [Multigrid method](#multigrid-method)
* [Regularised method](#regularised-method)
* [Curvature penalty (extension)](#curvature-penalty)
* [Segmentation refinement (extension)](#segmentation-refinement)
* [Memory and performance design](#memory-and-performance-design)
* [Differences from the original implementations](#differences)

## Pipeline

1. **Surface.** The outer bone surface is extracted from the binary mask with
   *surface nets*: one vertex per boundary-straddling cell at the mean of its
   edge crossings, one quad per boundary voxel edge, split along the shorter
   diagonal. Voxels outside the image count as background, so the surface is
   always closed and oriented outwards. Only two layers of the cell→vertex
   index are kept at a time, so memory is O(slice) plus the output. Optionally
   the mask is Gaussian-blurred first for sub-voxel vertex placement.
2. **Smoothing.** Taubin λ|μ smoothing (uniform umbrella operator,
   `1/λ + 1/μ = k_pb`), which low-pass filters the staircase without shrinkage.
3. **Node selection.** Vertices on *caps* (where the mask is cut by the image
   boundary, e.g. the ends of an HR-pQCT stack) are excluded. With a
   sub-mask, only vertices next to a sub-mask voxel (any of the 8 voxels of
   their cell) are kept. With a normal constraint, vertices whose constrained
   direction deviates by more than ~70° from the surface normal are dropped.
4. **Sampling.** At every node `p` with inward unit normal `n` (optionally
   projected into a plane or replaced by an axis), the image is sampled by
   trilinear interpolation at `p + x_j n`, `x_j = −outside, …, +inside`, and
   optionally smoothed along the line with a 1D Gaussian. The image is read in
   its own dtype (no float copy); profiles are stored as `f32`.
5. **Initialisation.** One model is fitted to all profiles at once.
6. **Fit** with the chosen method.

## The profile model

```text
f̂(x) = ρs + (ρc − ρs)/2 · [1 + erf((x − m + t/2) / (σ√2))]
          + (ρb − ρc)/2 · [1 + erf((x − m − t/2) / (σ√2))]
```

`m` is the distance from the node to the cortical mid-line (positive inward),
`t` the cortical thickness, `σ` the blur, `ρs`, `ρb` the soft-tissue and
trabecular/marrow densities and `ρc` the (fixed) cortical density. The
periosteal edge is at `m − t/2`, the endosteal edge at `m + t/2`. The
gradient with respect to `(m, t, σ, ρs, ρb)` is analytic. With
`u₁ = (x − m + t/2)/σ`, `u₂ = (x − m − t/2)/σ`, `φ` the standard normal
density, `a₁ = ρc − ρs`, `a₂ = ρb − ρc`, `g_k = a_k φ(u_k)/σ`:

```text
∂f̂/∂m = −g₁ − g₂            ∂f̂/∂ρs = (1 − erf(u₁/√2))/2
∂f̂/∂t = (g₁ − g₂)/2         ∂f̂/∂ρb = (1 + erf(u₂/√2))/2
∂f̂/∂σ = −(g₁u₁ + g₂u₂)
```

## Residual weighting

`γ_j = γ0 + (1 − γ0) |x_j| / max|x|`: residuals at the node weigh `γ0` times
those at the end of the line, so the model prefers to place its edges near
the surface instead of on trabeculae (Treece et al. 2010). The absolute value
follows the original Python implementation in Bonelab (the paper writes
the weighting without it).

## Local method

Each profile is fitted independently by minimising
`½ Σ_j w_j (f̂(x_j) − f_j)²` (`w = γ / Σγ`) over at most five parameters with
a **bounded Levenberg–Marquardt** iteration:

* normal equations `JᵀJ` (5×5) and `Jᵀr` formed directly from the analytic
  gradient, solved by Cholesky;
* Marquardt scaling `(JᵀJ + μ diag JᵀJ)`, so the iteration is invariant to the
  very different scales of positions (mm) and densities (hundreds);
* Nielsen's damping update from the gain ratio;
* bounds by clamping the step and freezing variables that sit on a bound with
  the gradient pointing outwards (projected LM);
* stopping on relative cost decrease (`ftol`), the MINPACK-style scaled
  gradient `max_i |g_i| / (‖J_i‖‖r‖)` (`gtol`) or step size (`xtol`).

Nodes are fitted in parallel with no shared state. This is the classic CBM
formulation as re-implemented in our 2025 paper ("local method").

## Blur

`σ` can be

* **free** per node (Neeteson & Boyd);
* **fixed** to one value;
* **from a PSF**: for a Gaussian PSF with per-axis standard deviations
  `s`, the blur along a line with direction `n` is `σ(n) = √(Σ_a n_a² s_a²)`;
  this is how Treece et al. (2012, 2015) hold the blur fixed while fitting;
* **estimated PSF**: a first local fit with free `σ`; then `σ_i²` is
  regressed on `n_i²` (isotropic, axial = in-plane + along-axis, or fully
  anisotropic; non-negative least squares by active-set enumeration) using
  nodes where `σ` is identifiable (`t ≥ min_ratio · σ`, `σ` not on a bound),
  with two rounds of 3·MAD outlier rejection; then the chosen method is run
  with `σ` fixed from the PSF.

The cortical density `ρc` is always fixed; `ρc` and `σ` cannot both be
determined from a thin cortex (Treece et al. 2012).

## Cortical density

`ρc` is fixed during the fit: with `ρc` free, thickness and density trade off
(a thinner, denser cortex blurs into nearly the same profile), so `t` would
be poorly determined where the cortex is thin. Three ways to choose it
(`ModelOptions::rho_c_mode`, Python `rho_c=`):

* **fixed** (default, `ρc = 1000`, Table 1): right for a density-calibrated
  scanner when the cortical tissue density is known. A wrong value biases
  thickness roughly in proportion, `t ∝ 1/ρc`, because the fit conserves the
  cortical mass under the blurred profile.
* **estimate**: one value for all nodes, the cortical plateau level where the
  cortex is thick enough to show one. Starting from the 90th percentile of
  the profile maxima, fit every node locally (`σ` free), keep nodes whose
  fitted cortex has at least 3 samples farther than `2σ` from both edges,
  take the median of those samples per node, and the median over nodes;
  repeat once. It measures the *apparent* density of the cortex, i.e. with
  its sub-resolution porosity, which is what a uniform-cortex model sees.
  Profile maxima alone are biased up by noise and down by blur and profile
  smoothing on a thin cortex.
* **profile_max**: per node, the maximum of its (smoothed) profile, as the
  original Python implementation does when no density is given. Per-node
  densities need no change to the solvers: the model is linear in
  `(ρs, ρc, ρb)`, so each profile is scaled by `ρ̄c / ρc,i` before the fit
  (`ρ̄c` the median) and `ρs, ρb` and the residual are scaled back afterwards;
  `m, t, σ` are unchanged. This differs from a per-node density only in the
  relative weight of each node's data term against the regularisation. A thin
  or blurred cortex never reaches its tissue density, so this underestimates
  `ρc` and overestimates `t` exactly where the cortex is thin.

## The global engine

The multigrid, regularised and initial global fits are all instances of one
weighted least-squares problem over all profiles,

```text
L = 1/(2NM) Σ_ij γ_j (f̂_ij − f_ij)²                     (data)
  + ½ Σ_c κ_c ‖ω v_c‖²                                  (smoothing)
  + ½ Σ_{surfaces} ‖r_curv‖²                            (curvature, extension)
```

in which each of the five parameters plays a *role*: fixed (optionally per
node), one global value, or a *field* with one value per degree of freedom.
Field degrees of freedom are the nodes (regularised) or control points
interpolated to the nodes with a sparse matrix `A` (multigrid). The initial
global fit has every parameter global.

It is solved with **bounded Gauss–Newton / Levenberg–Marquardt with
matrix-free preconditioned conjugate gradients**:

* each outer iteration evaluates residuals and, per node, the 5×5
  Gauss–Newton block `B_i = Σ_j w_j ∇f̂_ij ∇f̂_ijᵀ` (15 doubles, packed) and the
  gradient;
* the step solves `(JᵀJ + μD) δ = −g` by PCG, with `JᵀJ v` applied
  matrix-free: map `v` to node space (`A v` or identity), apply `B_i`
  per node, add `κ ωᵀω v` and the curvature Gauss–Newton term, and map back
  (`Aᵀ`). Global parameters couple through sums over nodes;
* block-Jacobi preconditioner (one small dense block per degree of freedom
  plus the global block, including the diagonals of the penalty terms);
* Marquardt damping `D = diag(JᵀJ)`, Nielsen's update, projected steps and
  active-set freezing for the bounds; the predicted reduction uses the
  actual (projected) step;
* the gradient of every term has been checked against finite differences for
  all roles, with and without interpolation, for both curvature modes and
  both Laplacians (`global::tests`).

Compared with first-order L-BFGS-B on the same objective (as in our 2025 paper)
this exploits the least-squares structure: it converges in 5–30 outer
iterations and reaches the same optimum (the SciPy comparison agrees on the
final cost to 8 significant digits; see [performance](performance.md)).

All parallel reductions use fixed-size chunks summed in a fixed order, so
results are **bitwise identical** across runs and thread counts.

## Multigrid method

For each control-point spacing `s` (coarse to fine):

1. control points: random sequential selection with minimum separation `s`
   (equivalent to the paper's procedure: visit points in random order, accept
   a point if no accepted point lies within `s`; O(N) with a hash grid;
   deterministic for a seed);
2. interpolation matrix `A` (N × Q): each node averages its `n` nearest
   control points with normalised Gaussian weights of standard deviation `s`;
3. solve with `m, t, σ` as fields on the control points and `ρs, ρb` global,
   initialised from the previous level (or the global fit).

## Regularised method

`m, t, σ` per node, `ρs, ρb` global, plus the smoothing penalty:

```text
λ/(2N) · ρc² · [ ‖ω m‖²/t0² + ‖ω t‖²/t0² + ‖ω σ‖²/σ0² ]
```

with `t0`, `σ0` from the global fit and `ω` the `N × N` smoothing operator,
`(ω x)_i = x_i − Σ_k g(d_ik) x_k / Σ_k g(d_ik)` over the `k` nearest
neighbours (excluding the node itself) with a Gaussian of standard deviation
`σ_reg`. Neighbours farther than `cutoff · σ_reg` (default 3) are dropped
(the nearest is always kept); `cutoff = ∞` gives the plain k-NN operator.

**Note on scale.** With `k` nearest neighbours the operator acts over the
radius they cover, `≈ √(k/π) · node spacing`. At HR-pQCT node spacing (~60 µm),
`k = 500` covers ~0.76 mm, so with `σ_reg = 1.9424 mm` the Gaussian weights are
nearly uniform and the effective length scale is set by `k`, not by
`σ_reg`. The implementation is faithful to the paper in this respect; the
curvature Laplacian below avoids it by sampling.

## Curvature penalty

An extension: penalise the curvature of the *fitted surfaces*

```text
S_peri = V + (m − t/2) n,     S_endo = V + (m + t/2) n
```

using a fixed discrete Laplace–Beltrami operator `L` of the input surface
`V`, with `½ L S ≈ H n` (mean-curvature normal) and the scalar mean
curvature taken as the normal component `H_i = n_i · ½ (L S)_i` (robust to
the tangential drift of point-cloud Laplacians on irregular sampling). Two
modes:

* **`change`**: curvature *added by the fit*, i.e. in excess of the parallel
  surface of `V` at the local offset. To first order
  `H(V + d n) − H(V_d) ≈ ½ Δ_S d`, computed as
  `n_i · ½ Σ_k L_ik (d_k − d_i) n_k`. A uniform offset (any parallel surface)
  costs nothing, so neither the shape of the bone nor a uniform cortex is
  penalised, only bumps. (An earlier version penalised the full curvature
  change `‖H(S) − H(V)‖`; that biases thickness, because the endosteal surface
  of a cylinder is *meant* to be more curved than the periosteal one. The
  phantom showed a −0.6 mm bias, which motivated the parallel-surface form.)
  Residual per node and surface:
  `r = √(μ/N) · ρc · ℓ · n_i · ½ Σ_k L_ik (d_k − d_i) n_k`, linear in `(m, t)`.
* **`excess`**: absolute mean curvature above a threshold `κ`,
  `r = √(μ/N) · ρc · ℓ · max(0, |H_i| − κ)`, leaving curvature below `κ` free
  (Gauss–Newton with the active set and sign re-evaluated each iteration).

`ℓ` (default: the globally fitted thickness `t0`) makes curvature
dimensionless so that `μ` plays the same role as `λ`. Either or both surfaces
can be penalised. It can be combined with the spatial smoothing term or used
alone (`λ = 0`).

Two Laplacians:

* **Gaussian point-cloud Laplacian at physical scale `σ_c`** (default,
  `σ_c = 0.25 mm`, 64 neighbours): the Gaussian-weighted neighbourhood mean
  within `3σ_c` is estimated by *importance sampling* `k` neighbours with
  probability ∝ `exp(−d²/2σ_c²)` (deterministic per node). Rows are normalised
  by the weighted second moment `M2_i = Σ w_ik |p_k − p_i|²` so that
  `L f = 4(Σ w f − f)/M2 ≈ Δf` for any truncation. Sampling (instead of the
  `k` nearest) keeps the operator at the requested scale whatever the node
  density, which keeps the Gauss–Newton systems well conditioned: at 134k
  nodes, nearest-neighbour rows needed 1447 CG iterations and 65 s; sampled
  rows need 435 and 21 s with the same accuracy.
* **Cotangent Laplacian** (`curvature_scale=None`): curvature at mesh
  resolution, the textbook discrete mean curvature. Its bending energy scales
  as `h⁻⁴` with edge length `h`, so the systems are stiff and need thousands
  of CG iterations; use it for small surfaces or as a reference.

## Segmentation refinement

The fit locates the periosteal edge at `x = m − t/2` along each profile
independently of where the segmentation placed the node. Refinement iterates

```text
v_i ← v_i + α (m_i − t_i/2) n_i        (0 < α ≤ 1, default 0.5)
```

* only **trusted fits** move a vertex. The fitted periosteal edge must lie
  inside the sampled range (with a margin of `σ`), `m` and `t` must be off
  their bounds, the cortex must be brighter than both neighbouring tissues,
  and the residual must be ≤ 3× the median. Without this gate, profiles on the
  steep flanks of a gross segmentation bump (lines running alongside the
  cortex) produced fits at the bounds that pushed the bumps *outwards*: Dice
  peaked at iteration 4 and then fell (see the lobed phantom in
  [examples](examples.md));
* the displacement field is extended from trusted nodes to all other
  vertices by harmonic (neighbour-average) interpolation, so excluded or
  untrusted regions follow smoothly instead of tearing;
* cap vertices (where the mask meets the image edge) may only slide within
  their cap plane;
* a few Taubin iterations keep the mesh regular; normals are recomputed, the
  profiles resampled and the model refitted, warm-started with `m` shifted by
  the move;
* stop when the RMS displacement is below the tolerance (default a tenth of
  the smallest voxel spacing). With `α = 0.5` the displacement halves every
  iteration (it is a damped fixed-point iteration whose fixed point is the
  fitted periosteal surface); the floor is `α ×` the noise of the fitted edge.

Outputs: the refined periosteal surface, the fitted endosteal surface, and
masks of the regions they enclose (even–odd ray-parity rasterisation along
axis 0) and of the cortex between them. The same masks can be produced from
any single fit (`CbmResult.masks`, `cortical_masks`).

## Memory and performance design

* The image is never copied or converted; all integer and float dtypes are
  sampled in place. Surface extraction keeps two slices of index.
* Profiles are `f32` (`4M` bytes per node).
* Global solves hold, per node, one packed 5×5 Gauss–Newton block and a few
  work vectors (≈ 25 doubles), and never assemble the Hessian.
* Sparse neighbourhood operators are row-compressed with `u32` indices and
  (for `ω`) `f32` weights: 8 bytes per neighbour. They are filled in place at
  a fixed stride and compacted, so construction never holds two copies.
  Transposed products scatter into a fixed number (16) of per-chunk buffers
  instead of storing the transpose.
* k-d tree: implicit (permutation + reordered coordinates), median pivots.

## Differences

From the original Python implementation in Bonelab, on purpose:

| | Bonelab | this crate |
|---|---|---|
| optimiser | SciPy L-BFGS-B (global), TRF with numerical Jacobian (local) | bounded LM with analytic Jacobian (local), Gauss–Newton/LM + PCG (global) |
| regularised loss | m/t penalty scaled by `ρc²/t0` in the loss but `ρc²/t0²` in the gradient; σ penalty only in the gradient | the smoothing penalty consistently (`ρc²/t0²`, `ρc²/σ0²`) in loss and gradient |
| local residuals | `γ_j r_j` (weights effectively `γ²`) | `√γ_j r_j` (weights `γ`, consistent with the global methods) |
| `ω` neighbours | `k` including the point itself | `k` excluding it, optional distance cut-off |
| surface | PyVista contour + `smooth_taubin` (windowed sinc) | surface nets + Taubin λ\|μ |
| reproducibility | | bitwise identical across thread counts |

From Treece et al.: the original CBM software additionally constrains and
smooths parameters in ways not described in full in the literature; this
crate implements the model, the weighting, fixed/PSF blur and a PSF estimate
from thick-cortex nodes, not a bit-exact replica.
