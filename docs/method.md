# Method

This page describes what the crate computes and how, and where it
deliberately differs from the original implementations. The notation follows
our 2025 paper (Neeteson & Boyd 2025). The experiments are on separate pages:
[examples.md](examples.md), [segmentation.md](segmentation.md),
[volumetric_validation.md](volumetric_validation.md),
[clinical_ct.md](clinical_ct.md) and [performance.md](performance.md).

## Pipeline

1. **Surface.** The outer bone surface is extracted from the binary mask with
   surface nets: one vertex per boundary-straddling cell at the mean of its
   edge crossings, and one quad per boundary voxel edge, split along the
   shorter diagonal. Voxels outside the image count as background, so the
   surface is closed and oriented outwards. Only two layers of the
   cell-to-vertex index are kept at a time. The mask can be Gaussian-blurred
   first for sub-voxel vertex placement.
2. **Smoothing.** Taubin λ|μ smoothing (uniform umbrella operator,
   `1/λ + 1/μ = k_pb`) low-pass filters the staircase without shrinkage.
3. **Nodes.** Vertices on caps (where the mask is cut by the image boundary,
   such as the ends of an HR-pQCT stack) are excluded. With a sub-mask, only
   vertices next to a sub-mask voxel are kept. With a normal constraint,
   vertices whose constrained direction deviates by more than about 70° from
   the surface normal are dropped.
4. **Sampling.** At every node `p` with inward unit normal `n` (optionally
   projected into a plane or replaced by an axis), the image is sampled by
   trilinear interpolation at `p + x_j n`, `x_j = −outside, …, +inside`, and
   optionally smoothed along the line with a 1D Gaussian. The image is read in
   its own dtype, and profiles are stored as `f32`.
5. **Initialisation.** One model is fitted to all profiles at once.
6. **Fit** with the chosen method.

## The profile model

```text
f̂(x) = ρs + (ρc − ρs)/2 · [1 + erf((x − m + t/2) / (σ√2))]
          + (ρb − ρc)/2 · [1 + erf((x − m − t/2) / (σ√2))]
```

`m` is the distance from the node to the cortical mid-line (positive
inward), `t` the cortical thickness, `σ` the blur, `ρs` and `ρb` the
soft-tissue and trabecular densities, and `ρc` the fixed cortical density.
The periosteal edge is at `m − t/2` and the endosteal edge at `m + t/2`. With
`u₁ = (x − m + t/2)/σ`, `u₂ = (x − m − t/2)/σ`, `φ` the standard normal
density, `a₁ = ρc − ρs`, `a₂ = ρb − ρc` and `g_k = a_k φ(u_k)/σ`, the
gradient is

```text
∂f̂/∂m = −g₁ − g₂            ∂f̂/∂ρs = (1 − erf(u₁/√2))/2
∂f̂/∂t = (g₁ − g₂)/2         ∂f̂/∂ρb = (1 + erf(u₂/√2))/2
∂f̂/∂σ = −(g₁u₁ + g₂u₂)
```

Residuals are weighted by `γ_j = γ0 + (1 − γ0) |x_j| / max|x|`, so that
residuals at the node weigh `γ0` times those at the end of the line and the
model prefers to place its edges near the surface rather than on trabeculae
(Treece et al. 2010). The absolute value follows the original Python
implementation in Bonelab; the paper writes the weighting without it.

## Blur

`σ` can be free per node (as in our 2025 paper), fixed to one value, or set
from a Gaussian PSF with per-axis standard deviations `s`, for which the blur
along a line with direction `n` is `σ(n) = √(Σ_a n_a² s_a²)` (how Treece et
al. 2012 and 2015 hold the blur fixed while fitting). The PSF can also be
estimated: a first local fit with free `σ` gives each node's blur, and `σ_i²`
is regressed on `n_i²` over nodes where `σ` is identifiable (`t ≥ min_ratio ·
σ`, `σ` not on a bound), with two rounds of 3·MAD outlier rejection, by
non-negative least squares. The model is isotropic, axial (in-plane plus
along-axis, the model of Treece & Gee 2015, appendix A) or fully
anisotropic. The chosen method then runs with `σ` fixed from the PSF.

## Cortical density

`ρc` is never fitted. Thickness and density trade off (a thinner, denser
cortex blurs into nearly the same profile), so with `ρc` free `t` would be
poorly determined wherever the cortex is thin (Treece et al. 2012). Because
the fit conserves the cortical mass under the blurred profile, a wrong `ρc`
biases thickness roughly as `t ∝ 1/ρc`. There are three choices
(`ModelOptions::rho_c_mode`, Python `rho_c=`):

- fixed (the default, 1000, from Table 1 of the paper), right for a
  density-calibrated scanner when the cortical tissue density is known;
- estimated: one value, the cortical plateau where the cortex is thick enough
  to show one. Starting from the 90th percentile of the profile maxima, every
  node is fitted locally (`σ` free); nodes whose fitted cortex has at least 3
  samples more than `2σ` from both edges give the median of those samples,
  and `ρc` is the median over nodes; this is repeated once. It measures the
  apparent density of the cortex, including its sub-resolution porosity,
  which is what a uniform-cortex model sees. The estimate needs some cortex
  much thicker than about `4σ` in view; where all of it is thin, only nodes
  fitted too thick qualify, and `ρc` comes out too low
  ([volumetric_validation.md](volumetric_validation.md#clinical-ct));
- per-node profile maximum (`profile_max`), as the original Python
  implementation does when no density is given. Because the model is linear
  in `(ρs, ρc, ρb)`, each profile is scaled by `ρ̄c / ρc,i` before the fit
  (`ρ̄c` the median) and `ρs`, `ρb` and the residual are scaled back
  afterwards, so the solvers are unchanged. A thin or blurred cortex never
  reaches its tissue density, so this underestimates `ρc` and overestimates
  `t` exactly where the cortex is thin.

## Local method

Each profile is fitted independently by minimising
`½ Σ_j w_j (f̂(x_j) − f_j)²` (`w = γ / Σγ`) over at most five parameters with a
bounded Levenberg–Marquardt iteration: 5×5 normal equations formed directly
from the analytic gradient and solved by Cholesky, Marquardt scaling
(`JᵀJ + μ diag JᵀJ`, invariant to the very different scales of positions and
densities), Nielsen's damping update, bounds by clamping the step and
freezing variables on a bound with the gradient pointing outwards, and
stopping on relative cost decrease (`ftol`), the MINPACK-style scaled
gradient (`gtol`) or step size (`xtol`). Nodes are fitted in parallel. This is
the classic CBM formulation as re-implemented in our 2025 paper.

## The global engine

The multigrid, regularised, volumetric and initial global fits are all one
weighted least-squares problem,

```text
L = 1/(2NM) Σ_ij γ_j (f̂_ij − f_ij)²                     (data)
  + ½ Σ_c κ_c ‖ω v_c‖²                                  (smoothing)
  + ½ Σ_{surfaces} ‖r_curv‖²                            (curvature, extension)
```

in which each parameter is fixed (optionally per node), one global value, or
a field with one value per degree of freedom: the nodes (regularised), or
control points interpolated to the nodes by a sparse matrix `A` (multigrid).
The initial fit has every parameter global.

It is solved by bounded Gauss–Newton / Levenberg–Marquardt with matrix-free
preconditioned conjugate gradients. Each outer iteration evaluates the
residuals and, per node, the packed 5×5 Gauss–Newton block
`B_i = Σ_j w_j ∇f̂_ij ∇f̂_ijᵀ` and the gradient. The step solves
`(JᵀJ + μD) δ = −g` by PCG, applying `JᵀJ v` without forming it: map `v` to
node space (`A v` or the identity), apply the `B_i`, add `κ ωᵀω v` and the
curvature term, and map back (`Aᵀ`); global parameters couple through sums
over nodes. The preconditioner is block-Jacobi (one small dense block per
degree of freedom plus the global block, including the penalties'
diagonals). Damping is Marquardt's `D = diag(JᵀJ)` with Nielsen's update, and
bounds are handled by projected steps and active-set freezing. Each PCG solve
stops at a relative residual of `η = min(0.1, √g)` (inexact Newton), down to
`cg_tolerance` near the optimum. The gradient of every term is checked
against finite differences for all roles, with and without interpolation,
for both curvature modes and both Laplacians.

A solve converges when the scaled gradient `max_v |g_v| / (√D_v ‖r‖)` is
below `gtol`, an accepted step reduces the cost by less than `ftol`, the step
is below `xtol`, or the last 10 accepted steps together reduced the cost by
less than 10⁻⁴ (which ends the slow crawl of projected steps when many nodes
sit on a bound). It stops unconverged, flagged
(`FitInfo::budget_exhausted`), when it has used `max_iterations` linear
solves (accepted and rejected trial steps alike) or `cg_max_total` CG
iterations. The budget matters for badly posed inputs: on a segmentation
that has leaked into the marrow, most profiles carry no cortical
information, the smoothing term dominates, and the block-Jacobi
preconditioner needs about 70 CG iterations per step instead of about 14.

Compared with first-order L-BFGS-B on the same objective (as in our 2025
paper), this exploits the least-squares structure: it converges in 5–30
outer iterations to the same optimum as a converged L-BFGS-B solve
([performance.md](performance.md)). All parallel reductions use fixed-size
chunks summed in a fixed order, so results are bitwise identical across runs
and thread counts.

## Multigrid method

For each control-point spacing `s`, from coarse to fine: control points are
chosen by random sequential selection with minimum separation `s` (equivalent
to the paper's procedure, O(N) with a hash grid, deterministic for a seed);
the interpolation matrix `A` lets each node average its `n` nearest control
points with normalised Gaussian weights of standard deviation `s`; and the
fit solves for `m`, `t`, `σ` as fields on the control points with `ρs`, `ρb`
global, initialised from the previous level or the global fit.

## Regularised method

`m`, `t` and `σ` are fields on the nodes and `ρs`, `ρb` global, with the
smoothing penalty

```text
λ/(2N) · ρc² · [ ‖ω m‖²/t0² + ‖ω t‖²/t0² + ‖ω σ‖²/σ0² ]
```

where `t0`, `σ0` come from the global fit and `ω` is the `N × N` smoothing
operator `(ω x)_i = x_i − Σ_k g(d_ik) x_k / Σ_k g(d_ik)` over the `k` nearest
neighbours (excluding the node itself), with a Gaussian of standard deviation
`σ_reg`. Neighbours farther than `cutoff · σ_reg` (default 3) are dropped,
except the nearest; `cutoff = ∞` gives the plain k-NN operator. The densities
can also be fields (`global_params`), with their own smoothing weight.

With `k` nearest neighbours the operator acts over the radius they cover,
about `√(k/π)` times the node spacing. At HR-pQCT node spacing (about
60 µm), `k = 500` covers about 0.76 mm, so with `σ_reg = 1.9424 mm` the
weights are nearly uniform and the effective length scale is set by `k`. The
implementation is faithful to the paper in this; the curvature Laplacian
avoids it by sampling.

## Curvature penalty

The curvature penalty (an extension) penalises the curvature of the fitted
surfaces `S_peri = V + (m − t/2) n` and `S_endo = V + (m + t/2) n`, using a
fixed discrete Laplace–Beltrami operator `L` of the input surface `V`, with
`½ L S ≈ H n` and the scalar mean curvature taken as the normal component
`H_i = n_i · ½ (L S)_i`, which is robust to the tangential drift of
point-cloud Laplacians on irregular sampling. There are three modes (see
[volumetric_refinement.md](volumetric_refinement.md#1-curvature-penalties)
for how they compare in refinement):

- `"change"` penalises the curvature added by the fit, in excess of the
  parallel surface of `V` at the local offset. To first order
  `H(V + d n) − H(V_d) ≈ ½ Δ_S d`, computed as
  `n_i · ½ Σ_k L_ik (d_k − d_i) n_k`, so a uniform offset costs nothing and
  only bumps are penalised. (Penalising the full change `‖H(S) − H(V)‖`
  instead biases thickness, because the endosteal surface of a cylinder is
  meant to be more curved than the periosteal one: −0.6 mm on the cylinder
  phantom.) The residual per node and surface is
  `r = √(μ/N) · ρc · ℓ · n_i · ½ Σ_k L_ik (d_k − d_i) n_k`, linear in `(m, t)`.
- `"absolute"` penalises the curvature of the fitted surface above a
  threshold `κ`, `r = √(μ/N) · ρc · ℓ · max(0, K_i − κ)`, leaving curvature
  below `κ` free (Gauss–Newton with the active set re-evaluated each
  iteration). `K` is `|κ₁ + κ₂| = 2|H|` (`curvature_measure="mean"`) or
  `√(κ₁² + κ₂²)` (`"total"`, from a weighted quadric fit to the neighbours
  in each node's tangent plane), which is not zero at a saddle. Both equal
  the contour curvature `1/R` on a shaft; the default `κ = 5 mm⁻¹` is the
  contour-curvature threshold of our HR-pQCT segmentation network (Neeteson
  et al. 2023).
- `"variation"` penalises how fast the curvature changes along the surface,
  `½ ∫ |∇(κ₁ + κ₂)|²` (after the minimum-variation surfaces of Moreton &
  Séquin 1992), with the neighbours and weights of the Laplacian: smoothly
  curved bone is free, bumps and dents are not. It is quadratic in `(m, t)`,
  with `ℓ²` in place of `ℓ`.

The endosteal surface can take its own weight (`curvature_endosteal_weight`).

`ℓ` (by default the globally fitted thickness `t0`) makes curvature
dimensionless, so that `μ` plays the same role as `λ`. Either or both
surfaces can be penalised, and the penalty can be combined with the
smoothing term or used alone (`λ = 0`).

The default Laplacian is a Gaussian point-cloud Laplacian at physical scale
`σ_c` (0.25 mm, 64 neighbours): the Gaussian-weighted neighbourhood mean
within `3σ_c` is estimated by importance-sampling `k` neighbours with
probability ∝ `exp(−d²/2σ_c²)` (deterministic per node), and rows are
normalised by the weighted second moment `M2_i = Σ w_ik |p_k − p_i|²`, so
that `L f = 4(Σ w f − f)/M2 ≈ Δf` for any truncation. Sampling rather than
taking the `k` nearest keeps the operator at the requested scale whatever the
node density, which keeps the Gauss–Newton systems well conditioned. The
cotangent Laplacian (`curvature_scale=None`) gives curvature at mesh
resolution, but its bending energy scales as `h⁻⁴` with edge length `h`, so
its systems are stiff; it suits small surfaces or serves as a reference.

## Volumetric model

The volumetric model (an extension, `method="volumetric"`) fits the voxels of
a band around the fixed surface instead of one profile per node. The
reasoning is in [volumetric_model.md](volumetric_model.md) and its validation
in [volumetric_validation.md](volumetric_validation.md).

**Band.** The band holds the voxels at depth `d ∈ [−outside, inside]` below
the surface (subsampled by `vol_voxel_stride`), except those in an excluded
region. For each voxel the crate finds its closest point `c` on the surface
(a k-d tree over the vertices, then the faces around the nearest ones), the
face and barycentric coordinates `β` of `c`, the depth `d = |x − c|` signed by
whether the voxel is inside the closed mesh, the normal `n = (x − c)/d`, and
`a² = (n · e_z)²` along the scanner axis. The caps the mesh gets where the
bone leaves the image are removed, and a voxel whose closest point lies on a
cap's rim, or whose closest face belongs to unfitted or excluded nodes, is
dropped, because the planar model does not hold there. Each voxel carries the
1D residual weight of its depth. Voxels farther from the voxelised surface
than the sampling range plus two voxels are skipped before the closest-point
search.

**Model.** Each voxel's parameters are the barycentric interpolation of its
face's vertex fields, `θ(x) = Σ_j β_j θ_j`, and its model value is the 1D
profile model at its depth with a global PSF:

```text
I_model(x) = f( d ; m, t, σ_n, ρs, ρb ; ρc ),      σ_n² = σ_xy² (1 − a²) + σ_z² a²
```

This is the in-plane / out-of-plane blur model of Treece & Gee (2015), who
regress it on the blurs of 1D fits; here `σ_xy` and `σ_z` are fitted with the
model. Either can be fixed, or `σ_z` tied to `σ_xy` (`vol_sigma_z="tied"`),
which is needed on a shaft, where every normal is in-plane and `σ_z` is not
identified. `ρc` is fixed or estimated by the plateau rule from the band. The
PSF belongs to the scan, not to a bone, so several bones in one scan should
be fitted together: the mask of all bones gives one mesh component per bone,
the components share the PSF (and `ρc`), surface-distance neighbourhoods keep
them apart, and each band voxel belongs to its nearest surface. For per-bone
fits or refinements that PSF is then held (`vol_sigma_xy`, `vol_sigma_z`).

**Cost.**

```text
E = 1/(2V) Σ_x w(d) (I(x) − I_model(x))²  +  C(m, t)  +  ½ Σ_{c ∈ ρs, ρb} κ_c ‖ω x_c‖²
```

`V` is the number of band voxels and `C` the curvature penalty on the fitted
surfaces, the only regulariser of the edge positions (default weight
`vol_curvature_weight = 0.3`). The density fields are smoothed by the
regularised method's operator with `κ = vol_lambda / N`, or are one value
each with `vol_global_densities`.

**Solve.** The global engine solves it. Each voxel depends on the three
vertices of its face, so the Gauss–Newton matrix couples the vertices along
the mesh edges; it is never formed. One packed Gram matrix is built per face
over its three vertex blocks and the global PSF, the matrix–vector product is
accumulated face by face, and the block-Jacobi preconditioner takes each
vertex's own block. As in the 1D global fit, the fit starts with every
parameter global (one model for all band voxels) and then releases the
fields.

**Reported directions.** The model measures `m` and `t` along the surface
normal, each voxel's direction to its closest point, and the curvature
penalty and the axial part of the PSF use the node's surface normal `n`. The
results are reported along the 1D methods' directions `d`, those of
`normal_constraint` (none: the vertex normal; `"in_plane"`: its projection
into the slice plane). For a planar cortex an edge at distance `x` along `n`
lies at `x / (n · d)` along `d`, so reported `m` and `t` are the model's
divided by `n · d`, and warm starts are converted back. Within two voxels of
the image's edge the surface's tilt is the rounding of the cut, not anatomy,
so no conversion is made there. Reporting along in-plane directions also
keeps the fitted surfaces from folding at the cut ends, where the vertex
normals tilt towards the axis.

**Reliability.** `result.volumetric` reports the fitted PSF and four
reliability outputs. The noise standard deviation `s` and the lag-one
correlation `ρ_a` along each axis come from the residuals of the outer half
of the outside band (soft tissue). The local standard error of `t`,
`std_error_t`, is the sandwich error `s √[B⁻¹ C B⁻¹]_tt` from the node's own
voxels, with `B = Σ w_i g_i g_iᵀ` (`g_i = β_i ∇f`, the gradient of voxel `i`'s
model value in the node's parameters) and `C = Σ_ij ρ_ij (w_i g_i)(w_j g_j)ᵀ`
summed over pairs of the node's voxels with the separable correlation
`ρ_ij = Π_a ρ_a^{|Δ_a|}`; it is the error of a node fitted from its own voxels
alone, without the regulariser or its neighbours. (The block factor
`Π(1 + ρ_a)/(1 − ρ_a)`, also reported, holds only for a block long along
every axis; a node's voxels form a column about one voxel across, and on the
clinical knee the block factor would have given median standard errors of
1.1–2.9 mm instead of 0.09–0.17 mm.) `data_share` is `B / (B + P)` per field,
the data's share of the curvature against the penalties' `P`, and `misfit`
the node's RMS weighted residual relative to `s`, about 1 where the model
holds. In refinement the PSF is held between partial steps, like the global
densities, and refitted on full steps.

## Neighbourhoods

The smoothing operator, the curvature Laplacian and the multigrid control
points need neighbourhoods of surface nodes, and `neighbour_distance` chooses
how they are measured. `"euclidean"` uses a k-d tree in space and is the
default of the 1D methods. `"surface"` uses the shortest path along mesh
edges, a Dijkstra search from each node truncated at the operator's radius
with ties broken by node index, and is the volumetric default. Nodes on
opposite cortices of a thin plate, or on two bones in one mesh, are close in
space but far along the surface, so they are not coupled. Edge paths
overestimate geodesic distance by at most 15% on a mesh split by one diagonal
([volumetric_model.md](volumetric_model.md#5-distance-along-the-surface)).
With surface distance, multigrid control points are chosen by coverage, and
each node interpolates from the controls within three spacings of it along
the surface.

## Segmentation refinement

The fit locates the periosteal edge at `x = m − t/2` along each profile,
independently of where the segmentation placed the node. Refinement (an
extension) iterates

```text
v_i ← v_i + α (m_i − t_i/2) n_i        (0 < α ≤ 1, default 0.5)
```

an under-relaxed fixed-point iteration whose fixed point is the fitted
periosteal surface. Only trusted fits move a node: the fitted edge must lie
inside the sampled range (with a margin of `σ`), `m` must be off its bounds
and `t` off its upper bound (a thickness at its lower bound is a cortex too
thin to resolve, and its edge is still located), the cortex must be brighter
than both neighbouring tissues, and the residual at most three times the
median. Without this gate, profiles along the steep flanks of a gross
segmentation bump produced fits at the bounds that pushed the bump outwards.
The displacement is extended from trusted nodes to the others by harmonic
interpolation, so untrusted regions follow smoothly instead of tearing.

The surface is carried either as the same mesh (cap vertices slide within
their cap plane, and a few Taubin iterations keep the mesh regular) or, with
`level_set=True`, as the zero level of a signed distance on the image grid,
remeshed every iteration so that it cannot fold. After each move the
normals are recomputed, the profiles resampled and the model refitted,
warm-started with `m` shifted by the move. While the surface moves, the inner
fit stops once its edges change by less than `edge_ratio` times the last
move; with `freeze=True`, regions that have stopped moving are frozen and no
longer refitted. Refinement converges when a full, converged fit moves the
surface by less than the tolerance (a tenth of the smallest voxel spacing,
RMS) overall and by less than twice that in every region; a fit that
exhausts its work budget does not stop the loop but cannot end it as
converged, and a run that makes no progress for several fits stops as
stalled, reporting why its profiles fail. The details and the experiments
behind them are in [refinement_experiments.md](refinement_experiments.md).

Refinement corrects the position of a surface, not its topology: a
segmentation that has leaked into the marrow (through a cortical break, say)
has part of the endosteal surface and the trabecular network in its outer
surface, and no profile fit can turn that into a periosteum. The outputs are
the refined periosteal surface, the fitted endosteal surface, and masks of
the regions they enclose (even–odd ray-parity rasterisation along axis 0) and
of the cortex between them; the same masks can be produced from any single
fit (`CbmResult.masks`, `cortical_masks`).

## Starting segmentations

Refinement needs a starting surface that is topologically right (one closed
surface around each bone) and close enough that the profiles cross the
cortex. `hull_segmentation` (recommended for HR-pQCT) thresholds the image
(default 320 mg HA/cm³), applies a binary median against noise speckle,
keeps the largest 3D component and takes the convex hull of every slice. It
cannot leak, since it always contains the thresholded bone, and its errors
are outward bulges over concave parts of the periosteum, which refinement
pulls in; its limits are concavities deeper than the inward sampling range
and two bones joined at the threshold. `bone_hulls` (for several bones, as in
the knee) closes each bone with a ball instead, which keeps wide concavities
such as the femoral notch open. `buie_segmentation` is the dual-threshold
method of Buie et al. (2007), the standard-of-care reference; it is usually
close but fails topologically on cortical breaks of about 1.2 mm and more.
[segmentation.md](segmentation.md) has the implementation details and the
comparison.

## Memory and performance

The image is never copied or converted: all integer and float dtypes are
sampled in place, and surface extraction keeps two slices of index. Profiles
are `f32`. Global solves hold, per node, one packed Gauss–Newton block and a
few work vectors (about 25 doubles) and never assemble the Hessian. Sparse
neighbourhood operators are row-compressed with `u32` indices and (for `ω`)
`f32` weights, 8 bytes per neighbour, filled in place at a fixed stride and
compacted, so construction never holds two copies; transposed products
scatter into 16 fixed per-chunk buffers instead of storing the transpose.
The k-d tree is the `kiddo` crate's immutable tree, wrapped so that results
are canonical (k-NN results are the `k` smallest by squared distance and
index, radius results are sorted by index, with distances recomputed in a
fixed order) and do not depend on the tree's layout.

## Differences from the original implementations

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
smooths parameters in ways not described in full in the literature. This
crate implements the model, the weighting, fixed and PSF blur and a PSF
estimate, not a bit-exact replica.
