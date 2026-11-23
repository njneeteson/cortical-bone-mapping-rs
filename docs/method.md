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
* [Volumetric model (extension)](#volumetric-model)
* [Neighbourhoods: Euclidean or surface distance](#neighbourhoods)
* [Segmentation refinement (extension)](#segmentation-refinement)
* [Starting segmentations](#starting-segmentations)
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

## Cortical density

`ρc` is never fitted. Thickness and density trade off: a thinner, denser
cortex blurs into nearly the same profile, so with `ρc` free `t` would be
poorly determined wherever the cortex is thin (Treece et al. 2012). There are
three ways to choose it (`ModelOptions::rho_c_mode`, Python `rho_c=`):

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
* inexact Newton: each PCG solve stops at a relative residual of
  `η = min(0.1, √g)` (`g` the scaled gradient below), down to `cg_tolerance`
  near the optimum, so early steps are cheap and late steps accurate;
* the gradient of every term has been checked against finite differences for
  all roles, with and without interpolation, for both curvature modes and
  both Laplacians (`global::tests`).

**Stopping.** A solve stops, converged, when

* the scaled gradient `max_v |g_v| / (√D_v ‖r‖)` is below `gtol`,
* an accepted step reduces the cost by less than `ftol` (relative),
* the step is below `xtol`, or
* the last 10 accepted steps together reduced the cost by less than 10⁻⁴
  (relative). This *stall* test ends the slow crawl of projected steps when
  many nodes sit on a bound (e.g. profiles that miss the cortex).

It stops *unconverged* when it runs out of budget: `max_iterations` linear
solves (accepted and rejected trial steps alike) or `cg_max_total` CG
iterations in all. The fit is then flagged (`FitInfo::budget_exhausted`).
The budget matters for badly posed inputs. On a segmentation that has leaked
into the marrow, most profiles carry no cortical information, the smoothing
term dominates, and the block-Jacobi preconditioner, which only sees each
node's own block, needs ~70 CG iterations per step instead of ~14.

Compared with first-order L-BFGS-B on the same objective (as in our 2025 paper)
this exploits the least-squares structure: it converges in 5–30 outer
iterations and reaches the same optimum as a converged L-BFGS-B solve (see
[performance](performance.md)).

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
  penalised, only bumps. (Penalising the full change `‖H(S) − H(V)‖` instead
  biases thickness: the endosteal surface of a cylinder is *meant* to be more
  curved than the periosteal one; −0.6 mm on the cylinder phantom.)
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
  density, which keeps the Gauss–Newton systems well conditioned (about 3×
  fewer CG iterations than nearest-neighbour rows at 134k nodes, with the
  same accuracy).
* **Cotangent Laplacian** (`curvature_scale=None`): curvature at mesh
  resolution, the textbook discrete mean curvature. Its bending energy scales
  as `h⁻⁴` with edge length `h`, so the systems are stiff and need thousands
  of CG iterations; use it for small surfaces or as a reference.

## Volumetric model

`method="volumetric"` is a separate inner fit that runs alongside the 1D
methods. Instead of one profile per node, it fits the voxels of a band
around the fixed surface. The reasoning is in
[volumetric_model.md](volumetric_model.md); this section is what is
computed.

**The band.** The band is the voxels at depth `d ∈ [−outside, inside]`
below the surface (subsampled by `vol_voxel_stride`), except those in an
excluded region (another bone). For each voxel we find:

* its closest point `c` on the surface, and the face and barycentric
  coordinates `β` of `c` (k-d tree over the vertices, then the faces around
  the nearest ones);
* its depth `d = |x − c|`, signed by whether the voxel is inside the full
  closed mesh (even–odd voxelisation);
* its normal `n = (x − c)/d`, and `a² = (n · e_z)²` along the scanner axis.

The caps the mesh gets where the bone leaves the image are not cortex, so
they are removed. A voxel whose closest point lies on a cap's rim is
dropped, because the planar model does not hold there. So is a voxel whose
closest face belongs to unfitted or excluded nodes. Each voxel carries the
1D residual weight of its depth.

**The model.** Each voxel's parameters are the barycentric interpolation of
its face's vertex fields, `θ(x) = Σ_j β_j θ_j`. The voxel is then the 1D
profile model at its depth:

```text
I_model(x) = f( d ; m, t, σ_n, ρs, ρb ; ρc ),      σ_n² = σ_xy² (1 − a²) + σ_z² a²
```

The PSF is global: `σ_xy` and `σ_z` are fitted. Either can be fixed, or
`σ_z` can be tied to `σ_xy` (`vol_sigma_z="tied"`). Tie it on a shaft,
where every normal is in-plane and `σ_z` is not identified. `ρc` is fixed
or estimated from the band (the plateau rule of the 1D methods); profile
maxima do not apply. As for the 1D methods, the estimate needs some cortex
much thicker than about `4σ` in view: where all of it is thin, the plateau
comes only from nodes fitted too thick, `ρc` is underestimated and the
thickness overestimated, and the PSF with it
([volumetric_validation.md](volumetric_validation.md#what-limits-the-clinical-sites)).
Fix `ρc` there.

The PSF belongs to the scanner and the reconstruction, not to a bone. With
several bones in one scan, fit them together: the mask of all bones gives a
mesh with one component per bone, which share the PSF (and `ρc`), and
surface-distance neighbourhoods keep them apart; each band voxel belongs to
its nearest surface, so no exclusion is needed. For per-bone fits or
refinements, hold that PSF (`vol_sigma_xy`, `vol_sigma_z`). Fitted bone by
bone, the PSF absorbs whatever the model misses locally: on the clinical
knee the per-bone estimates ranged over 0.53–0.80 mm in-plane and
0.57–0.88 mm axially (one bone with the anisotropy reversed), against
0.58 / 0.85 mm for all four bones together
([volumetric_validation.md](volumetric_validation.md#the-clinical-knee)).

**The cost.** The cost is

```text
E = 1/(2V) Σ_x w(d) (I(x) − I_model(x))²  +  C(m, t)  +  ½ Σ_{c ∈ ρs, ρb} κ_c ‖ω x_c‖²
```

* `V` is the number of band voxels.
* `C` is the curvature penalty of the previous section, applied to the
  fitted surfaces. It is the only regulariser of the edge positions. Its
  default weight is `vol_curvature_weight = 0.3`, the best compromise over
  the site phantoms ([volumetric_validation.md](volumetric_validation.md)).
* The density fields are smoothed by the regularised method's operator `ω`,
  with `κ = vol_lambda / N` scaled as there. With `vol_global_densities`
  they are one value each instead.

**The solve.** The solve is the global engine. Each voxel depends on the
three vertices of its face, so the Gauss–Newton matrix couples the vertices
and the edges of the mesh. It is never formed. Per face we build one packed
Gram matrix over the face's three vertex blocks and the global PSF, and the
matrix–vector product is accumulated face by face. The block-Jacobi
preconditioner takes each vertex's own block from these. As in the 1D global
fit, it starts with every parameter global (one model for all band voxels)
and then releases the fields.

**Reliability.** `result.volumetric` reports four reliability outputs.

* **Noise.** The noise sd `s` comes from the residuals of the outer half of
  the outside band (soft tissue). So does the lag-one correlation `ρ_a`
  along each axis, and, for reference, the factor `Π(1 + ρ_a)/(1 − ρ_a)` by
  which correlated voxels overstate the number of independent ones in a
  block long along every axis.
* **`std_error_t`.** The sandwich error `s √[B⁻¹ C B⁻¹]_tt` from the node's
  own voxels: `B = Σ w_i g_i g_iᵀ` with `g_i = β_i ∇f` the gradient of voxel
  `i`'s model value in the node's parameters, and
  `C = Σ_ij ρ_ij (w_i g_i)(w_j g_j)ᵀ`, `ρ_ij = Π_a ρ_a^{|Δ_a|}` the noise
  correlation of the voxel pair. A node's voxels form a column about one
  voxel across, so the block factor would overstate the variance by its
  in-plane factors: on the clinical knee, where `ρ_a` ≈ 0.65–0.88, the
  block factor gave median standard errors of 1.1–2.9 mm, the pair sum
  0.09–0.17 mm. This is the error of a node fitted from its own
  voxels alone, without the regulariser or its neighbours. A node owns a
  share of the voxels of its faces, fewer than one interpolated profile
  touches, so this error is larger than the actual error of a local 1D fit.
* **`data_share`.** `B / (B + P)` per field: the data's share of the
  curvature, against the penalties' share `P`.
* **`misfit`.** The node's RMS weighted residual relative to `s`. It is
  about 1 where the model holds.

**Refinement.** In refinement the outer loop is unchanged. Between
partial steps, the fitted PSF is held like the global densities are, and it
is refitted on full steps.

## Neighbourhoods

The smoothing operator, the curvature Laplacian and the multigrid control
points and interpolation all need neighbourhoods of surface nodes.
`neighbour_distance` chooses between two:

* `"euclidean"` uses a k-d tree in space. It is the default of the 1D
  methods, and their results are unchanged.
* `"surface"` uses the shortest path along mesh edges: a Dijkstra search
  from each node, truncated at the operator's radius, with ties broken by
  node index so that the result is deterministic. It is the volumetric
  default. Nodes on opposite cortices of a thin plate, or on two bones in
  one mesh, are close in space but far along the surface, so they are not
  coupled. Edge paths overestimate geodesic distance, by at most 15% on a
  mesh split by one diagonal
  ([volumetric_model.md](volumetric_model.md#5-distance-along-the-surface)).

With surface distance, the multigrid control points are chosen by coverage,
and each node interpolates from the controls within three spacings of it
along the surface.

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
  Refinement also stops, unconverged, if a fit exhausts its work budget:
  moving the surface on an unreliable fit would only propagate it.

Refinement corrects the *position* of a surface, not its *topology*. A
segmentation that has leaked into the marrow (through a cortical break, say)
has part of the endosteal surface and the trabecular network in its outer
surface; no profile fit can turn that into a periosteum. Hence the choice of
starting segmentation below.

Outputs: the refined periosteal surface, the fitted endosteal surface, and
masks of the regions they enclose (even–odd ray-parity rasterisation along
axis 0) and of the cortex between them. The same masks can be produced from
any single fit (`CbmResult.masks`, `cortical_masks`).

## Starting segmentations

Refinement needs a starting surface that is *topologically* right (one
closed surface around each bone) and *geometrically* close enough that the
profiles cross the cortex (within the sampled range). Two are provided:

* **`hull_segmentation`** (recommended): threshold (default 320 mg HA/cm³),
  binary median (Buie's step 3) against noise speckle, the largest 3D
  component (drops a separate fibula/ulna and stray voxels), then the convex
  hull of every slice. The hull cannot leak: it always contains the
  thresholded bone, so its errors are outward bulges over concave parts of
  the periosteum, which refinement pulls in. Its limits are concavities
  deeper than the inward sampling range, and two bones joined at the
  threshold across a thin soft-tissue gap (they become one component, and
  one hull per slice).
* **`buie_segmentation`**: the dual-threshold method of Buie et al. (2007),
  as the standard-of-care reference. It is usually close, but it fails
  topologically on cortical breaks of ~1.2 mm and more (the marrow mask
  leaks through the break into the cortex); see [segmentation](segmentation.md) for its
  implementation and the comparison.

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
* k-d tree: the `kiddo` crate's immutable tree, wrapped so that results are
  canonical (k-NN results are the `k` smallest by (squared distance, index),
  radius results are sorted by index, with distances recomputed in a fixed
  order), so they do not depend on the tree's layout.

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
