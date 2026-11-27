---
orphan: true
---

# HR-pQCT: progress and decisions

A working log while the method is made to work on HR-pQCT (TRIKNEE, the
right proximal tibia of participant 22, scans TR_1–3, used with the
participant's permission). Newest first. Each entry says what was found,
what was decided, and why.

## Where things stand (2026-11-27, before merging)

- **Example for users:** `examples/hrpqct_periarticular.py`, the
  "HR-pQCT proximal tibia" section of the examples page. 1D regularised
  refinement from our own start, compared with the provided masks in the
  periarticular regions:
  - 15 moves (not converged), 15 min;
  - bone Dice 0.981, trabecular 0.898, cortex 0.360;
  - Ct.Th 2.21 mm against 0.80 mm: the dense subchondral trabecular bone is
    taken as cortex.
  The local method diverges there; the volumetric method is left out until
  its refinement is tuned.
- **Done since the entry below:**
  - the compact band (9 bytes per voxel), validated;
  - the band built without a second copy;
  - the narrow-band level set (initialisation 404 → 57 s, ~61 s per move
    against ~110 s);
  - the active-contour idea recorded in `extension_ideas/`.
- **Next** (also in the user's to-do list): refinement's surface motion
  (re-binarising with fold cleanup, a smoothing option between fits,
  step sizes from control theory, convergence on "the masks stop
  changing"); the regularisation's strength and the subchondral cortex;
  then the volumetric method on HR-pQCT.

## Where things stood (end of 2026-11-26)

Test case throughout: the tall slab of TR_1 (256 slices, ±128 around the
mask's mid-height), `grid_step` 4 (139 820 nodes), σ_reg 25 voxels
(1.52 mm, neighbours reaching 3σ_reg), inside 5 mm, `vol_init_from_local`,
the fibula excluded, curvature penalty on.

**Done this session** (all tests, clippy and rustfmt pass; uncommitted):

- The field fit's stall with the curvature penalty on: degenerate rows of
  the Gaussian Laplacian, now without a curvature residual (below).
- A line search along accepted steps with gain ratio `ρ > 1`, for the
  linear convergence of Gauss–Newton on large (misfit) residuals: the same
  solution (5 µm at most per node after 100 iterations), the cost reached by
  iteration 61 instead of 100, the 99th percentile step measure about 3×
  smaller at a given iteration.
- A `cg_forcing` solver option (the forcing term's cap; default 0.1, as
  before). Tighter linear solves did not help, so the default stays.
- Solver diagnostics in the debug log: each step's gain ratio; where the
  step and gradient measures are largest (variable, the node's values on
  every channel, `*` on a bound, its data weight) and their median and 99th
  percentile; the number of variables on a bound; at trace level, a rejected
  step's slope and curvature against central differences, by term and by
  block of variables.
- The band: an exact bounded nearest-vertex search along rows (closest
  points 197 → 141 s on the slab); 20 bytes per voxel instead of 36 (the
  weight computed from the depth, the grid indices only for the reliability
  outputs); built slice by slice into arrays in grid order and reordered by
  face one array at a time, so the build no longer holds a second copy of
  every voxel. Benchmarks: `benches/closest.rs` (new), `benches/volumetric.rs`.
- Intermediate fits' per-node residuals from one residual pass instead of
  the reliability's node moments.

**Not yet measured:** the last change (the band built without a second
copy) passed the tests but has not been run on the slab or benchmarked;
the slab's peak memory before it was 18.25 GB (whether at the band build or
in the diagnostic script's full-size arrays is not known; the script now
records the peak after loading and after the fit).

**Next, in order:**

1. Benchmark the band build (`cargo bench --features bench --bench
   volumetric -- band`) and run the slab once (`hr_diag3.py`, as above) to
   confirm the band time and the peak memory after the fit.
2. Regularisation strength (asked by the user): the penalties end at about
   3 % of the cost (data 9.99·10⁴, smoothing 2.5·10³, curvature 5.8·10²).
   The nodes that converge slowly have little contrast (`ρs` close to
   `ρb`), so the data barely fix their position and the penalties are too
   weak to. Run the slab with `reliability=True` for the per-node data share
   (how many nodes the data alone decide), then with `vol_edge_lambda` and
   `vol_lambda` at 10× to see whether the slow tail converges and how much
   the thickness moves. The choice of λ is the user's.
3. Stopping tests: they are maxima over the variables, held up by those
   few weakly determined nodes, so fits run to `max_iterations`. Decide,
   after 2, whether stronger regularisation removes the tail or the tests
   need a robust form (still free of units and size); the measures are
   also not meaningful at nodes with almost no data, whose Gauss–Newton
   diagonal is mostly penalty.
4. The band's size: 1 600 voxels per node at 61 µm (outside and inside
   5 mm) drives the build (149 s), the memory (about 4.5 GB for the slab's
   band, 1.8× that for the whole TR_1 tibia) and every cost evaluation
   (about 3 s). Whether the model needs every voxel (`vol_voxel_stride`; the
   PSF is about 2 voxels and the noise correlated at 0.69 between
   neighbours) and 5 mm outside is the user's call; measure what stride 2
   changes.
5. Compact band storage (discussed with the user): 16 of the band's 20
   bytes per voxel are geometry fixed for a fit, and none of its fields
   needs more than 16 bits; the computation stays in f64, only storage
   changes. Proposed: intensity as u16 over the band's own range (rounding
   SD ≈ 0.01 against a noise SD of about 100: lossless in effect, and exactly
   the precision of int16 AIMs and integer HU), depth as u16 over
   `[−outside, inside]` (0.15 µm steps; not 8 bits, 39 µm, coarser than the
   changes we want to see), the two barycentric coordinates as u16 each,
   the axial normal component² as u8 (it only mixes σ_xy and σ_z, nearly
   equal on HR-pQCT): 9 bytes instead of 20 (slab 4.5 → 2 GB, whole tibia
   about 8 → 3.6 GB). Quantised inside the library at the band's build, so
   it does not depend on what type users pass: users cannot be expected to
   keep their images in int16 from import onwards. Validate as for a
   replacement: the same results within tolerance on the benchmark slab and
   the TR_1 slab (thickness and positions per node well below 1 µm, masks'
   Dice ≈ 1); check whether an evaluation (about 3 s on the slab, probably
   bound by erf/exp, not memory) gets faster at all.
   Other memory and efficiency to look at, measured before changing:
   - the band build's other full-box arrays: the distance to the
     rasterised boundary (f32 over the whole padded box, 1.9 GB on the slab)
     and the rasterised inside mask; both could be built and used per slab
     of slices, since the closest points already run slice by slice;
   - the smoothing operator (1.4 GB on the slab: 1 263 weights per node);
   - the example scripts: read AIMs without full-size float copies, crop
     before converting, and free the provided masks once cropped;
   - the per-node standard errors (reliability) at the HR-pQCT noise
     correlation, still the slowest output when it is wanted.
6. Then: the whole TR_1 tibia once; the plateau blocks; and later the
   accuracy and precision work (inside depth 7/10/12 mm, ρc, TR_1–3 with
   SimpleITK registration, the clinical CT's right knee).
7. Before committing: rerun the example scripts (knee, phantoms,
   benchmarks) whose results and provenance changed.

## 2026-11-26

### Refinement's level set (2026-11-27)

- **Found.** On the periarticular slab (468 × 1267 × 1573) the level set
  took 404 s to initialise and about 82 s per move, against 3–14 s per 1D
  fit: its band was the largest move allowed (half the sampling range plus
  3 voxels, 2.68 mm) while the moves were 0.01–0.5 mm, and the
  initialisation searched the closest point for every voxel of the box.
- **Decided** (exact): the band follows the moves, `max|s|` plus
  `√3 · grid_step + 1` voxels (the extraction reads `φ` at the corners of
  the cells the surface crosses); widened with exact distances only on the
  voxels it newly covers (found by a distance transform of the sign),
  narrowed by clamping; the level set tracks how far the surface moved since
  the clamped voxels were refreshed. Test: the same surfaces as a band wide
  enough for every move, over a sequence of varying moves in and out.
  Initialisation on the periarticular slab: 404 → 57 s.
- **Why the level set at all** (the user's question): moving the same
  vertices tangled the mesh over the knee's intercondylar notch from a hull
  start (`refinement_experiments.md`, removed in the docs cleanup; git
  `d40bf70`); it bought validity of the mesh, not accuracy. Next (user):
  re-binarising the moved mesh (with fold cleanup and winding-number
  rasterisation) as a cheaper remesh, a smoothing option between fits, and
  control-theory step sizes; never keeping the same nodes while they move.
- **Local method on HR-pQCT diverges.** From our start on the periarticular
  slab, every move was larger than the last (RMS 0.30 → 0.52 mm over 12
  moves, largest 2.5 mm), the untrusted fits grew from 510 to 24 500 and
  the re-extracted surface from 188 000 to 1.05 million nodes: unregularised
  per-node fits on HR-pQCT noise roughen the surface, which then gets
  rougher. Expected; the example shows the regularised method only.

### Compact band storage (2026-11-27)

- **Done.** The band stores 9 bytes per voxel: the intensity and the depth as
  16-bit codes (over the image's range in the band's box, and over the
  band's depth range), the two barycentric coordinates as 16 bits each, the
  axial normal component² as 8 bits (`fit/volumetric/quantise.rs`); the
  computation reads them back as f64. The build appends a group of 8
  slices at a time (rows in parallel) and reorders by face one array at a
  time.
- **Validated** on the tall slab against the f32 band (same settings, 100
  iterations each): thickness per node differs by a median of 0.002 µm, at
  most 0.05 µm; cortex Dice between the two 0.99999; Ct.Th, correlation and
  Dice against the reference identical. The band build on the benchmark:
  5.62 s (5.68 s before the build changes). Run time unchanged (575 against
  586 s: evaluations are not memory-bound).
- **Memory** of the diagnostic run: peak 13.6 GB (18.25 GB with the f32
  band and its double copy), of which 9.4 GB is the script's loading (the
  full-size AIM, its float crop, the provided masks) before any fit: the
  example scripts are now the larger part.
- **Not done:** the per-node data share run was stopped in the standard
  errors (correlated noise 0.73–0.77 per axis; the process at 18.5 GB): the
  data share needs a route that does not compute the standard errors.

### Priority (from the user)

- Robust convergence, free of scale and size, and run times feasible on
  normal hardware come first; matching the reference cortical mask, and
  tuning parameters for accuracy and precision, come later.

### The band's cost

- **Found.** On the tall slab (256 slices) the band holds 223 million voxels
  (1 600 per node at 61 µm, outside and inside 5 mm). Building it took 209 s,
  197 s of it the closest points: one 4-nearest-vertex search per voxel,
  measured (criterion, `benches/closest.rs`) at 4.4 µs per band voxel in
  the k-d tree itself, 91 % of a query. Inside a concave bone, at a few mm
  depth, many surface vertices are nearly equidistant, so any exact search
  must visit many of them. The whole TR_1 tibia is about 1.8 times the
  slab's height.
- **Decided** (exact, no change of results):
  - along a row of voxels, the previous voxel's 4th-nearest distance plus
    the step bounds the next one's search (−14 % per query);
  - the band stores 20 bytes per voxel instead of 36: the residual weight
    is computed from the depth and the face's reach where it is used (no
    measurable cost: linearise 130.7 ms against 131.8 ms), and the grid
    indices and the soft-tissue list are kept only for the reliability
    outputs;
  - the band is generated slice by slice (rows in parallel) into arrays in
    grid order, then reordered by face one array at a time (a stable
    counting sort), instead of sorting a copy of every voxel: at most 16
    bytes per voxel more than the band during the build (it was 36 more);
  - the per-node residuals of an intermediate fit come from one pass of
    residuals, not from the reliability's node moments (which also built
    each node's Gauss–Newton block and noise covariance, then dropped them).
- **Open.** The band's size itself (voxels per node) drives the build, the
  memory and every evaluation of the cost (3 s each on the slab). How finely
  the band must sample (the voxel stride, the outside depth) is a question
  of the model's information, for the user.

### Why the field fit stalled with the curvature penalty on

- **Found.** On the tall slab, every step after the second was rejected
  although the solver predicted a decrease. Comparing each rejected step's
  slope `gᵀs` with central differences of the cost, block by block, put the
  whole predicted decrease on `m` and `t` at two nodes with gradients near
  10¹⁵, where the steps (≈ 6·10⁻¹⁷ mm) are below the floating-point
  resolution of the values themselves (`x + s = x`): no representable step
  can follow such a gradient. The gradients came from the curvature
  penalty's Gaussian Laplacian: its rows are normalised by their second
  moment `M2`, and 24 of 139 820 rows (nodes of tiny fragments of the coarse
  surface, a few nearly coincident nodes with nothing else within 3σ) had
  `M2` down to 10⁻¹⁶ of `2σ²`, so entries of order 10¹⁶.
- **Decided.** A row whose second moment is below ¼ of `2σ²` (its
  neighbourhood samples less than half the scale) has no curvature residual,
  like the boundary rows of a cotangent Laplacian. A sound row has about 1,
  and an open edge or corner of a flat surface also about 1 (the second
  moment of a half or quarter disc about its centre is that of the whole
  disc). Tested on a grid with a far fragment.
- **Why.** Such a row measures curvature at a far finer scale than asked
  for, on nodes whose geometry is meaningless at that scale; dropping its
  residual changes nothing elsewhere.
- **Result.** With the curvature penalty on, the field fit now progresses
  (cost −17 %): per-node correlation with the reference's local thickness
  0.771, Ct.Th of the cortical mask 0.592 mm (reference 0.614), cortex Dice
  0.509, bone Dice 0.989. The best agreement yet, but it still ran to its
  100 iterations (632 s).

### Why the field fit crawls

- **Found.** The solver now logs, per trial step, where the step and
  gradient measures are largest and their median and 99th percentile. By
  iteration 27 the median step was 2·10⁻⁵ of the noise, the 99th percentile
  4·10⁻³, and the largest 8·10⁻², always on a position `m`, at a different
  node each time, and shrinking by only about 10 % per iteration. Each
  linear solve stopped at a relative residual of 0.1 after about 4 CG
  iterations, because the forcing term `min(0.1, √gmax)` follows the
  largest gradient, which stays above 10⁻². The slow modes are the ones the
  block-Jacobi preconditioner does not see: positions coupled over many
  nodes by the smoothing and curvature penalties. An outer iteration costs
  about 3 s of evaluation and a CG iteration about 0.25 s, so solving each
  step more accurately should be far cheaper than taking many more steps.
- **Decided.** A solver option, `cg_forcing` (the forcing term's cap,
  default 0.1 as before), to test tighter linear solves before touching the
  stopping tests.
- **Result.** With `cg_forcing = 0.01` each step took about twice the CG
  iterations, but the cost and the measures' percentiles followed the same
  path per outer iteration (iteration 16: cost 1.03006·10⁵, 99th
  percentile step 1.3·10⁻²; default, iteration 17: 1.03005·10⁵, 1.4·10⁻²).
  So the crawl is the Gauss–Newton iteration itself converging linearly on
  a few weakly determined positions, not inexact linear solves; the default
  stays 0.1.
- **Result.** Near-exact linear solves (relative residual 10⁻⁵, about 25 CG
  iterations each, so the preconditioned system is well conditioned)
  followed the same path too (iteration 10: 99th percentile step 3.5·10⁻²,
  against 3.2·10⁻² at iteration 11 by default): the slow convergence is the
  Gauss–Newton iteration's, not the linear solves'.
- **What the tail changes.** A fit stopped at 25 iterations against the one
  run to 100: bone Dice between them 0.99992, cortex 0.9936; thickness per
  node differs by a median of 0.25 µm, but by more than 10 µm at 4 % of the
  nodes (99th percentile 39 µm, largest 0.22 mm); Ct.Th 0.5949 against
  0.5917 mm. At the precision we want, that tail matters: the slow positions
  must converge faster, not be ignored by a looser stopping test.
- **Found, the cause.** The solver now logs every step's gain ratio ρ
  (actual over predicted decrease). It stayed above 1 throughout, rising
  from 1.46 to about 2. On a quadratic, a step from a curvature overestimated
  by a factor `c` gives `ρ = (2c − 1)/c` and shrinks the error by only
  `1 − 1/c` per iteration. The Gauss–Newton product itself is right (it
  equals the Hessian on exact data, `gauss_newton_product_is_the_hessian_on_exact_data`).
  But Gauss–Newton neglects the residuals' second derivatives, and here
  the residuals are mostly model misfit (trabecular texture), not noise, so
  `JᵀJ` overestimates the curvature along the steps and every step falls
  short by a similar factor. The phantom's global stage (one profile for a
  whole bone, also a large misfit) showed the same: ρ = 1.365 at every step,
  its gradient shrinking by exactly `1 − 1/c` = 0.36 per iteration.
- **Decided.** A line search along accepted steps with `ρ > 1`: the minimum
  of the parabola through the cost at `x`, its slope `gᵀs` and the cost at
  the trial, at `α* = −gᵀs / κ`, `κ = 2 (L(x′) − L(x) − gᵀs)` (from 1.25 to 10
  times the step), taken if it lowers the cost further. It costs one more
  evaluation only when `ρ > 1`, and is dimensionless and free of the
  problem's size. On the phantom's global stage: 12 → 9 iterations, the
  gradient falling 5–10× per iteration instead of 3×; the field stage
  (ρ ≈ 1) is unchanged.
- **Also found.** Early on, the largest step and gradient measures sat at
  nodes with almost no data (data weight 2000× below the mean): their
  measure divides by the node's data weight while the Gauss–Newton diagonal
  includes the penalties. Later the slowest nodes had ordinary weights, so
  this does not explain the crawl, but the measures are not meaningful at
  nodes the penalties alone determine. To revisit with the stopping tests.

### The volumetric start

- **Found.** On a 256-slice slab of TR_1 (15.5 mm, step 4), the volumetric
  field fit never leaves its start: the cost is unchanged over every step,
  with the PSF estimated (0.54 mm) or held at a sane value (0.107 / 0.147
  mm), and with or without the fibula excluded. Its start is one model
  fitted to all band voxels, whose thickness collapses to the lower bound
  (0.061 mm) when the band reaches deep into trabecular bone and the cortex
  varies a lot over the bone. On thin slabs (32 slices) the band holds only
  about 1 mm of bone and the fit progressed.
- **Decided.** A new option, `vol_init_from_local` (off by default until
  validated): every node starts from its own 1D local fit (trusted fits
  only, the rest from the 1D global model), and the PSF from the local fits'
  blur on thick-cortex nodes (`t > 3σ`, Treece & Gee 2015), less the profile
  smoothing in quadrature. The profiles are sampled once, for the density
  estimate and the start.
- **Why.** It keeps the volumetric model unchanged and replaces only the
  start; the 1D local fits see each node's own cortex.

- **Found next.** Started from the raw local fits, the field fit began at a
  cost of 4.7·10³¹, almost all of it the curvature penalty: single local fits
  are noisy from node to node, the fitted surfaces of such a start are very
  rough, and the curvature penalty (relative to the input surface) punishes
  that by many orders of magnitude more than the data. The PSF from the local
  fits was sane (0.139 / 0.126 mm; 127 000 of 140 000 fits trusted).
- **Decided.** Every node starts from the parameter-wise median of the
  trusted local fits within `σ_reg` of it (robust to outlier fits; untrusted
  nodes take their neighbours' median).
- **Result.** The best agreement with the reference so far on the tall slab:
  CBM `t` against the reference's local thickness per node correlates at
  0.69 (1D regularised: 0.41), median `t` 0.386 mm (reference 0.42), Ct.Th
  of the cortical mask 0.76 mm (0.61), cortex Dice 0.54; 262 s at step 4.
  But the field fit still stops after two accepted steps: every further step
  raises the cost even at strong damping, which a consistent Gauss–Newton
  linearisation cannot do near a reasonable start. Suspects: the curvature
  penalty's linearisation, or a preconditioner block that is not positive
  definite (which would also explain a conjugate-gradient step of zero).
  Testing with the curvature penalty off.
- **Result, curvature penalty off.** The field fit progresses (cost −18 %),
  and the agreement is the best yet: per-node correlation 0.76, Ct.Th of the
  cortical mask 0.59 mm against the reference's 0.61 mm, cortex Dice 0.51,
  PSF 0.114 / 0.115 mm. So the curvature penalty's linearisation fails at
  such a state: **next**, check its gradient and Gauss–Newton product against
  finite differences there.
- **Found.** Without the curvature penalty the solve crawled to its 100
  iterations (342 s) with negligible changes of the cost: the gate of the
  step test (gradient ≤ √gtol, a maximum over 560 000 variables) is held up
  by a few awkward nodes. **Next**: a robust form of the gate (still free of
  units and of the problem's size), re-validated on the benchmarks.

### Control: the 1D regularised method

- The 1D regularised method on the same slab converges out of the box (70
  iterations, 1881 CG, 270 s at step 4), so the trouble is specific to the
  volumetric start.
- Its median thickness at the nodes, 0.418 mm, matches the reference's
  local thickness at the same nodes (0.403 mm); per node they correlate at
  0.41. But 12.5 % of nodes fit `t > 2 mm` (up to 4.2 mm), where the model
  probably takes dense trabecular or subchondral bone for cortex; Ct.Th of
  the resulting cortical mask is therefore 2.04 mm (reference 0.61) and its
  Dice 0.28. To check on the plateau blocks, where the reference is
  trustworthy.

### Plateau blocks

- Blocks of 20 × 20 mm in-plane and 15.5 mm tall (10 mm below the plateau
  surface, 5.5 mm above for the outside sampling), centred on each plateau:
  located from the tibia mask's top-surface height map, split at the spines,
  each block at the middle of its half's upper half (leaving out a 10 mm band
  around the spines). The fibula lies at low x in TR_1, so the low-x block is
  the lateral plateau, the high-x block the medial one.

### Solver stopping tests

- **Found.** The global solver's tests were relative to the whole problem
  (gradient to the total residual, step to the norm of all variables, cost
  decrease to the total cost): they pass trivially at scale, and the step
  test depends on where the intensity units put their zero.
- **Decided** (with the user): per-variable, dimensionless tests. Variables
  are scaled by their Jacobian, residuals by the noise (the weighted RMS
  residual); the gradient test is the cosine between a variable's Jacobian
  column and its own residuals, the step test the largest change a step
  makes in any node's predictions in units of the noise, tested on accepted
  steps near a stationary point (`gradient ≤ √gtol`). Conjugate gradients
  stop in the preconditioned norm. Defaults `gtol = 1e-8`, `xtol = 1e-3`.
- **Validated.** Same accuracy on every method and size of the benchmark
  slabs (RMSE identical to 0.001 µm), run time −2 % overall; tests that the
  same problem in other length units (×1024) or with an intensity offset
  (+1000) takes the same iterations to the same solution.

### Penalty weights

- **Decided** (with the user): the regularised and volumetric penalties are
  weighted by the squared contrast `(ρc − ρs)²` (ρs from the global model),
  not by `ρc²` as in our 2025 paper, which depends on the intensities' zero.

### Units

- **Decided** (with the user): compile-time units for the physical
  computations, own types (`typenum` exponents): a generic
  `Quantity<Length, IntensityContrast>` and two affine types, `Intensity`
  (absolute) and `Position`. Done so far: the profile model and its
  gradient, and the penalty weights.

### Speed

- Standard errors: computed only for the returned result (refinement's
  intermediate fits use residuals only), and 21.6 → 2.3 s on the benchmark
  slab with correlated noise.
- Node count: `grid_step` extracts the surface on a coarser grid (step 4: 18×
  fewer nodes); the field fit on a 32-slice slab went from 158 s to 6.6 s.
- Starts (`bone_hulls`): each bone on its own padded box, 11 min → 2 min on
  TR_1, identical results.
- Refinement builds its level set at the first move (a refinement that ends
  after its first fit never needs it; it was 46 % of a diagnostic run).

### Test geometry

- Thin axial slabs are cut by caps where the image ends; voxels whose
  nearest surface is a cap are dropped, so a 32-slice slab holds only about
  1 mm of bone and cannot test band depth. Use tall slabs, or blocks
  centred on the plateaus (where the reference mask is trustworthy).
