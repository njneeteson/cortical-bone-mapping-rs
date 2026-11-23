# Design note: a volumetric cortical model

*Status: a working note, partly implemented. The aim is to think the model
through mathematically and plan a method that runs alongside the 1D Treece
et al. (2010) model, not to replace it. Steps 4–6 of the plan (section 8)
are implemented as `method="volumetric"` and the `neighbour_distance`
option; what was built, and where it departs from this note, is in
[method.md](method.md#volumetric-model). The site phantoms (step 1) and the
first two stages of the validation (step 8) are in
[volumetric_validation.md](volumetric_validation.md).*

The 1D method samples one density profile along the normal at every surface
node and fits a blurred three-level step to it. The image is 3D. This note
asks what a model of the *voxels* around the surface would look like, and
what it would buy: a better model of the cortex, judged by more accurate,
or at least more precise, measurements of thickness and surface position
than the 1D methods give. Corners and deep concavities such as the femoral
intercondylar notch are where the 1D assumptions fail most visibly (see
[refinement_experiments.md](refinement_experiments.md)), but they are one
test case, not the goal.

## 1. The 1D model and its assumptions

At a node `p` with inward normal `n`, the profile `y(x) = I(p + x n)` is
modelled as

```text
f(x) = ρs + (ρc − ρs) Φ((x − m + t/2) / σ) + (ρb − ρc) Φ((x − m − t/2) / σ)
```

with `Φ` the standard normal CDF. Implicit in this are:

1. **A planar layer.** The cortex is a slab of thickness `t` with its
   mid-line at depth `m`, perpendicular to `n`.
2. **The blur acts only along `n`.** For a planar slab, a Gaussian PSF with
   covariance `Σ` blurs the layer exactly like a 1D Gaussian of variance
   `σ² = nᵀ Σ n`. The in-plane part of the PSF does not matter.
3. **The normal is known.** It comes from the mesh, not from the image.
4. **Each profile is one line of samples.** Neighbouring voxels off the line
   are used only through the interpolation, and profiles of nearby nodes
   overlap inside the bone.
5. **No profile, no information.** A profile that never reaches bone (a
   surface bridging a concavity deeper than `inside`) gives an untrusted fit.
   The node then moves only with its neighbours.

Assumptions 1–3 fail at corners and in tight concavities. Assumption 5 is
why a bridged notch never fills unless its rim zips it shut (the
experiments).

## 2. The volumetric model: the same model evaluated at every voxel

The volumetric model changes only the inner loop of the refinement: how the
parameters of each node are fitted from the image. As now, the surface is
fixed during a fit, and the fit answers, for each node, how far along the
normal the middle of the cortex really is (`m`) and how thick it is (`t`).
The outer loop does not change: move each node by the fitted `m − t/2`,
remesh, recompute the normals, depths and neighbourhoods, and fit again.
Everything in sections 2, 4 and 5 is computed once per outer iteration from
the fixed surface. Nothing in a fit moves the surface.

Let `φ` be the signed distance to the current surface (negative inside),
as in the level-set refinement. For a voxel at `x` near the surface, let

* `d(x) = −φ(x)`: its depth below the surface (positive inside);
* `c(x)`: its closest point on the surface, and `n(x) = −∇φ(x)` the inward
  normal there;
* `θ(c) = (m, t, ρs, ρb)`: the model fields at that surface point.

Under assumptions 1 and 2, locally around `c(x)`, the blurred density at `x`
is the 1D model evaluated at the voxel's depth:

```text
I_model(x) = f( d(x) ; m(c), t(c), σ_n(x), ρs(c), ρb(c) ),     σ_n(x)² = n(x)ᵀ Σ n(x)
```

So **the planar 3D model is the 1D model with every voxel in a band around
the surface as a sample**, placed by its signed distance and owning the
parameters of its closest surface point. The cost is

```text
E = Σ_{x ∈ band} w(d(x)) (I(x) − I_model(x))²  +  regularisation of m, t, ρs, ρb over the surface
```

The band is the voxels with `−outside ≤ d(x) ≤ inside`. Because the surface
is fixed during a fit, the band and every voxel's depth are fixed too, and
the cost is smooth in the parameters. When the surface moves, the band moves
with it and new voxels come in, as new profile samples do now.

* **A tapering weight.** `w(d)` can fall off towards the ends of the band, so
  voxels far from the surface count less, and the voxels that enter or leave
  the band at a move change the cost only a little. The taper is in depth
  from the fixed surface, not from the fitted edge `m − t/2`. Weights that
  depended on the fit would let it lower the cost by moving the cortex away
  from the voxels that disagree with it.
* **Other bones get zero weight.** Voxels inside another bone are excluded,
  as profile samples are now (`Profiles::exclude`), so the tibia has no
  influence on the femur's parameters, and the outside of the band needs no
  cap. Voxels in the gap within about `2σ` of the other bone still carry its
  blurred cortex. If that matters, dilate the exclusion mask by that much
  (the 1D method has the same issue).

Compared with profiles:

* **Real samples, no double counting.** Each voxel is used once, at its true
  position, with no interpolation. The closest-point assignment partitions
  the band by the closest triangle of the surface.
* **No hand-made normals.** The normal of a sample is `∇φ` at the sample,
  which varies across a cell. At a corner, the voxels on either side of the
  corner see different normals, which a single profile cannot.
* **The PSF becomes a global property.** `σ_n` follows from one scanner PSF
  and the local normal (below), not from a `σ` fitted per node.
* **The same solver structure.** Each sample depends on the parameters of
  the three vertices of its closest triangle (section 4), so the Jacobian is
  block-sparse. The global solver forms per-node Gauss–Newton blocks once
  per iteration and runs matrix-free PCG over the blocks, never over the
  samples. The volumetric model adds one block per mesh edge, and a shared
  PSF is a global parameter, handled as the global `σ` is now. What changes
  for the data is that the "profiles" become ragged: a variable number of
  samples per triangle, each with its own depth.

### The point-spread function

The PSF is modelled on how a CT scanner blurs: a Gaussian whose covariance
is fixed in the scanner's frame,

```text
Σ = R diag(σ_xy², σ_xy², σ_z²) Rᵀ
```

`σ_xy` is the in-plane blur (focal spot, detector, reconstruction kernel)
and `σ_z` the blur along the table (the slice sensitivity profile:
collimation, pitch, reconstructed slice thickness). `R` is the rotation from
the scanner's axes to the image's: the identity for an axial acquisition, a
rotation about `x` for a tilted gantry (from the DICOM gantry/detector tilt),
and any reorientation of the image since. `R` is known, not fitted. For a
voxel whose normal is `n` in the scanner frame,

```text
σ_n² = σ_xy² (1 − n_z²) + σ_z² n_z²
```

So two global parameters replace the per-node `σ` of the 1D methods. A
surface facing along the table is blurred by `σ_z`, one facing sideways by
`σ_xy`. This is the blur model of Treece & Gee (2015, appendix A),
`σ = √((σ_i cos α)² + (σ_o sin α)²)` for a line at angle `α` to the slices
(`n_z = sin α`). They estimate `σ_i` and `σ_o` for each scan by regressing
the blurs of per-location 1D fits on `α`; here the two are parameters of
the model, fitted with everything else.

* **Identifiability.** Both are determined only if the surface has normals
  in both directions. That holds for a whole bone, but not for a short shaft
  segment, whose normals are all nearly in-plane. There `σ_z` should be
  fixed. HR-pQCT is close to isotropic, so `σ_z = σ_xy` is a useful option.
* **Voxel averaging.** A voxel's value is the image averaged over its
  footprint, which adds `h_i²/12` per axis to `Σ` (`h_i` the voxel size).
  The fitted values absorb it, so they are the effective blur of the image,
  not the scanner's PSF. Subtract it before comparing with a PSF measured on
  a phantom.
* **What this leaves out.** Real reconstruction kernels are not Gaussian
  (sharp kernels overshoot at edges), and the blur varies across the field
  of view. Both are open questions (section 7).

### Cost

Clinical CT, femur: 4 mm outside plus 8 mm inside over about 30,000 mm² of
surface at 0.26 mm³ per voxel is about 1.4 M voxels. The current profiles
are 48,000 nodes × 49 samples ≈ 2.4 M samples, so the cost is similar.
HR-pQCT at 61 µm: a 2 + 4 mm band over about 5,000 mm² is about 140 M
voxels. That is a lot of rows, but not a lot of unknowns. The unknowns are
still the parameters of the nodes (or of coarse centres, section 4), and
each voxel touches only the three vertices of its triangle, so the
Gauss–Newton matrix has the sparsity of the mesh however many voxels there
are. The voxels enter only when the blocks and the gradient are formed,
once per Gauss–Newton iteration: a streaming pass whose time is linear in
the voxel count and whose memory scales with the nodes. A voxel adds a
rank-one update to the 6 vertex and edge blocks of its triangle: about 150
multiply-adds, or about 2 × 10¹⁰ for the whole band, which is seconds on a
multicore machine.

What needs care is memory and the order of operations, not the size:

* **Blocks, not voxels, in PCG.** As now, the products inside PCG use the
  blocks, so PCG never touches the voxels. Only forming the blocks does.
  Form them in spatial chunks, so a thread's voxels share their vertices and
  the scatter does not contend.
* **Store the minimum per voxel.** Value, depth, closest triangle and two
  barycentric coordinates are about 16 bytes, or 2–3 GB.
* **The closest-point query.** Every band voxel needs its closest triangle
  after each remesh: 140 M queries, or a narrow-band closest-point transform.
  Tens of seconds, but the step to watch.

The reasons to subsample at HR-pQCT are cost against benefit, not
feasibility. Neighbouring voxels within the PSF are strongly correlated, so
140 M voxels carry far fewer independent measurements, and subsampling the
band by 2 in each direction probably loses little. And at this resolution
trabeculae and cortical pores are resolved, so a constant `ρb` is a poor
model deep inside the bone and the inside band should be shorter anyway
(the 1D model has the same problem). For comparison, a mesh near voxel
spacing has hundreds of thousands of nodes, and their profiles already give
tens of millions of samples, so the volumetric model is a small multiple of
the current cost, not a different regime.

## 3. Bone beyond the band: region forces

This section is about what a fit says, not about a new way of moving the
surface. The outer loop always moves the surface by the fitted `m − t/2`
(section 2).

With the surface fixed, changing a node's periosteal offset `m − t/2` by `v`
moves the modelled periosteal edge by `v` along the normal. So the
derivative of `E` with respect to the offset field is the shape derivative
of the data term with respect to the edge. Write the unblurred model as
indicators of the modelled periosteal region `P` and endosteal region `B`:

```text
D(x) = ρs + (ρc − ρs) χ_P(x) + (ρb − ρc) χ_B(x),      I_model = G_Σ * D
```

For a normal displacement `v` of the modelled periosteal edge, the first
variation is a surface integral

```text
δE = −2 ∫_S v(s) (ρc − ρs) (G_Σ * r)(s) ds,      r = w (I − I_model)
```

That is, the residual image blurred by the PSF and read on the edge, times
the height of the periosteal step. It is the Chan–Vese (2001) region force
with a blurred step model. Through the basis of section 4 it becomes the
gradient of `E` with respect to each node's offset. Its sign is what matters
for a bridged notch. At the start of a fit, with the modelled edge on a
surface that bridges a gap:

* just inside the edge the model says cortex (`ρc`) and the image says soft
  tissue, so `r < 0` there;
* the gradient pushes the offset inward, with no need for any voxel to reach
  the bone across the gap.

That is the direction at the start. A fit run to convergence keeps going:
with nothing but soft tissue in the band, the cortex is pushed past the
inner end of the band, the edge lands at the inner bound, and the fit is
called untrusted. The volumetric model does not change this. Over a bridge,
every voxel in the band is soft tissue whichever node it belongs to, as
along a profile. The volumetric model helps at the rim of the notch, where
the walls are in the band, not over the middle of the bridge.

### A rule for fits at the inner bound

The region force shows that this outcome is not uninformative: it says the
bone is *further in than the band reaches*. Treating "edge at the inner
bound with a soft-tissue band" as a confident inward move of the maximum
step, rather than as no information, gives the refinement the region force
without changing the outer loop. The rule is the same for profiles and for
the volumetric model. It is a cheap experiment with the existing 1D methods.

## 4. Parameter fields on the surface

Section 2 gives each voxel the parameters of its closest node. That
partitions the band into cells, and `I_model` jumps across the cell
boundaries wherever neighbouring nodes differ. The model chosen here keeps
the geometry from the fixed surface and makes only the parameters smooth,
with basis functions centred on the surface.

The simplest basis is the mesh's own. The closest point `c(x)` lies in a
triangle, and the fields there are interpolated linearly from its three
vertices, with barycentric weights `β_v(x)`. These are hat functions:
compactly supported basis functions centred on the nodes, each reaching only
its one-ring. To spread the parameters further, compose them with a coarse
basis of radial functions of surface distance (section 5) centred on
control points `q`, with weights `A_vq`. That is the interpolation matrix `A`
of the multigrid method, built with surface distance instead of Euclidean:

```text
θ(x) = Σ_{v ∈ tri(x)} β_v(x) θ_v,      θ_v = Σ_q A_vq θ_q
I_model(x) = f( −φ(x) ; θ(x), σ_n(x) )
```

With `A = I` the fields are one value per node, as in the regularised
method. Depth and normal still come from the fixed surface, which represents
corners and concavities exactly, so the basis functions never decide the
geometry. They only spread the parameters.

* **No leaks across thin structures.** The hat functions touch only one
  triangle, and `A` built with surface distance does not couple the two
  walls of the notch or the two cortices of a plate. A kernel centred on the
  voxel with Euclidean distance would do both.
* **Coarse centres.** Control points (say one per 2 mm) separate the
  resolution of the parameters from the resolution of the surface: fewer
  unknowns, and the basis does part of the regularisation.
* **The solver carries over.** A voxel's Gauss–Newton contribution is
  `(β βᵀ) ⊗ (∇f ∇fᵀ)`, which couples the three vertices of its triangle. So
  the data term's Gauss–Newton matrix `B` has a `5 × 5` block per vertex and
  one per mesh edge, about four times the blocks of the current block
  diagonal. It is formed once per iteration by one pass over the voxels.
  With control points the products are `Aᵀ B A`, as in the multigrid method
  now. The block-Jacobi preconditioner uses the vertex blocks, also as now.
* **Bounded, not clamped.** Both current solvers handle the parameter bounds
  inside the optimisation. The global solver is a projected
  Levenberg–Marquardt with an active set: a variable on a bound whose
  gradient points outward is held for the step, PCG solves over the free
  variables only, and each trial point is projected into the box *before*
  its cost decides whether the step is accepted. So the other parameters are
  always fitted given the bounded ones, and a bound cannot leave them
  degenerate, as clamping a finished fit would. The local solver does the
  same per node. The volumetric fit uses the same solver, so this carries
  over. One property must be kept: the barycentric weights and the
  normalised Gaussian weights of `A` are non-negative and sum to one, so
  `θ(x)` at every voxel is a convex combination of the fitted values, and
  bounds on those values bound the parameters at every voxel. A basis with
  negative weights (interpolating RBFs, for example) would lose this. The
  bounds are boxes per parameter; there is no ordering constraint such as
  `ρb > ρs`, here or in the 1D methods. (The outer loop's `max_step` caps the
  surface move, not a fitted parameter.)

### Alternatives considered

* **Blending predictions instead of parameters.** Give each node its own
  planar step, `f_i(x) = f(n_iᵀ(x − p_i); θ_i)`, and blend the predictions,
  `I_model = Σ_i w_i f_i`: a partition-of-unity model in the spirit of
  Ohtake et al. (2003). It needs no level set, and the normals `n_i` can be
  fitted. But it rounds corners. Just past the edge of face A at a convex
  corner, a voxel lies below the extension of plane A, so `f_A` says bone
  where `f_B` says soft tissue, and the blend sits in between. Narrowing the
  kernel to fix this brings back the closest-point cells. It fails exactly
  where the volumetric model is meant to help.
* **Kernel-weighted local fits ("thick profiles").** Each node fits its own
  planar model to the voxels around it, weighted by an anisotropic kernel:
  width `h` along the surface, the usual `[−outside, inside]` band along the
  normal. A profile is the limit of a tangential kernel of zero width. This
  is not a volumetric model, since voxels are shared between nodes, but it
  is a cheap upgrade of the 1D sampling: real voxels instead of
  interpolated samples, noise averaged over a disc instead of a line, and
  room to fit the normal. The trade-off is planar bias over the radius `h`
  against noise.
* **Basis functions for the surface itself.** Write
  `φ(x) = Σ_i λ_i ψ(|x − p_i|) + polynomial`, an RBF implicit surface
  (Carr et al. 2001) or parametric level set (Aghasi et al. 2011). The
  region force of section 3 then becomes a gradient with respect to `λ_i`,
  with no reinitialisation. But that fits the surface and the parameters
  together, which merges the inner and outer loops. This note keeps them
  apart (section 2).

## 5. Distance along the surface

The regulariser `ω` currently weights the `k` nearest neighbours of a node
in Euclidean distance (k-d tree). What is wanted is distance along the
surface: two points on opposite walls of the notch, or on the two cortices
of a thin plate, are close in space but far apart on the surface, and their
thicknesses should not be smoothed together. The coarse basis `A` of
section 4 needs the same distance.

**Truncated Dijkstra from every node.** The mesh already stores its
one-ring adjacency (`Mesh::adjacency`). From each node, run Dijkstra over
the edge graph with Euclidean edge lengths and stop when the frontier passes
a radius `R`. Each search visits only the `k` or so nodes inside `R`, so
building the whole operator costs about `O(N k log k)` and runs in parallel
over nodes. For 48,000 nodes and `k ≈ 50` that is well under a second. The
output is a list of `(node, distance)` pairs per row, which is exactly what
the sparse row matrices are assembled from now, and the weights `g(d)` do
not change: only the metric does.

* **A radius, not "N nearest" or "N steps".** Hop counts from powers of the
  adjacency matrix depend on the mesh density, and their fill-in grows fast.
  A distance cutoff with the existing weighting keeps the operator's meaning.
* **Edge paths are too long.** A path along edges zigzags, so it
  overestimates the geodesic. On a regular mesh with `2k` edge directions
  the worst case, midway between two directions, is `1/cos(π/2k)`: 8% for
  8 directions, 15% for 6 (a square grid split by one diagonal, as marching
  cubes produces locally) and 41% for 4. Averaged over directions it is a
  few percent. For a Gaussian weight that is a narrower, slightly
  anisotropic kernel. If it matters, local fast marching on the triangles (Kimmel &
  Sethian 1998) is exact to first order. The heat method (Crane et al. 2013)
  solves from one source over the whole mesh, which suits this badly.
* **Separate components separate themselves.** Two bones in one mesh are at
  infinite distance, so they never couple. The Euclidean neighbours cannot
  guarantee that.
* **No help for a bridged notch.** If the surface spans the gap, the two
  walls are close along the surface through the bridge. Surface distance
  keeps the walls apart only once the surface follows the bone. It
  complements the region force of section 3; it does not replace it.
* **Rebuild after every remesh.** The neighbourhoods belong to the mesh, so
  each refinement iteration that re-extracts the surface rebuilds them. That
  is cheap.
* **The other Euclidean neighbourhoods.** The multigrid interpolation `A`,
  the minimum separation of the control points, and the extension of the
  level-set speed from the nearest vertices all use the k-d tree too. Each
  could leak across the notch in the same way and could use the same
  routine.

**Distances between nodes only.** Surface distance is needed between nodes
(for `ω`) and between nodes and control points (for `A`), never per voxel.
Voxels reach the basis through the barycentric weights of their triangle.
So one routine serves the regulariser, the multigrid interpolation and the
volumetric model, and closes the leak across thin structures in all three.

For a Python prototype, `scipy.sparse.csgraph.dijkstra(G, indices=batch,
limit=R)` does truncated multi-source Dijkstra on the edge graph. It returns
dense rows, so run it in batches of sources and keep only the finite
entries. VTK's `vtkDijkstraGraphGeodesicPath` answers one source–target pair
at a time and is not suited to this.

## 6. Reliability of the fitted values

The 1D method calls a fit untrusted from its outcome: an edge at a bound, or
a fit that failed. The volumetric fit is one global problem, so reliability
is a property of each node's parameters within it: how much the data
determined them, and how well the model fits there. The options:

1. **Local standard errors from the data blocks.** The data-only vertex
   blocks `B_v` are formed at every iteration anyway. The standard error of
   `t` at node `v` is `s √[B_v⁻¹]_tt`, with `s` the noise level (from the
   residuals). Inverting the block, rather than taking `1/(B_v)_tt`,
   accounts for the trade-off of `t` with the node's other parameters (with
   `σ_n`, most of all, in thin cortices). Implementation: one `5 × 5` inverse
   per node at the end of the fit. Trade-offs: it ignores the edge blocks
   (neighbours sharing voxels) and the regulariser, so it is the error of a
   node fitted from its own voxels alone. That overstates the error where
   neighbours help, but it is the honest answer to "what did the data say
   here". CT noise is correlated over a few voxels by the reconstruction
   kernel, so the voxels are not independent and the formula understates
   the error. Correct it by a factor from the noise autocorrelation,
   measured in a homogeneous region (soft tissue): in effect, the number of
   independent voxels.
2. **Data share.** For each fitted value, the fraction of its curvature that
   comes from the data: `B_tt / (B_tt + κ (ωᵀω)_tt)`, from the diagonals the
   solver already computes for scaling and preconditioning. Near one, the
   data decided the value; near zero, the regulariser filled it in.
   Trade-offs: cheap, dimensionless, and directly the question "did the
   regulariser decide this", but it depends on `κ`, uses only diagonals, and
   says nothing about whether the model is right.
3. **Full posterior variance.** The diagonal of `(JᵀJ + R)⁻¹` for the
   regularised problem, with all the coupling. Estimate it stochastically
   (Bekas et al. 2007): solve with the existing PCG for some tens of random
   probe vectors. Trade-offs: the right number to report beside a map, and
   accurate up to the estimator's noise, but each probe is a full linear
   solve, so it costs tens of fits' worth of PCG. Better for validation than
   for routine use.
4. **Local misfit.** The RMS residual over a node's voxels, relative to the
   noise. Options 1–3 assume the model holds; this finds where it does not:
   porosity, a thin plate within the blur of its other cortex, a surface
   bridging a gap, metal. Cheap. A complement to 1–3, not a replacement.
5. **Bound activity.** Parameters on a bound at convergence, as now, and the
   nodes whose move came from the inner-bound rule (section 3) rather than
   from a fit.
6. **Resampling.** Refit with resampled residuals (a wild bootstrap), or on
   repeated noise realisations of a phantom, and take the spread of the
   thickness. Model-light and complete, but many fits. For validation only.
   Repeat scans (section 8) are the real version.

A reasonable default: report 2 and 4 for every node, and 1 as a standard
error with the noise-correlation correction. Use 3 and 6 in validation, to
check that 1 is calibrated. A trust flag like the 1D one then combines them:
data share above a threshold, misfit within a few times the noise, and no
parameter on a bound.

## 7. Open questions

1. **Curvature.** Blurring a curved layer is not blurring a planar one: to
   first order the blurred edge of a curved surface shifts by a term in
   `κ σ²` (mean curvature `κ`). The planar model ignores this, and so does the
   1D model. Is it worth adding to `I_model` (κ from `φ`), and at what
   curvatures does it matter at clinical resolution?
2. **Thin plates.** A voxel may be closest to one cortex of a thin plate
   but within the blur of the other. The 1D model has the same problem along
   the profile. The band should not reach past the medial axis, so the
   inside depth should be capped at the local half-thickness of the bone
   (from `φ`). A neighbouring bone needs no cap, since its voxels are
   excluded (section 2).
3. **The basis for the parameter fields.** Section 4 settles on barycentric
   weights on the closest triangle, optionally composed with a coarse basis
   `A` built on surface distance. Still open: the width of `A`'s kernel
   relative to the control-point spacing, and whether coarse centres can
   stand in for most of the explicit regularisation.
4. **ρc.** The same question as in 1D: fixed, estimated, or a field. With a
   global PSF fitted from all voxels, a global `ρc` may be better
   determined than from profiles.
5. **The shape of the PSF.** A Gaussian is the 1D model's assumption too.
   Sharp reconstruction kernels overshoot at edges, which a Gaussian cannot
   represent, and the blur varies across the field of view. Is a slowly
   varying `σ_xy` field, or a kernel with a negative lobe, worth the extra
   parameters?
6. **Topology.** The level set allows topology changes, and the outer loop's
   moves could split the notch region or the condyles in pathological
   cases. It may need a topology-preserving level set.

## 8. Plan

The aim is a better model of the cortex, judged by the accuracy and
precision of thickness and surface position against the 1D methods. Each
step says what decides it.

1. **Site phantoms.** Synthetic phantoms with realistic geometry for each
   site the method should apply to: for example the proximal femur, the
   knee (distal femur with its intercondylar notch, and the proximal tibia)
   in clinical CT, and the distal radius and tibia in HR-pQCT. Shape,
   cortical thickness, densities, PSF and noise come from the literature,
   and the truth is exact. Extend `BonePhantom` and `JointPhantom`.
   Neighbouring bones never touch, and any breaks are small. Generate several
   noise realisations per phantom, so precision can be measured as well as
   accuracy.
2. **Prototype in Python**, with the library's functions. Evaluate `I_model`
   and its residuals on a fixed surface, with parameters from a 1D fit (for
   the evaluation only), and compare the residual maps with those of the 1D
   model. Check the region-force direction (section 3) in the notch.
   Decided by: whether the volumetric residuals are smaller and less
   structured where the 1D model's planar assumption fails (corners,
   curvature, oblique slices).
3. **The inner-bound rule** (section 3) in the existing refinement. Decided
   by: surface error inside a notch mask on the knee phantom.
4. **Surface distance for the regulariser** (section 5). Truncated Dijkstra
   on the mesh adjacency, as an option beside the k-d tree in the smoothing
   operator, then in `A` and the control points. It stands on its own and is
   tested with the 1D methods. Decided by: thickness error near the notch
   walls and on thin plates, with Euclidean and with surface
   neighbourhoods.
5. **Rust: a `Method::Volumetric` beside the existing methods.** A new,
   self-contained inner loop. It does not start from a 1D fit. Its own
   initialisation is a fit with every parameter global (one model for all
   band voxels), whose fields are then released, as the 1D global method's
   initialisation does. Band voxels get their closest triangle, barycentric
   coordinates and depth from the fixed surface, and are stored ragged by
   triangle: value, depth, weight and barycentric coordinates. Excluded
   voxels (other bones) are dropped. The global solver gains edge blocks,
   optionally with `A` from step 4. The PSF is global (`σ_xy`, `σ_z`,
   section 2), the bounds are as now (section 4), and the reliability
   outputs are those of section 6.
6. **Refinement.** The outer loop is unchanged: move by the fitted
   `m − t/2`, with the inner-bound rule, remesh, refit. Only the inner fit
   differs between the methods being compared.
7. **Thick profiles, optionally** (section 4, alternatives): kernel-weighted
   local fits as a sampling option for the 1D methods.
8. **Validation, in stages.** The same data and settings for every method.
   1. *Phantoms.* Bias and RMS error of thickness and surface position
      against the truth, binned by true thickness relative to the PSF (the
      thin-cortex regime is where the methods should differ). Precision as
      the spread over noise realisations. Convergence and run time. Also
      whether the volumetric method gains anything from a 1D fit as its
      starting point.
   2. *Real clinical CT* (the knee example of
      [clinical_ct.md](clinical_ct.md)). No truth: agreement between the
      methods, residual maps and reliability maps.
   3. *A repeat-scan precision dataset.* Per-node precision of thickness
      (RMS SD and RMS CV%, Glüer et al. 1995) after mapping repeat scans
      to a common surface, which needs a registration step. Every method in
      this repository is compared with the others and with the
      implementation in the Bonelab package.

## References

> G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
> thickness and cortical bone density using clinical CT*, Medical Image
> Analysis 20(1):249–264, 2015.

> G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
> cortical bone thickness measurement from clinical CT data*, Medical Image
> Analysis 14(3):276–290, 2010.

> T. F. Chan, L. A. Vese, *Active contours without edges*, IEEE
> Transactions on Image Processing 10(2):266–277, 2001.

> S. Osher, J. A. Sethian, *Fronts propagating with curvature-dependent
> speed: algorithms based on Hamilton–Jacobi formulations*, Journal of
> Computational Physics 79(1):12–49, 1988.

> Y. Ohtake, A. Belyaev, M. Alexa, G. Turk, H.-P. Seidel, *Multi-level
> partition of unity implicits*, ACM Transactions on Graphics 22(3):463–470,
> 2003.

> J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright,
> B. C. McCallum, T. R. Evans, *Reconstruction and representation of 3D
> objects with radial basis functions*, Proceedings of SIGGRAPH 2001,
> 67–76.

> A. Aghasi, M. Kilmer, E. L. Miller, *Parametric level set methods for
> inverse problems*, SIAM Journal on Imaging Sciences 4(2):618–650, 2011.

> C. Bekas, E. Kokiopoulou, Y. Saad, *An estimator for the diagonal of a
> matrix*, Applied Numerical Mathematics 57(11–12):1214–1229, 2007.

> C.-C. Glüer, G. Blake, Y. Lu, B. A. Blunt, M. Jergas, H. K. Genant,
> *Accurate assessment of precision errors: how to measure the
> reproducibility of bone densitometry techniques*, Osteoporosis
> International 5(4):262–270, 1995.

> R. Kimmel, J. A. Sethian, *Computing geodesic paths on manifolds*,
> Proceedings of the National Academy of Sciences 95(15):8431–8435, 1998.

> K. Crane, C. Weischedel, M. Wardetzky, *Geodesics in heat: a new approach
> to computing distance based on heat flow*, ACM Transactions on Graphics
> 32(5):152, 2013.
