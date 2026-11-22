# Design note: a volumetric cortical model

*Status: a working note. Nothing here is implemented. The aim is to think
the model through mathematically and plan a method that runs alongside the
1D Treece et al. (2010) model, not to replace it.*

The 1D method samples one density profile along the normal at every surface
node and fits a blurred three-level step to it. The image is 3D. This note
asks what a model of the *voxels* around the surface would look like, and
what it would buy, especially at corners and deep concavities such as the
femoral intercondylar notch (see
[refinement_experiments.md](refinement_experiments.md)).

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

Take the surface as the zero level of a signed distance `φ` (negative
inside), as in the level-set refinement. For a voxel at `x` near the
surface, let

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

The band is the voxels with `−outside ≤ d(x) ≤ inside`, and `w` is the
residual weighting of the 1D method.

Compared with profiles:

* **Real samples, no double counting.** Each voxel is used once, at its true
  position, with no interpolation. The closest-point assignment partitions
  the band into the Voronoi-like cells of the surface nodes.
* **No hand-made normals.** The normal of a sample is `∇φ` at the sample,
  which varies across a cell. At a corner, the voxels on either side of the
  corner see different normals, which a single profile cannot.
* **The PSF becomes a global property.** `σ_n` follows from one scanner PSF
  `Σ` (in-plane `σ_xy` and axial `σ_z`, fitted globally) and the local
  normal. That matches treating the blur as a property of the imaging
  system, with anisotropic voxels and slice thickness handled correctly.
* **The same solver structure.** Each sample depends on the parameters of
  one node (or a few, if the fields are interpolated), so the Jacobian is
  block-sparse exactly as it is now. The regularised Gauss–Newton/PCG
  solver carries over. What changes is that the "profiles" become ragged:
  a variable number of samples per node, each with its own depth.

### Cost

Clinical CT, femur: 4 mm outside plus 8 mm inside over about 30,000 mm² of
surface at 0.26 mm³ per voxel is about 1.4 M voxels. The current profiles
are 48,000 nodes × 49 samples ≈ 2.4 M samples, so the cost is similar.
HR-pQCT at 61 µm: a 2 + 4 mm band over about 5,000 mm² is about 140 M
voxels. That is a lot of rows, but not a lot of unknowns. The unknowns are
still the parameters of the nodes (or basis centres, section 4), and each
voxel touches only the `K` centres in its support, so `JᵀJ` has the
sparsity of the regulariser however many voxels there are. The voxels enter
only when `JᵀJ` and `Jᵀr` are assembled, once per Gauss–Newton iteration:
a streaming pass whose time is linear in the voxel count and whose memory
scales with the centres. With `K ≈ 10` and four parameters, a voxel adds a
rank-one update to `K²` blocks of `4 × 4`: about 800 multiply-adds, or about
10¹¹ for the whole band, which is seconds on a multicore machine.

What needs care is memory and the order of operations, not the size:

* **Assemble `JᵀJ`, do not run matrix-free.** Matrix-free, every PCG
  product is a pass over all the voxels, and there are hundreds of them.
  Assembled, PCG never touches the voxels. Assemble in spatial chunks, so a
  thread's voxels share their centres and the scatter does not contend.
* **Store the minimum per voxel.** Value, depth, closest triangle and two
  barycentric coordinates are about 16 bytes, or 2–3 GB. Storing the basis
  weights as well is about 90 bytes, or 13 GB. The weights at `c(x)` can be
  interpolated from the triangle's vertices, whose weight rows the
  surface-distance routine of section 5 already computes.
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

## 3. Moving the surface: region forces

The refinement moves the surface by the fitted `m − t/2`. In the volumetric
formulation the surface can instead move along the gradient of `E` with
respect to `φ`. Write the unblurred model as indicators of the periosteal
region `P` (inside the surface) and the endosteal region `B` (inside the
endosteal surface):

```text
D(x) = ρs + (ρc − ρs) χ_P(x) + (ρb − ρc) χ_B(x),      I_model = G_Σ * D
```

The shape derivative of the data term for a normal velocity `v` of the
periosteal surface is a surface integral

```text
δE = −2 ∫_S v(s) (ρc − ρs) (G_Σ * r)(s) ds,      r = w (I − I_model)
```

That is, the residual image blurred by the PSF and read on the surface,
times the height of the periosteal step. It is the Chan–Vese (2001)
region force with a blurred step model. Its sign is what matters for a
bridged notch:

* just inside a surface that bridges a gap, the model says cortex (`ρc`)
  and the image says soft tissue, so `r < 0` there;
* the force moves the surface inward, *wherever it is*, with no need for any
  sample to reach the bone across the gap.

In other words, the volumetric model turns "this profile shows no bone"
from an untrusted fit into a direction. The force fades as the surface
approaches the true periosteum, where `r` changes sign across the step. The
same holds for the endosteal surface through `t`.

### A lesson for the 1D method

The same reasoning applies to profiles. When a profile is all soft tissue,
the least-squares 1D fit pushes the cortex past the inner end, the edge
lands at the inner bound, and the fit is called untrusted. That outcome is
not uninformative: it says the bone is *further in than the profile
reaches*. Treating "edge at the inner bound with a soft-tissue profile" as a
confident inward move of the maximum step, rather than as no information,
would give the 1D refinement the same region force. It is a cheap
experiment that does not need the volumetric model.

## 4. Parameter fields on the surface

Section 2 gives each voxel the parameters of its closest node. That
partitions the band into cells, and `I_model` jumps across the cell
boundaries wherever neighbouring nodes differ. The model chosen here keeps
the geometry from the level set and makes only the parameters smooth, with
radial basis functions centred on the surface:

```text
θ(x) = Σ_i w_i(x) θ_i,      w_i(x) = g(s(c(x), p_i)) / Σ_j g(s(c(x), p_j))
I_model(x) = f( −φ(x) ; θ(x), σ_n(x) )
```

Here `s` is the distance along the surface (section 5) and `g` a compactly
supported kernel (a truncated Gaussian, as in the regulariser, or a
Wendland function). Depth and normal still come from `φ`, which represents
corners and concavities exactly, so the basis functions never decide the
geometry. They only spread the parameters.

* **Distance from `c(x)`, not from `x`.** A kernel centred on the voxel,
  with Euclidean distance, reaches across the notch to the nodes of the
  opposite condyle wall, and across a thin plate to the other cortex.
  Measured from the closest surface point and along the surface, it does
  not.
* **The centres need not be the mesh nodes.** A coarser set of centres
  (say one per 2 mm) separates the resolution of the parameters from the
  resolution of the surface: fewer unknowns, and the basis does part of the
  regularisation. With the multigrid control points as centres this is the
  interpolation matrix `A` of the global solver, evaluated at `c(x)` instead
  of at the nodes.
* **The Jacobian widens, but stays sparse.** A voxel's row has a block for
  each centre in its support, `∂I_model/∂θ_i = w_i(x) ∂f/∂θ`, instead of
  one block. With compact support the regularised Gauss–Newton/PCG solver
  carries over unchanged.

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
  with no reinitialisation. This changes how the surface is represented,
  not how the parameters are attached to it, so it is independent of the
  choice above.

## 5. Distance along the surface

The regulariser `ω` currently weights the `k` nearest neighbours of a node
in Euclidean distance (k-d tree). What is wanted is distance along the
surface: two points on opposite walls of the notch, or on the two cortices
of a thin plate, are close in space but far apart on the surface, and their
thicknesses should not be smoothed together. The parameter fields of
section 4 need the same distance.

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
* **Edge paths are slightly too long.** A path along edges zigzags, so it
  overestimates the geodesic, typically by a few percent and up to about 8%
  on a regular mesh. For a Gaussian weight that is a slightly narrower
  kernel. If it matters, local fast marching on the triangles (Kimmel &
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

**One routine for both uses.** For the parameter fields, the source is the
closest point `c(x)`, which lies inside a triangle. Seed the same truncated
Dijkstra from the triangle's three vertices, starting at their distances
`|c(x) − v|`. The regulariser and the volumetric model then share one
surface-distance routine, and the leak across thin structures is closed in
both places at once.

For a Python prototype, `scipy.sparse.csgraph.dijkstra(G, indices=batch,
limit=R)` does truncated multi-source Dijkstra on the edge graph. It returns
dense rows, so run it in batches of sources and keep only the finite
entries. VTK's `vtkDijkstraGraphGeodesicPath` answers one source–target pair
at a time and is not suited to this.

## 6. Open questions

1. **Curvature.** Blurring a curved layer is not blurring a planar one: to
   first order the blurred edge of a curved surface shifts by a term in
   `κ σ²` (mean curvature `κ`). The planar model ignores this, and so does the
   1D model. Is it worth adding to `I_model` (κ from `φ`), and at what
   curvatures does it matter at clinical resolution?
2. **Thin plates and close surfaces.** A voxel may be closest to one surface
   but within the blur of another (the two cortices of a thin plate, or a
   neighbouring bone). The 1D model has the same problem along the profile.
   The band should not reach past the medial axis, so the inside depth
   should be capped at the local half-thickness of the bone (from `φ`).
3. **The basis for the parameter fields.** Section 4 settles on blending
   parameters at `c(x)` with surface-distance weights. Still open: the
   kernel (truncated Gaussian or Wendland), its width relative to the
   centre spacing, and whether coarse centres (the control points) can stand
   in for most of the explicit regularisation.
4. **ρc.** The same question as in 1D: fixed, estimated, or a field. With a
   global PSF fitted from all voxels, a global `ρc` may be better
   determined than from profiles.
5. **Topology.** The level set allows topology changes, and the region force
   could split the notch region or the condyles in pathological cases. It
   may need a topology-preserving level set.

## 7. Plan

1. **Prototype in Python** on the notch crop and on a phantom: evaluate
   `I_model` from the current level set (closest node, depth `−φ`) and the
   region force, and check that it points into the notch where the 1D fits
   are untrusted. No solver yet.
2. **Try the 1D lesson first** (section 3): a confident inward move for
   all-soft-tissue profiles in the existing refinement. If it fills the
   notch, the volumetric model is not needed for that particular problem.
3. **Surface distance for the regulariser** (section 5). Truncated Dijkstra
   on the mesh adjacency, as an option beside the k-d tree in the smoothing
   operator, then in `A` and the control points. This stands on its own and
   can be tested with the 1D methods: compare thickness near the notch walls
   and on the `JointPhantom` with Euclidean and surface neighbourhoods.
4. **Rust: a `Method::Volumetric` beside the existing methods.** Band voxels
   get their closest point and depth from the level set, and their
   parameters from the surface basis of section 4, with weights from the
   routine of step 3. The samples are stored ragged: depth, weight, value
   and basis weights per voxel. The global problem reuses the regularised
   solver with the wider rows, and the PSF is a global parameter
   (`σ_xy`, `σ_z`).
5. **Thick profiles, optionally** (section 4, alternatives): kernel-weighted
   local fits as a sampling option for the 1D methods, if the volumetric
   model turns out too costly at HR-pQCT resolution.
6. **Refinement with region forces.** Move the level set with
   `v = −∂E/∂φ` (relaxed), or with the fitted `m − t/2` as now. Compare.
7. **Validation against the 1D methods.** Use the existing phantoms with
   exact truth (`Phantom`, `BonePhantom`, `JointPhantom`), plus a new phantom
   with a deep notch. Measure thickness and surface errors, convergence and
   run time, with the same model settings in both.

## References

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

> R. Kimmel, J. A. Sethian, *Computing geodesic paths on manifolds*,
> Proceedings of the National Academy of Sciences 95(15):8431–8435, 1998.

> K. Crane, C. Weischedel, M. Wardetzky, *Geodesics in heat: a new approach
> to computing distance based on heat flow*, ACM Transactions on Graphics
> 32(5):152, 2013.
