# Design note: a volumetric cortical model

This note sets out the reasoning behind the volumetric model: why a model of
the voxels around the surface should measure the cortex better than one
profile per node, how it is formulated, and which alternatives were
considered. The model runs alongside the 1D model of Treece et al. (2010); it
does not replace it. What the crate computes is specified in
[method.md](method.md#volumetric-model), and how well it works is in
[volumetric_validation.md](volumetric_validation.md). The last section lists
what is implemented and what remains.

## 1. The 1D model and its assumptions

At a node `p` with inward normal `n`, the 1D method models the profile
`y(x) = I(p + x n)` as

```text
f(x) = ρs + (ρc − ρs) Φ((x − m + t/2) / σ) + (ρb − ρc) Φ((x − m − t/2) / σ)
```

with `Φ` the standard normal CDF. This assumes:

1. a planar layer: the cortex is a slab of thickness `t` with its mid-line at
   depth `m`, perpendicular to `n`;
2. blur only along `n`: for a planar slab, a Gaussian PSF with covariance `Σ`
   blurs the layer exactly like a 1D Gaussian of variance `σ² = nᵀ Σ n`, so
   the in-plane part of the PSF does not matter;
3. a known normal, taken from the mesh rather than the image;
4. one line of samples per node: neighbouring voxels off the line are used
   only through the interpolation, and profiles of nearby nodes overlap
   inside the bone;
5. no profile, no information: a profile that never reaches bone (a surface
   bridging a concavity deeper than `inside`) gives an untrusted fit, and the
   node moves only with its neighbours.

Assumptions 1–3 fail at corners and in tight concavities such as the femoral
intercondylar notch, and assumption 5 is why a bridged notch fills only if
its rim zips it shut ([refinement_experiments.md](refinement_experiments.md)).
More generally, a single line through a porous, trabecularised endosteum is
ambiguous, and all the information off the line is wasted. The question is
whether a model of the voxels does better, judged by more accurate or more
precise measurements of thickness and surface position.

## 2. The same model evaluated at every voxel

The volumetric model changes only how the parameters of each node are fitted
from the image. As for the 1D methods, the surface is fixed during a fit, and
the fit answers, for each node, how far along the normal the middle of the
cortex lies (`m`) and how thick it is (`t`); refinement then moves the
surface by the fitted `m − t/2` and fits again.

Let `φ` be the signed distance to the fixed surface (negative inside). For a
voxel at `x` near the surface let `d(x) = −φ(x)` be its depth, `c(x)` its
closest point on the surface with inward normal `n(x) = −∇φ(x)` there, and
`θ(c) = (m, t, ρs, ρb)` the model fields at that point. Under assumptions 1
and 2, locally around `c(x)`, the blurred density at `x` is the 1D model at
the voxel's depth:

```text
I_model(x) = f( d(x) ; m(c), t(c), σ_n(x), ρs(c), ρb(c) ),     σ_n(x)² = n(x)ᵀ Σ n(x)
```

So the planar 3D model is the 1D model with every voxel of a band around the
surface as a sample, placed by its signed distance and owning the parameters
of its closest surface point. The cost is

```text
E = Σ_{x ∈ band} w(d(x)) (I(x) − I_model(x))²  +  regularisation of m, t, ρs, ρb over the surface
```

over the band `−outside ≤ d(x) ≤ inside`. Because the surface is fixed during
a fit, the band and every depth are fixed, and the cost is smooth in the
parameters. The weight `w(d)` tapers towards the ends of the band, so that
voxels entering or leaving it at a move change the cost little; it depends on
the depth below the fixed surface, never on the fitted edge, which would let
the fit lower the cost by moving the cortex away from voxels that disagree
with it. Voxels inside another bone get zero weight, as profile samples do.

Compared with profiles, each voxel is used once, at its true position, with
no interpolation, and the closest-point assignment partitions the band by
the closest triangle. The normal of a sample is the direction to its closest
point, which varies across a cell, so on either side of a corner the voxels
see different normals, which a single profile cannot. And the PSF becomes a
global property of the scan rather than a `σ` fitted per node.

### The point-spread function

The PSF is modelled on how a CT scanner blurs: a Gaussian whose covariance is
fixed in the scanner's frame,

```text
Σ = R diag(σ_xy², σ_xy², σ_z²) Rᵀ
```

with `σ_xy` the in-plane blur (focal spot, detector, reconstruction kernel),
`σ_z` the blur along the table (the slice sensitivity profile), and `R` the
known rotation from the scanner's axes to the image's (the identity for an
axial acquisition). For a voxel whose normal is `n` in the scanner frame,

```text
σ_n² = σ_xy² (1 − n_z²) + σ_z² n_z²
```

so two global parameters replace the per-node `σ`. This is the blur model of
Treece & Gee (2015, appendix A), `σ = √((σ_i cos α)² + (σ_o sin α)²)` for a
line at angle `α` to the slices (`n_z = sin α`). They estimate `σ_i` and
`σ_o` for each scan by regressing the blurs of per-location 1D fits on `α`;
here the two are parameters of the model, fitted with everything else.

Both are determined only if the surface has normals in both directions,
which holds for a whole bone but not for a short shaft, whose normals are all
nearly in-plane; there `σ_z` must be fixed or tied to `σ_xy`. A voxel's value
is the image averaged over its footprint, which adds `h_i²/12` per axis to
`Σ`, so the fitted values are the effective blur of the image, not the
scanner's PSF. Real reconstruction kernels are not Gaussian (sharp kernels
overshoot at edges), and the blur varies across the field of view; both are
open (section 7).

### Cost

On a clinical CT of the femur, a 4 + 8 mm band over about 30 000 mm² of
surface at 0.26 mm³ per voxel is about 1.4 million voxels, similar to the
2.4 million samples of the profiles. At HR-pQCT resolution a 2 + 4 mm band
over 5000 mm² is about 140 million voxels: many rows, but not many unknowns.
The unknowns are still the node parameters, and each voxel touches only the
three vertices of its triangle, so the Gauss–Newton matrix has the sparsity
of the mesh however many voxels there are. The voxels enter only when the
blocks and the gradient are formed, once per iteration, in a pass linear in
the voxel count; PCG works on the blocks. The steps to watch are the
closest-point query for every band voxel and the memory per voxel.

## 3. Bone beyond the band: region forces

With the surface fixed, changing a node's periosteal offset `m − t/2` by `v`
moves the modelled periosteal edge by `v`, so the derivative of `E` with
respect to the offset field is the shape derivative of the data term. With
the unblurred model written as indicators of the modelled periosteal region
`P` and endosteal region `B`,

```text
D(x) = ρs + (ρc − ρs) χ_P(x) + (ρb − ρc) χ_B(x),      I_model = G_Σ * D
```

the first variation for a normal displacement `v` of the periosteal edge is

```text
δE = −2 ∫_S v(s) (ρc − ρs) (G_Σ * r)(s) ds,      r = w (I − I_model)
```

the residual image blurred by the PSF and read on the edge, times the height
of the periosteal step: the Chan–Vese (2001) region force with a blurred step
model. At the start of a fit on a surface that bridges a gap, the model says
cortex just inside the edge where the image says soft tissue, so `r < 0` and
the force pushes the offset inward without any voxel reaching the bone
across the gap. Run to convergence, though, the cortex is pushed past the
inner end of the band, the edge lands at the inner bound, and the fit is
called untrusted. The volumetric model does not change this: over a bridge
every voxel in the band is soft tissue. It helps at the rim of a notch, where
the walls are in the band.

The region force shows that this outcome is not uninformative: it says that
the bone is further in than the band reaches. Treating "edge at the inner
bound with a soft-tissue band" as a confident inward move of the maximum step
would give refinement the region force without changing the outer loop, for
profiles and voxels alike.

## 4. Parameter fields on the surface

Giving each voxel the parameters of its closest node would partition the band
into cells, with `I_model` jumping across cell boundaries wherever
neighbouring nodes differ. Instead the geometry comes from the fixed surface
and only the parameters are smoothed, with basis functions on the surface.
The closest point `c(x)` lies in a triangle, and the fields there are
interpolated linearly from its three vertices with barycentric weights
`β_v(x)`: hat functions centred on the nodes. They can be composed with a
coarse basis centred on control points `q` (the multigrid interpolation
matrix `A`, built with surface distance):

```text
θ(x) = Σ_{v ∈ tri(x)} β_v(x) θ_v,      θ_v = Σ_q A_vq θ_q
```

With `A = I` the fields are one value per node, as in the regularised method.
Depth and normal still come from the surface, which represents corners and
concavities exactly, so the basis never decides the geometry. The hat
functions touch only one triangle, and `A` built with surface distance does
not couple the two walls of a notch or the two cortices of a plate. A voxel's
Gauss–Newton contribution `(β βᵀ) ⊗ (∇f ∇fᵀ)` couples the three vertices of
its triangle, so the data term has a block per vertex and one per mesh edge,
and the block-Jacobi preconditioner uses the vertex blocks, as for the 1D
methods.

The bounds carry over. The global solver is a projected Levenberg–Marquardt
with an active set, so the other parameters are always fitted given the
bounded ones. Because the barycentric weights and the normalised Gaussian
weights of `A` are non-negative and sum to one, `θ(x)` at every voxel is a
convex combination of fitted values, and bounds on those values bound the
parameters at every voxel; a basis with negative weights, such as
interpolating radial basis functions, would lose this.

### Alternatives considered

Blending predictions instead of parameters, by giving each node its own
planar step `f_i(x) = f(n_iᵀ(x − p_i); θ_i)` and blending
`I_model = Σ_i w_i f_i` (a partition of unity in the spirit of Ohtake et al.
2003), needs no level set and lets the normals be fitted, but it rounds
corners: just past a convex corner, a voxel lies below the extension of one
face's plane, so that face's step says bone where the other says soft tissue.
Narrowing the kernel to fix this brings back the closest-point cells.

Kernel-weighted local fits ("thick profiles"), in which each node fits its
own planar model to the voxels around it with an anisotropic kernel (width
`h` along the surface, the usual band along the normal), are not a volumetric
model, since voxels are shared between nodes, but they would be a cheap
upgrade of the 1D sampling: real voxels, noise averaged over a disc, and room
to fit the normal, at the price of planar bias over `h`.

Basis functions for the surface itself, an implicit surface
`φ(x) = Σ_i λ_i ψ(|x − p_i|) + polynomial` (Carr et al. 2001; Aghasi et al.
2011), would turn the region force into a gradient with respect to `λ_i`, but
they fit the surface and the parameters together and merge the inner and
outer loops, which this design keeps apart.

## 5. Distance along the surface

The regulariser `ω` originally weighted the `k` nearest neighbours of a node
in Euclidean distance. What is wanted is distance along the surface: two
points on opposite walls of the notch, or on the two cortices of a thin
plate, are close in space but far apart on the surface, and their thicknesses
should not be smoothed together. Two bones in one mesh are then at infinite
distance and never couple.

The mesh stores its one-ring adjacency, so from each node a Dijkstra search
over the edge graph with Euclidean edge lengths can stop when its frontier
passes a radius `R`. Each search visits only the nodes inside `R`, so
building the whole operator costs about `O(N k log k)`, in parallel over
nodes, and produces the same `(node, distance)` rows the sparse operators are
built from; only the metric changes. A radius, rather than "`k` nearest" or
"`k` steps", keeps the operator's meaning independent of the mesh density.

A path along edges zigzags and overestimates the geodesic. On a regular mesh
with `2k` edge directions the worst case, midway between two directions, is
`1/cos(π/2k)`: 8% for 8 directions, 15% for 6 (a square grid split by one
diagonal, as marching cubes produces locally) and 41% for 4; averaged over
directions it is a few percent. For a Gaussian weight that is a narrower,
slightly anisotropic kernel. Local fast marching on the triangles (Kimmel &
Sethian 1998) would be exact to first order if it mattered; the heat method
(Crane et al. 2013) solves from one source over the whole mesh, which suits
this badly. Surface distance does not help over a bridged notch, since the
two walls are then close along the surface through the bridge; it keeps them
apart only once the surface follows the bone. It is needed between nodes and
between nodes and control points, never per voxel, so one routine serves the
regulariser, the multigrid interpolation and the volumetric model.

## 6. Reliability of the fitted values

The 1D method calls a fit untrusted from its outcome, such as an edge at a
bound. The volumetric fit is one global problem, so reliability is a property
of each node's parameters within it: how much the data determined them, and
how well the model fits there. Six measures were considered.

1. A local standard error from the data-only vertex blocks `B_v`, which are
   formed anyway: `s √[B_v⁻¹]_tt`, with `s` the noise level. Inverting the
   block accounts for the trade-off of `t` with the node's other parameters.
   It ignores the neighbours sharing voxels and the regulariser, so it is the
   error of a node fitted from its own voxels alone, the honest answer to
   "what did the data say here". CT noise is correlated over a few voxels by
   the reconstruction kernel, so the voxels are not independent and the
   correlation must enter the estimate.
2. The data share: for each fitted value, the fraction of its curvature that
   comes from the data, `B_tt / (B_tt + P_tt)` with `P` the penalties'
   diagonal. Near one the data decided the value, near zero the regulariser
   filled it in. It is cheap but depends on the penalty weights and says
   nothing about whether the model is right.
3. The full posterior variance, the diagonal of `(JᵀJ + R)⁻¹`, estimated
   stochastically with tens of probe vectors (Bekas et al. 2007): the right
   number to map, but tens of solves; for validation.
4. The local misfit, the RMS residual over a node's voxels relative to the
   noise. Measures 1–3 assume the model holds; this finds where it does not
   (porosity, a thin plate within the blur of its other cortex, a surface
   bridging a gap, metal).
5. Bound activity, as for the 1D methods.
6. Resampling (a wild bootstrap, or repeated noise realisations of a
   phantom): complete but many fits; for validation.

The crate reports 1, 2 and 4 for every node. For 1 it sums the noise
correlation over the pairs of the node's voxels rather than applying a block
factor, because a node's voxels form a column about one voxel across, for
which the block factor overstates the variance many times
([method.md](method.md#volumetric-model)). Measure 6 is what the phantom
validation does with five noise realisations.

## 7. Open questions

1. Curvature of the layer: blurring a curved layer shifts its blurred edge by
   a term in `κ σ²` (mean curvature `κ`). The planar model ignores this, and
   so does the 1D model. At what curvatures does it matter at clinical
   resolution?
2. Thin plates: a voxel may be closest to one cortex of a thin plate but
   within the blur of the other. The band should not reach past the medial
   axis, so the inside depth could be capped at the local half-thickness of
   the bone.
3. The coarse basis `A`: the width of its kernel relative to the
   control-point spacing, and whether coarse centres can stand in for much of
   the explicit regularisation.
4. `ρc`: fixed, estimated or a field. The validation shows that the estimate
   fails when no cortex in view is much thicker than the blur.
5. The shape of the PSF: is a slowly varying `σ_xy` field, or a kernel with a
   negative lobe, worth the extra parameters?
6. Refinement with the volumetric model plateaus at about a tenth of a voxel
   on phantoms and does not finish on real data. A likely cause is that the
   fit is not exactly shift-equivariant: it regroups the voxels by their
   closest face whenever the surface moves.
7. Topology: the level set allows topology changes, which could split the
   notch region or the condyles in pathological cases.

## 8. Status

Implemented: the volumetric model with barycentric fields (`A = I`), the
global anisotropic PSF, the curvature penalty on the edge positions and
smoothing of the density fields (`method="volumetric"`); surface-distance
neighbourhoods for every method (`neighbour_distance`); the reliability
measures 1, 2 and 4; refinement with the volumetric inner fit; site phantoms
of the distal radius and tibia (HR-pQCT) and the proximal femur and knee
(clinical CT); and validation on those phantoms and on the clinical knee
([volumetric_validation.md](volumetric_validation.md)).

Not implemented: the inner-bound rule and the region force (section 3), the
coarse basis `A` for the volumetric model (section 4), thick profiles, and a
repeat-scan precision study, which would compare the per-node precision of
thickness (RMS SD and RMS CV%, Glüer et al. 1995) of every method after
mapping repeat scans to a common surface.

## References

* G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
  thickness and cortical bone density using clinical CT*, Medical Image
  Analysis 20(1):249–264, 2015.
* G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
  cortical bone thickness measurement from clinical CT data*, Medical Image
  Analysis 14(3):276–290, 2010.
* T. F. Chan, L. A. Vese, *Active contours without edges*, IEEE Transactions
  on Image Processing 10(2):266–277, 2001.
* Y. Ohtake, A. Belyaev, M. Alexa, G. Turk, H.-P. Seidel, *Multi-level
  partition of unity implicits*, ACM Transactions on Graphics
  22(3):463–470, 2003.
* J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright,
  B. C. McCallum, T. R. Evans, *Reconstruction and representation of 3D
  objects with radial basis functions*, Proceedings of SIGGRAPH 2001, 67–76.
* A. Aghasi, M. Kilmer, E. L. Miller, *Parametric level set methods for
  inverse problems*, SIAM Journal on Imaging Sciences 4(2):618–650, 2011.
* C. Bekas, E. Kokiopoulou, Y. Saad, *An estimator for the diagonal of a
  matrix*, Applied Numerical Mathematics 57(11–12):1214–1229, 2007.
* C.-C. Glüer, G. Blake, Y. Lu, B. A. Blunt, M. Jergas, H. K. Genant,
  *Accurate assessment of precision errors: how to measure the
  reproducibility of bone densitometry techniques*, Osteoporosis
  International 5(4):262–270, 1995.
* R. Kimmel, J. A. Sethian, *Computing geodesic paths on manifolds*,
  Proceedings of the National Academy of Sciences 95(15):8431–8435, 1998.
* K. Crane, C. Weischedel, M. Wardetzky, *Geodesics in heat: a new approach
  to computing distance based on heat flow*, ACM Transactions on Graphics
  32(5):152, 2013.
