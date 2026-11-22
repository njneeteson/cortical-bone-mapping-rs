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
voxels, far too many. Subsample the band there, or keep profiles. Either
way, the volumetric model fits large, coarse images best, like the density
fields.

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

## 4. Open questions

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
3. **Interpolated or piecewise fields.** Closest-point ownership makes
   `I_model` jump across cell boundaries when neighbouring nodes differ.
   Interpolating the fields at `c(x)` from the few nearest nodes makes the
   model smooth, at the cost of a wider Jacobian stencil.
4. **ρc.** The same question as in 1D: fixed, estimated, or a field. With a
   global PSF fitted from all voxels, a global `ρc` may be better
   determined than from profiles.
5. **Topology.** The level set allows topology changes, and the region force
   could split the notch region or the condyles in pathological cases. It
   may need a topology-preserving level set.

## 5. Plan

1. **Prototype in Python** on the notch crop and on a phantom: evaluate
   `I_model` from the current level set (closest node, depth `−φ`) and the
   region force, and check that it points into the notch where the 1D fits
   are untrusted. No solver yet.
2. **Try the 1D lesson first** (section 3): a confident inward move for
   all-soft-tissue profiles in the existing refinement. If it fills the
   notch, the volumetric model is not needed for that particular problem.
3. **Rust: a `Method::Volumetric` beside the existing methods.** Band voxels
   are assigned to nodes (closest point, via the level set), and the
   samples are stored ragged: depth, weight and value per voxel. The
   global problem reuses the regularised solver with ragged rows, and the
   PSF is a global parameter (`σ_xy`, `σ_z`).
4. **Refinement with region forces.** Move the level set with
   `v = −∂E/∂φ` (relaxed), or with the fitted `m − t/2` as now. Compare.
5. **Validation against the 1D methods.** Use the existing phantoms with
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
