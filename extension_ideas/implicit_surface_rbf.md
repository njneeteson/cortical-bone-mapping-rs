# Implicit surface from radial basis functions

*Status: idea, not started. Noted 2026-11-26 while making the volumetric
model practical on HR-pQCT (millions of surface nodes at 60.7 µm).*

## The idea

Parametrise the bone surface itself with a small number of degrees of
freedom, instead of carrying one node per boundary voxel:

- Control points every 16–32 voxels along the initial surface.
- Each control point is the centre of a radial basis function. The surface
  is the zero level of an implicit function built from them (for example the
  initial signed distance plus a sum of RBF corrections), so it is defined
  everywhere and stays a closed, non-self-intersecting surface.
- Treece's model of the image around a surface (the blurred three-level
  profile, or the volumetric model's voxel-wise version) is then written in
  terms of the signed distance to that implicit surface. The fit adjusts the
  RBF weights (where the surface is) together with the cortical parameters
  (thickness, densities), rather than fitting edge offsets `m` at fixed nodes
  and moving the surface afterwards.

## Why it might help

- Far fewer unknowns: thousands of RBF weights instead of millions of node
  parameters, and the smoothness comes from the basis rather than from a
  smoothing penalty with hundreds of neighbours.
- Refinement could become a single fit: the surface position is a fitted
  quantity, not a fixed-point iteration of move–refit.
- The level set keeps topology and avoids folds by construction.

## Inspiration from smoothed particle hydrodynamics

The idea is close to how SPH represents continua with sparse particles,
and several of its tools carry over:

- **Kernel interpolation.** SPH carries properties on particles and
  interpolates them with a compact smoothing kernel `W(r, h)`:
  `A(x) = Σ_j V_j A_j W(x − x_j, h)`, with analytic gradients from `∇W`.
  Here the particles would carry the cortical parameters (`m`, `t`, the
  densities) and the surface offset, so every band voxel gets its
  parameters from the kernel-weighted particles around it, not from one
  triangle's three vertices.
- **Free surfaces from a colour function.** SPH finds a fluid's free
  surface, its normal and its curvature from a "colour field"
  `c(x) = Σ_j V_j W(x − x_j, h)` (normal `∇c / |∇c|`, curvature `−∇·n̂`), as
  in surface-tension models (Brackbill et al. 1992; Morris 2000). Particles
  placed inside the bone would define the bone's surface as a level of `c`:
  an implicit surface with normals and curvature built in.
- **Particle shifting.** SPH keeps its particles evenly spread by small
  corrective shifts; here particles would follow the surface as it moves
  (Lagrangian), instead of remeshing a level set every move.
- **Kernel corrections.** Near a boundary SPH renormalises its kernel
  (Shepard and moving-least-squares corrections) so that constants and
  linear fields are reproduced, much as `ω` normalises its Gaussian weights.
  The one-sided neighbourhoods at the image's cut ends need the same.
- **Neighbour search.** Compact kernels plus cell lists keep everything
  `O(N)`.

A close cousin from graphics is point-set surfaces: a surface defined by
moving least squares over a cloud of points with Gaussian weights (Alexa et
al. 2003; Öztireli et al. 2009). It gives a smooth surface from sparse
samples and has well-studied projection operators.

References for this section:

- J. U. Brackbill, D. B. Kothe, C. Zemach, *A continuum method for modeling
  surface tension*, Journal of Computational Physics 100(2):335–354, 1992.
- J. P. Morris, *Simulating surface tension with smoothed particle
  hydrodynamics*, International Journal for Numerical Methods in Fluids
  33(3):333–353, 2000.
- J. J. Monaghan, *Smoothed particle hydrodynamics*, Reports on Progress in
  Physics 68(8):1703–1759, 2005.
- M. Alexa, J. Behr, D. Cohen-Or, S. Fleishman, D. Levin, C. T. Silva,
  *Computing and rendering point set surfaces*, IEEE Transactions on
  Visualization and Computer Graphics 9(1):3–15, 2003.
- A. C. Öztireli, G. Guennebaud, M. Gross, *Feature preserving point set
  surfaces based on non-linear kernel regression*, Computer Graphics Forum
  28(2):493–501, 2009.

## Open questions

- Does the joint fit stay well posed? Surface position, thickness and blur
  trade off; the RBF scale must be chosen so the data can tell them apart.
- How do the RBF scale and spacing interact with thin features (a thin
  cortex on a sharp ridge, the tibial spines, the patellar ridge)?
- The signed distance of an implicit surface changes non-linearly with the
  weights: Gauss–Newton needs its derivative at every band voxel.
- Compactly supported RBFs (Wendland) keep the system sparse; global ones
  (thin-plate) do not.
- Initialisation: fit the RBF weights to the starting segmentation's signed
  distance first.

## Where to develop it

On its own branch, or as a fallback if the node-based volumetric model
cannot be made fast and reliable on HR-pQCT.
