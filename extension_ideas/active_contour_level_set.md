# Refinement as an active contour driven by the profile model

*Idea for future work (not implemented). Recorded 2026-11-27.*

## The idea

Refinement today is a fixed-point iteration: fit the profile model (or the
volumetric model) at every node of a fixed surface, move each node by a
fraction of its fitted edge offset `m − t/2`, remesh, and fit again. It
converges in practice, but nothing guarantees that it descends a single
objective, and its convergence test is a heuristic on the displacements.

The same problem can be posed as a level-set active contour: the periosteal
surface `Γ` is the zero level of `φ`, evolving as

```text
φ_τ + F |∇φ| = 0
```

with a normal speed `F` derived from the cortical bone mapping model, so that
the surface flows down the model's misfit. Two forms:

- **The shape derivative of the volumetric model's cost.** With the band's
  voxels assigned to the surface by closest point, the cost is
  `E(Γ, θ) = ½ Σ_x w(x) (I(x) − f(d_Γ(x); θ))² + penalties`, `d_Γ` the signed
  distance to `Γ` and `θ` the fields (`t`, densities, PSF). For fixed `θ`,
  the speed `F = −δE/δΓ` at a surface point is the derivative of the misfit
  of the voxels it owns with respect to moving it along its normal, which is
  what the Gauss–Newton step in `m` already estimates. Alternating
  (`θ` by the current solver, `Γ` by the flow) is block coordinate descent on
  one energy.
- **A speed field from the 1D fits.** Each profile's fitted `m` is a local
  estimate of where the edge is; `F` interpolated from the trusted nodes'
  `m` (the current refinement's displacement, made a continuous field) gives
  a flow whose fixed points are the current method's.

Regularisation enters as the standard terms: mean-curvature flow
(`F += κ H`, the length or area penalty of Chan & Vese), or the curvature
penalty we already have. A second, coupled level set for the endosteal surface
with a thickness constraint is the coupled-surfaces construction of Zeng et
al. (1999) for the cerebral cortex, here with `t` from the model instead of a
fixed range.

## Why it is interesting

- One energy, so convergence (and the stopping tests) are about that energy
  rather than about displacements.
- Topology changes (a notch opening, a hull bridge breaking) are native to
  level sets.
- The speed is defined everywhere near the surface, not only at nodes, which
  separates the fit's resolution from the surface's.
- Region-based active contours (Chan & Vese 2001) fit a piecewise-constant
  image model; the profile model is a piecewise model blurred by the PSF,
  which makes this a forward-model ("model-based") active contour.

## Open questions

- Time step and stability: the flow moves by sub-voxel amounts per step; an
  implicit or Newton-like step (the current fit's `m` is one) avoids tiny
  explicit steps.
- How the speed is extended off the surface (Adalsteinsson & Sethian 1999)
  and how often `φ` is re-distanced.
- Whether the alternating scheme converges faster than the current
  fit-and-move iteration on real data (TRIKNEE, the clinical knee).
- Cost: the narrow band only needs to cover one step's motion (see the
  level-set work of 2026-11-27 in `docs/notes/hrpqct.md`).

## References

- S. Osher, J. A. Sethian, *Fronts propagating with curvature-dependent speed:
  algorithms based on Hamilton–Jacobi formulations*, Journal of Computational
  Physics 79(1):12–49, 1988.
- M. Kass, A. Witkin, D. Terzopoulos, *Snakes: active contour models*,
  International Journal of Computer Vision 1(4):321–331, 1988.
- V. Caselles, R. Kimmel, G. Sapiro, *Geodesic active contours*, International
  Journal of Computer Vision 22(1):61–79, 1997.
- X. Zeng, L. H. Staib, R. T. Schultz, J. S. Duncan, *Segmentation and
  measurement of the cortex from 3-D MR images using coupled-surfaces
  propagation*, IEEE Transactions on Medical Imaging 18(10):927–937, 1999.
- D. Adalsteinsson, J. A. Sethian, *The fast construction of extension
  velocities in level set methods*, Journal of Computational Physics
  148(1):2–22, 1999.
- T. F. Chan, L. A. Vese, *Active contours without edges*, IEEE Transactions
  on Image Processing 10(2):266–277, 2001.
- G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
  cortical bone thickness measurement from clinical CT data*, Medical Image
  Analysis 14(3):276–290, 2010.
