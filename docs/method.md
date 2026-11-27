# Method

What the package computes. The notation follows our 2025 paper (Neeteson &
Boyd, *Comput. Biol. Med.*).

## Pipeline

| step | what happens |
|---|---|
| 1. surface | the mask's outer surface by surface nets, smoothed by Taubin λ\|μ smoothing |
| 2. nodes | every surface vertex, except those on caps where the image cuts the bone (and outside a `sub_mask`) |
| 3. profiles | the image sampled by trilinear interpolation along each node's inward normal, from `-outside` to `+inside` |
| 4. initial fit | one model fitted to all profiles at once |
| 5. fit | the chosen method (below) |
| 6. refinement (optional) | move the surface to the fitted periosteum and repeat from step 2 |

## The profile model

$$
\hat f(x) = \rho_s + \frac{\rho_c - \rho_s}{2}\left[1 + \operatorname{erf}\frac{x - m + t/2}{\sigma\sqrt2}\right]
          + \frac{\rho_b - \rho_c}{2}\left[1 + \operatorname{erf}\frac{x - m - t/2}{\sigma\sqrt2}\right]
$$

| symbol | meaning |
|---|---|
| $x$ | distance along the inward normal from the node |
| $m$ | position of the cortical mid-line |
| $t$ | cortical thickness |
| $\sigma$ | blur |
| $\rho_s$, $\rho_b$ | soft-tissue and trabecular densities |
| $\rho_c$ | cortical density, never fitted |

- Periosteal edge at $m - t/2$, endosteal edge at $m + t/2$.
- Residuals are weighted $\gamma_j = \gamma_0 + (1-\gamma_0)\,|x_j|/\max|x|$
  ($\gamma_0 = 10$), so edges near the surface are preferred over trabeculae
  (Treece et al. 2010).

## Methods

| method | unknowns | coupling between nodes | source |
|---|---|---|---|
| `local` | $m, t, \sigma, \rho_s, \rho_b$ per node | none | Treece et al. 2010 |
| `multigrid` | $m, t, \sigma$ on control points, coarse to fine; $\rho_s, \rho_b$ global | interpolation from nearby control points | Neeteson & Boyd 2025 |
| `regularized` | $m, t, \sigma$ per node; $\rho_s, \rho_b$ global or per node | smoothing penalty $\lVert\omega v\rVert^2$ | Neeteson & Boyd 2025 |
| `volumetric` | $m, t, \rho_s, \rho_b$ per node; one PSF for the scan | smoothing + curvature penalty | extension |

The global methods minimise

$$
L = \frac{1}{2NM}\sum_{ij}\gamma_j(\hat f_{ij} - f_{ij})^2
  + \frac{\lambda(\rho_c - \rho_{s0})^2}{2N}\left[\frac{\lVert\omega m\rVert^2}{t_0^2} + \frac{\lVert\omega t\rVert^2}{t_0^2}
  + \frac{\lVert\omega\sigma\rVert^2}{\sigma_0^2}\right]
$$

- $\omega x_i = x_i - \sum_k g(d_{ik})x_k / \sum_k g(d_{ik})$ over the `reg_neighbours`
  nearest nodes, with a Gaussian $g$ of standard deviation `reg_sigma`.
- $t_0$, $\sigma_0$, $\rho_{s0}$ come from the initial fit.
- The penalty is weighted by the squared contrast of the cortex against soft
  tissue. Our 2025 paper weights it by $\rho_c^2$, which depends on where the
  image's units put their zero; in calibrated mg HA/cm³ or in HU, where soft
  tissue is near 0, the two nearly agree.
- Solved by bounded Gauss–Newton / Levenberg–Marquardt with matrix-free
  preconditioned conjugate gradients; the local fit by bounded
  Levenberg–Marquardt with analytic derivatives.
- Stopping tests are dimensionless and per variable: the largest cosine
  between a variable's Jacobian column and its own residuals (`gtol`), and
  the largest change a step makes in any node's predictions, in units of the
  noise (`xtol`). They do not depend on the number of nodes, the length unit,
  or the units of the image values and their zero.
- Results are bitwise identical across runs and thread counts.

## Volumetric model

Instead of one profile per node, every voxel of a band around the surface
is fitted. Each voxel takes the parameters of its closest surface point
(interpolated over the triangle), and its model value is the profile model
at its depth $d$ with one PSF for the scan:

$$
I(x) = \hat f(d;\, m, t, \sigma_n, \rho_s, \rho_b),\qquad
\sigma_n^2 = \sigma_{xy}^2(1 - a^2) + \sigma_z^2 a^2
$$

- $a$ is the normal's component along the scanner axis (Treece & Gee 2015).
- $\sigma_{xy}$, $\sigma_z$ are fitted with the model, fixed, or tied
  (`vol_sigma_z="tied"`, for a straight shaft where $\sigma_z$ is not
  measurable).
- $m$, $t$ and the densities are smoothed with the same operator $\omega$
  (`vol_edge_lambda`, `vol_lambda`), and the fitted surfaces carry the
  `"change"` curvature penalty (`vol_curvature_weight`).
- `result.volumetric` reports the PSF, the image noise and, per node, the
  standard error of $t$ from its own voxels.

## Curvature penalty

An optional penalty on the curvature of the fitted surfaces
$S = V + (m \mp t/2)\,n$:

| `curvature=` | penalises | free |
|---|---|---|
| `"change"` | curvature added by the fit relative to the input surface | uniform offsets |
| `"absolute"` | curvature above `curvature_kappa` (default 5 mm⁻¹, Neeteson et al. 2023) | curvature below the threshold |
| `"variation"` | $\tfrac12\int\lvert\nabla(\kappa_1+\kappa_2)\rvert^2$, how fast curvature changes | smoothly curved bone |

Curvature is measured with a Gaussian point-cloud Laplacian at four times
the node spacing (`curvature_scale_spacings`).

## Cortical density

$\rho_c$ trades off against $t$ (a thinner, denser cortex blurs into nearly
the same profile), so it is fixed, never fitted:

| `rho_c=` | value | note |
|---|---|---|
| a number | as given | right for a calibrated scanner with known tissue density |
| `"estimate"` | one value: the median plateau of nodes whose cortex is wide enough (> 4σ) to show it | the apparent density, including sub-resolution porosity |
| `"profile_max"` | each profile's maximum | as Bonelab; overestimates thin cortices |

Bones of one scan share the estimate (`shared_scanner`).

## Segmentation refinement

$$
v_i \leftarrow v_i + \alpha\,(m_i - t_i/2)\,n_i,\qquad \alpha = 0.5
$$

- Only trusted fits move a node: edge inside the sampled range, parameters
  off their bounds, a cortex brighter than its surroundings and a residual at
  most three times the median. Other nodes follow their trusted neighbours
  (harmonic interpolation).
- The surface is a level set on the image grid, remeshed every move, so it
  cannot fold.
- Regions that have stopped moving are frozen; the scanner parameters
  ($\rho_c$, PSF) are re-estimated between passes.
- Converged: a full fit moves the surface less than a tenth of a voxel RMS,
  and less than twice that in every region.

## Starting segmentations

| function | for | how |
|---|---|---|
| `hull_segmentation` | one bone (HR-pQCT radius, tibia) | threshold, median, largest component, convex hull per slice |
| `bone_hulls(closing_radius=4)` | several bones (clinical CT) | per bone: closing with a 4 mm ball, holes filled; keeps wide concavities such as the femoral notch |
| `buie_segmentation` | comparison | dual-threshold method of Buie et al. (2007) |

## Differences from Bonelab

| | Bonelab (Python) | this package |
|---|---|---|
| optimiser | SciPy L-BFGS-B (global), TRF with numerical Jacobian (local) | Gauss–Newton/LM + PCG (global), bounded LM with analytic Jacobian (local) |
| regularised loss | penalty scaled by $\rho_c^2/t_0$ in the loss but $\rho_c^2/t_0^2$ in the gradient | $(\rho_c - \rho_{s0})^2/t_0^2$ in both |
| stopping tests | SciPy's `ftol`, `gtol`, relative to the whole problem | per variable, in units of the noise |
| local residual weights | $\gamma^2$ | $\gamma$, as the global methods |
| $\omega$ neighbours | $k$ including the node | $k$ excluding it, with a distance cut-off |
| surface | PyVista contour + windowed-sinc smoothing | surface nets + Taubin λ\|μ |

## References

- G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
  cortical bone thickness measurement from clinical CT data*, Med. Image
  Anal. 14(3):276–290, 2010.
- G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
  thickness and cortical bone density using clinical CT*, Med. Image Anal.
  20(1):249–264, 2015.
- N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
  optimization*, Comput. Biol. Med., 2025,
  [doi:10.1016/j.compbiomed.2025.110636](https://doi.org/10.1016/j.compbiomed.2025.110636).
- N. J. Neeteson et al., *Automatic segmentation of trabecular and cortical
  compartments in HR-pQCT images using an embedding-predicting U-Net and
  morphological post-processing*, Sci. Rep. 13:252, 2023.
- H. R. Buie et al., *Automatic segmentation of cortical and trabecular
  compartments based on a dual threshold technique for in vivo micro-CT bone
  analysis*, Bone 41:505–515, 2007.
