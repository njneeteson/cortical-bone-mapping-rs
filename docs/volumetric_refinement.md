# Making volumetric refinement converge

Refinement moves the surface to the edges that each fit locates, so it is
what lets cortical bone mapping start from a poor segmentation. With the
[volumetric model](method.md#volumetric-model) as the inner fit it does not
yet work. On the clinical knee CT it never converged: after 150 moves, 9.2
hours against 138 s for the regularised fit, the surfaces still moved by
about 0.005 mm per move
([clinical_ct.md](clinical_ct.md#results)). On the femoral notch it stalled
after 44 moves with 2.6 cm³ of bone outside the surface
([refinement_experiments.md](refinement_experiments.md)). On the phantoms the
regularised refinement had also become worse at bridging cortical gaps and at
staying off a neighbouring bone than it was before the knee work.

This page follows the work in phases, each committed separately, and records
what each phase changed, how it was tested and what it did. Phantoms only
check that nothing breaks; whether a change helps is judged on the clinical
knee CT, and the choices will be revisited on real HR-pQCT data.

## Causes

* **The curvature penalty ratchets.** The default `"change"` penalty resists
  curvature *added* to the surface being fitted. During refinement that
  surface is the previous fit's, so curvature added in one fit is free in the
  next and a dent can deepen move after move. The old loop stopped early on a
  global RMS; the regional convergence test added for the knee keeps it
  iterating while a region moves, which is where the bridging got worse.
* **The PSF is re-fitted on every full fit**, so the edges the surface moves
  toward change with it.
* **Parameters taken from HR-pQCT.** The curvature is measured at 0.25 mm,
  below the 0.65 mm node spacing of clinical CT, and the density fields are
  smoothed over 500 neighbours. Both make the solves stiff: the full fits on
  the knee used up their conjugate-gradient budget.
* **Voxels go to the nearest surface point.** In a concavity, voxels beside
  one wall can be assigned to the other.
* **Untrusted regions lagged.** Nodes whose fits cannot be trusted move with
  their trusted neighbours by harmonic interpolation, done with 50 Jacobi
  sweeps; that reaches about 7 rings, about 4.5 mm on the knee, against a
  notch about 20 mm wide.
* **The trust tests were tuned for profiles**, not for the volumetric fit.

## Phases

| phase | change | status |
|---|---|---|
| 1 | absolute and curvature-variation penalties | done; none beat `"change"`, which stays the default |
| 2 | untrusted regions moved by an exact harmonic fill | done; the volumetric knee still does not converge |
| 3 | PSF and `ρc` in an outer loop, held during each refinement | planned |
| 4 | parameters scaled to the scanner and the node spacing | planned |
| 5 | voxels assigned along the normals; adaptive band | planned |
| 6 | weights and thresholds chosen on the knee | planned |

## 1. Curvature penalties

### Methods

Two penalties that depend only on the fitted surface, not on the surface the
fit starts from, were added beside `"change"`
([curvature.rs](../crates/cortical-bone-mapping/src/operators/curvature.rs)):

* **absolute** (`curvature="absolute"`): `w · max(0, K − κ)²`, the curvature
  `K` of the fitted surface above a threshold `κ`. `K` is `|κ₁ + κ₂|`
  (`curvature_measure="mean"`) or `√(κ₁² + κ₂²)` (`"total"`, from a local
  quadric fit, [hessian.rs](../crates/cortical-bone-mapping/src/operators/hessian.rs)),
  which unlike the mean curvature is not zero at a saddle. Both equal the
  contour curvature `1/R` on a shaft, so `κ` means what it meant in our
  segmentation network (Neeteson et al. 2023), which penalised contour
  curvature above 5 mm⁻¹, set from real radius and tibia segmentations. The
  squared excess has a continuous slope at `κ`, which a Gauss–Newton solver
  needs; the network's `ReLU((κ/κ_thresh)² − 1)` does not.
* **variation** (`curvature="variation"`): `w · ½ ∫ |∇(κ₁ + κ₂)|²`, how fast
  the curvature changes along the surface
  ([variation.rs](../crates/cortical-bone-mapping/src/operators/variation.rs);
  after the minimum-variation surfaces of Moreton & Séquin 1992). Spheres,
  cylinders, their parallel surfaces and slowly varying curvature cost
  nothing; bumps and dents do. It is exactly quadratic in `(m, t)`.

The endosteal surface can be weighted separately
(`curvature_endosteal_weight`), and `cbm.mesh_curvature` returns the
curvature the penalties measure at every vertex of a mesh.

Tests: the curvature of a sphere and of a saddle against their exact values;
the variation penalty is zero on a sphere and its parallel surfaces and not
on a bump; the gradients of every mode against finite differences; each mode
fits a noisy cylinder within 0.02 mm.

### Results

**Healthy shaft phantom** (8.5 mm radius, 40 slices, 60.7 µm voxels,
regularised refinement from the hull, curvature at 0.25 mm):

| penalty | moves | Dice | periosteal distance (mm) | thickness RMSE (mm) |
|---|---:|---:|---:|---:|
| change, weight 1 | 10 | 0.978 | 0.009 | 0.040 |
| absolute, κ = 0, weight 0.1 → 3 | 8–15 | 0.967 → 0.868 | 0.021 → 0.106 | 0.041 → 0.052 |
| absolute, κ = 0.3, weight 0.1 → 3 | 8–11 | 0.971 → 0.943 | 0.016 → 0.042 | 0.046 |
| absolute, κ = 1, any weight | 7–13 | 0.971 | 0.013 | 0.052 |
| absolute, κ = 5, any weight | 8–12 | 0.966 | 0.014 | 0.058 |
| variation, weight 0.1 → 1 | 9–21 | 0.958 → 0.885 | 0.030 → 0.085 | 0.054 → 0.114 |

A smaller shaft (5 mm radius, 24 slices;
[curvature_sweep.py](scripts/curvature_sweep.py)) checked the penalties at
three scales of the curvature measurement, for both inner fits (thickness RMSE in
mm; the first row of each method is `"change"`, weight 1):

| inner fit, penalty | 0.25 mm | 0.5 mm | 1 mm |
|---|---:|---:|---:|
| regularised, change | 0.054 | 0.053 | 0.053 |
| regularised, absolute κ = 0.3, weight 0.1 / 1 | 0.059 / 0.056 | 0.059 / 0.056 | 0.061 / 0.058 |
| regularised, absolute κ = 1, weight 0.1 / 1 | 0.063 / 0.063 | 0.064 / 0.065 | 0.066 / 0.065 |
| regularised, variation, weight 0.1 / 1 | 0.094 / 0.086 | 0.061 / 0.060 | 0.057 / 0.063 |
| volumetric, change | 0.033 | 0.036 | |
| volumetric, absolute κ = 0.3, weight 0.1 / 1 | 0.046 / 0.223 | 0.140 / 0.188 | |
| volumetric, absolute κ = 1, weight 0.1 / 1 | 0.068 / 0.059 | 0.121 / 0.092 | |
| volumetric, variation, weight 0.1 / 1 | 0.058 / 0.174 | | |

`"change"` converged in 7–8 moves everywhere; most volumetric runs with the
other penalties did not converge in 15.

`"change"` was best. With `κ = 0` the penalty flattens the shaft itself;
with `κ ≥ 1` it almost never acts, and the noise stays. The reason is the
starting surface: smoothed from a voxel mask, its curvature at 0.25 mm has a
long tail from the remaining staircase (on a 3.5 mm cylinder with 0.1 mm
voxels, `|κ₁ + κ₂|` has a 90th percentile of 2.7 mm⁻¹ at 0.25 mm, 1.7 at
0.5 mm, 1.0 at 1 mm and 0.8 at 2 mm). `"change"` ignores the starting
surface's own curvature; the absolute and variation penalties have to undo it
through the fit. At 0.25 mm the comparison is therefore mostly about the
mesh, which makes the scale of the curvature measurement (phase 4) part of
the choice of penalty. With `κ = 5 mm⁻¹` the volumetric fit is effectively
unregularised (thickness RMSE 0.108 mm on a noisy cylinder, against 0.028 mm
for the local fit), so `"change"` remains the default for both methods.

**Femoral notch of the knee CT** (regularised refinement from the 4 mm
closing, curvature at 1 mm, the nodes being about 0.65 mm apart;
[notch_experiment.py](scripts/notch_experiment.py), variants
`closing_freeze_*`):

| penalty | moves | time | notch left (cm³) | bone outside (cm³) |
|---|---:|---:|---:|---:|
| none | 4 | 81 s | 0.00 | 1.12 |
| change, weight 1 | 4 | 101 s | 0.00 | 1.12 |
| absolute, κ = 0.3, weight 0.3 | 14 | 142 s | 0.00 | 2.37 |
| absolute, κ = 1, weight 0.3 | 4 | 87 s | 0.00 | 1.12 |
| variation, weight 0.1 | 5 | 103 s | 0.00 | 1.23 |
| variation, weight 1 | 18 | 364 s | 0.00 | 3.65 |

"Bone outside" is thresholded bone more than 2 mm outside the refined
surface. The regularised refinement already empties the notch without a
penalty. `"change"` at weight 1 and the absolute penalty above 1 mm⁻¹ change
nothing measurable; the absolute penalty above 0.3 mm⁻¹ and the variation
penalty at weight 1 do act, and they hold the surface off curved bone
(twice to three times the bone outside) and take four times as many moves.
On this knee, with the regularised fit, no penalty improved on none.

## 2. Untrusted regions

### Methods

The displacement of every node without a trusted fit is now the exact
harmonic extension of the trusted displacements around it: the discrete
Laplace equation is solved by conjugate gradients over each connected region
of untrusted nodes, with the trusted nodes around it as boundary values
([fill.rs](../crates/cortical-bone-mapping/src/refine/fill.rs)). A region
moves as a membrane, and by the maximum principle no node moves further than
its boundary. A region with no trusted node around it stays put.

A seventh trust test was tried and dropped: a volumetric node whose data's
share of the curvature of the cost in `m` was below ½. The share as computed
divides the node's data curvature by the diagonal of the penalty, which is
large even for a penalty that only resists bumps; the median share was
0.002–0.03 with every penalty, so every node failed and the surface did not
move.

Tests: a band 61 nodes wide follows its moved boundary completely (the 50
Jacobi sweeps left its middle behind); linear boundary values are
interpolated linearly; a region without a boundary stays put.

### Results

On the notch the regularised refinement converges in 4 moves as before.
The volumetric refinement, curvature at 1 mm, capped at 15 moves (a run that
has not converged by then does not count as solved):

| penalty | time | converged | notch left (cm³) | bone outside (cm³) | last RMS / regional RMS (mm) | untrusted nodes |
|---|---:|---|---:|---:|---|---:|
| change, weight 0.3 | 303 s | no | 0.26 | 2.04 | 0.044 / 0.25 | 2446 |
| absolute, κ = 0.3, weight 0.3 | 509 s | no | 0.30 | 0.99 | 0.032 / 0.13 | 5975 |
| variation, weight 0.1 | 951 s | no | 2.44 | 0.01 | 0.046 / 0.41 | 1372 |

None converged. In each, the surface as a whole settled below the tolerance
(0.062 mm RMS) within a few moves, but some region kept moving faster than
the regional tolerance (0.125 mm RMS) to the end. The variation penalty left
most of the notch filled. Of the six causes, the penalty and the slow fill
are therefore not what keeps the volumetric refinement from converging;
the PSF re-fit (phase 3) and the stiff, budget-limited full fits (phase 4)
are next.

## 3. PSF and cortical density

### Methods

Planned: fit the PSF once to all bones on their starting surfaces, hold it
and `ρc` through a refinement, re-fit both on the refined surfaces and
relax, `σ ← σ + α (σ_fit − σ)` with `α = ½`, and repeat until the PSF
changes by less than 1 %, as in the alternating solves of SIMPLE-type flow
solvers.

### Results

## 4. Parameters for the scanner

### Methods

Planned: the curvature scale from the node spacing, the density-field
smoothing in millimetres from the voxel size, the band from the PSF, and
inexact inner solves while the surface still moves.

### Results

## 5. Voxels along the normals

### Methods

Planned: assign each band voxel to the face whose prism along the
interpolated vertex normals contains it (shell maps; Porumbescu et al. 2005),
cap each side of a concavity at half the distance to the facing surface, and
extend a node's band outward until the image has passed the bone edge.

### Results

## 6. Weights and thresholds

### Methods

Planned: choose the penalty, its weight and threshold on the knee CT, by eye
on sections and by the notch and edge measures, with the phantoms as a check.

### Results

## References

* H. P. Moreton, C. H. Séquin, *Functional optimization for fair surface
  design*, Computer Graphics (Proceedings of SIGGRAPH 92) 26(2):167–176,
  1992.
* N. J. Neeteson, B. A. Besler, D. E. Whittier, S. K. Boyd, *Automatic
  segmentation of trabecular and cortical compartments in HR-pQCT images
  using an embedding-predicting U-Net and morphological post-processing*,
  Scientific Reports 13:252, 2023, doi:10.1038/s41598-022-27350-0.
* S. D. Porumbescu, B. Budge, L. Feng, K. I. Joy, *Shell maps*, ACM
  Transactions on Graphics 24(3):626–633, 2005.
