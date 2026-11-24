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
| 3 | PSF and `ρc` in an outer loop, held during each refinement | done; no effect on the knee, whose volumetric run never settles |
| 4 | parameters scaled to the scanner and the node spacing | done; 40 % less time per move, no change in convergence |
| 5 | adaptive band | done; no effect: the surface sinks into a dark region inside the condyle |
| 6 | edge smoothing of `m`, `t` for the volumetric model; `ρc` | done; with both, the volumetric knee notch converges (11 moves) |
| 7 | weights and thresholds chosen on the knee | planned |

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

The scanner's parameters (the volumetric model's PSF, a global `σ`, an
estimated `ρc`) are now held through a refinement *pass*, in full fits as
well as partial ones, instead of being re-fitted in every full fit
([scanner.rs](../crates/cortical-bone-mapping/src/refine/scanner.rs)).
After a pass they are re-fitted on the refined surface and moved part of the
way, `p ← p + β (p_fit − p)` with `β = ½`, and another pass follows while a
re-fit changes them by more than 1 %, up to 3 passes
(`scanner_passes`, `scanner_relaxation`, `scanner_tolerance`), as segregated
flow solvers alternate between their unknowns (Patankar & Spalding 1972).
A PSF or `ρc` given in the options re-fits to itself, so refinement then
stops after one pass. `RefineResult.scanner` lists the values held in each
pass.

While looking at why the volumetric notch did not converge, a second option
was added: one global soft-tissue density with a per-node trabecular field
(`vol_global_soft_tissue`).

### Results

On the notch the regularised refinement converges in 4 moves as before; the
re-fit after its pass changed `ρc` by 0.2 % (1269 → 1270 HU), so one pass
sufficed.

The volumetric refinement (curvature `"change"`, 1 mm) behaved exactly as
before: it never reached a second full fit, because the surface never
settled. The region that keeps moving does not oscillate. About 1000 nodes,
13 × 18 × 10 mm, move inward by 0.24 mm every move: the surface there lies
3.4 mm inside the regularised one, inside the femur, in trabecular bone
(about 260 HU). The fits there have `t` at its lower bound, a soft-tissue
density of 219 HU and a trabecular density of 310 HU: once the band lies
inside the bone, the per-node soft-tissue density takes the value of bone,
the fit sees no edge, yet passes every trust test, and its `m` keeps pulling
the surface inward. This is the case where bone is out of reach of the band,
but the node is not recognised as untrusted.

With one global soft-tissue density the sinking stops (bone outside 0.02
cm³, against 2.04), but the notch no longer empties (2.12 cm³ left, against
0.26): below a surface bridging the notch, the per-node trabecular density
takes the value of the soft tissue in the notch (median 130 HU), and the fit
again sees no edge. Neither run converged in 15 moves.

| volumetric, 15 moves | converged | notch left (cm³) | bone outside (cm³) | last RMS / regional RMS (mm) |
|---|---|---:|---:|---|
| per-node densities | no | 0.26 | 2.04 | 0.044 / 0.25 |
| global soft tissue | no | 2.12 | 0.02 | 0.034 / 0.22 |

Both failures are a band that does not reach the tissue on the other side of
the edge, with a density field free to absorb what it sees instead. That is
the target of phase 5 (voxels assigned along the normals, a band that
reaches past the edge).

## 4. Parameters for the scanner

### Methods

* **Curvature scale.** The curvature penalties now measure curvature at a
  multiple of the mean node spacing by default (`LaplacianScale::NodeSpacing`,
  `curvature_scale_spacings` = 4), instead of a fixed 0.25 mm. At HR-pQCT
  resolution that is about 0.25 mm, so those results do not change; on the
  knee CT (nodes about 0.65 mm apart) it is about 2.6 mm instead of a scale
  finer than the mesh. A physical scale (`curvature_scale`, mm) or the
  cotangent Laplacian (`"cotangent"`) can still be given.
* **Inexact inner fits skip the reliability outputs.** The noise estimate,
  standard errors and data shares cost about a tenth of a volumetric fit and
  nothing in a refinement uses them before the last fit; the inexact inner
  fits now compute only the per-node residuals.

The density fields' smoothing was left as it is: its neighbourhood is cut at
`3 σ_reg`, which already makes its reach physical.

### Results

Volumetric notch refinement, 15 moves (`"change"`, weight 0.3):

| | time in fits | time per move | reliability |
|---|---:|---:|---:|
| curvature at 1 mm (phase 2) | 303 s | 20 s | |
| curvature at 4 node spacings (2.6 mm) | 220 s | 14 s | 19 s |
| and residuals only in inner fits | 184 s | 12 s | 8 s |

The inner fits take about 70–80 conjugate-gradient iterations each. The
surface moves exactly as before, and still does not converge. On the healthy
shaft phantom both methods converge in 6–7 moves with the same accuracy as in
phase 1 (volumetric thickness RMSE 0.033 mm).

## 5. The band

### Methods

Each node's band can now reach further than the sampling range, along its
normal, until it has crossed a bone edge (`vol_adaptive_band`, a cap as a
multiple of the sampling range;
[reach.rs](../crates/cortical-bone-mapping/src/fit/volumetric/reach.rs)):
outward past bright bone into soft tissue, where the surface lies inside the
bone, and inward through soft tissue to the next bone, where it bridges a
concavity. The levels come from the profiles: soft tissue `s` is the median
of the outer half of the outside range, bone `c` the 90th percentile inside,
`low = s + ¼ (c − s)` and `high = s + ½ (c − s)`. The trust tests use each
node's actual reach (`volumetric["reach"]`).

Assigning voxels along the normals (shell maps) was not implemented. The
closest-point assignment already splits a concavity at its medial axis (a
voxel between two walls belongs to the nearer one), which is what capping
each side at half the distance to the facing surface would do; the two
differ only around convex corners, where neither failure occurred.

### Results

| volumetric, 15 moves | converged | notch left (cm³) | bone outside (cm³) | last regional RMS (mm) |
|---|---|---:|---:|---:|
| fixed band | no | 0.38 | 2.23 | 0.27 |
| adaptive band (up to 2×) | no | 0.42 | 2.13 | 0.32 |
| adaptive band, global soft tissue | no | 2.20 | 0.09 | 0.37 |

The adaptive band changed nothing. At the sinking nodes it was rarely
extended (27 % of them, against 84 % of all nodes), and the sections show
why: the region the surface sinks into is not trabecular bone behind a
cortex but a dark area inside the lateral femoral condyle, next to the
notch, at 100–170 HU. Along the regularised surface's normal there, the
profile stays at 100–170 HU from 6 mm outside to the surface and rises
slowly to 340 HU 9 mm inside, with no cortex to cross; the regularised
refinement also cuts into it, less deeply. Neither the levels nor the model
can tell this region from soft tissue, so the surface keeps growing a pocket
into it. Whether it belongs inside the bone is an anatomical question about
this region, not one the band can settle.

![Sections through the region](images/volumetric_refinement_pocket.png)

*Axial, coronal and sagittal sections through the centre of the sinking
region (green cross) of the femur crop: the 4 mm closing start (yellow,
dashed), the refined periosteal surface (red) and the fitted endosteal
surface (cyan). Top: regularised refinement, converged in 4 moves. Bottom:
volumetric refinement with the adaptive band after 15 moves, not converged.
Everything more than 10 mm outside the start is blanked (display only).
`python docs/scripts/notch_experiment.py knee.pkl --out results figures
closing_freeze closing_freeze_volumetric_adaptive --centre 45,95,124
--figure volumetric_refinement_pocket.png`.*

The region is inside the bone (low-density marrow in the condyle, not the
notch), so the periosteal surface should not enter it: this is what the
regularisation is meant to prevent, and the regularised 1D refinement
prevents it. Section 6 finds why the volumetric one does not.

## 6. Edge smoothing and cortical density

### Methods

The two refinements differed in more than the data model. The regularised
1D fit smooths the edge fields `m` and `t` with the spatial penalty of our
2025 paper (`λ ‖ω m‖²`, `λ ‖ω t‖²`; on the knee λ = 1, σ_reg = 3 mm, 500
neighbours) and has no curvature penalty; the volumetric fit smoothed only
its density fields, and its edges only through the curvature penalty. The
volumetric model now has the same smoothing of `m` and `t`
(`vol_edge_lambda`, same operator and weighting `λ ρc² / (N t0²)`; default 0,
as before). The two fits also estimated different cortical densities on this
knee: 1270 HU (1D) and 860–884 HU (volumetric).

Each variant below changes one thing from the one above it or from the
regularised refinement, all from the 4 mm closing with the level set and
regional convergence, capped at 15 moves.

### Results

| refinement | edge smoothing | curvature penalty | `ρc` | moves | converged | notch left (cm³) | bone outside (cm³) | convex distance (mm) | median thickness (mm) |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| regularised (1D) | λ = 1 | none | estimated, 1270 | 4 | yes | 0.00 | 1.12 | 1.03 | 0.57 |
| volumetric | none | change, 0.3 | estimated, ≈ 880 | 15 | no | 0.38 | 2.23 | 0.84 | 0.73 |
| volumetric | λ = 1 | none | estimated, 884 | 15 | no | 0.01 | 1.66 | 0.88 | 0.72 |
| volumetric | λ = 1 | change, 0.3 | estimated | 15 | no | 0.01 | 1.54 | 0.88 | 0.72 |
| volumetric | none | change, 0.3 | fixed, 1270 | 15 | no | 0.22 | 2.22 | 0.99 | 0.50 |
| volumetric | λ = 1 | none | fixed, 1270 | 11 | **yes** | 0.00 | 1.26 | 1.02 | 0.50 |

"Convex distance" is the median distance from the refined surface to the
thresholded surface where the start already lay on it.

Neither change alone was enough; together they made the volumetric
refinement converge on the knee for the first time, in 11 moves (137 s),
with the notch emptied and no pocket in the condyle. The edge smoothing
holds a node's edge near its neighbours', as in the 1D fit; the curvature
penalty at weight 0.3 did not. With `ρc` estimated at about 880 HU the faint
transition into the dark marrow fits as a thin cortex, and the surface
follows it in; at 1270 HU it does not. The low estimate is the known
weakness of the volumetric `ρc` estimate, which needs cortex thicker than
about four times the blur in view ([volumetric_validation.md](volumetric_validation.md)):
this crop of the femur has little of it, while the 1D estimate comes from
the whole femur's profiles.

![One change at a time](images/volumetric_refinement_regularisation.png)

*Sections through the region of section 5 (green cross): the regularised
refinement, then the volumetric refinement with edge smoothing, with `ρc`
fixed, and with both (bottom). Start yellow and dashed, refined periosteal
surface red, endosteal cyan; everything more than 10 mm outside the start
is blanked (display only). `python docs/scripts/notch_experiment.py knee.pkl
--out results figures closing_freeze closing_freeze_volumetric_edge_smoothing
closing_freeze_volumetric_rho_c closing_freeze_volumetric_edge_smoothing_rho_c
--centre 45,95,124 --figure volumetric_refinement_regularisation.png`.*

What remains open: a `ρc` for the volumetric model that does not depend on
borrowing the 1D estimate (for example the shared scanner value of all the
bones, estimated where the cortex is thick), and the weights, in section 7.

## 7. Weights and thresholds

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
* S. V. Patankar, D. B. Spalding, *A calculation procedure for heat, mass
  and momentum transfer in three-dimensional parabolic flows*,
  International Journal of Heat and Mass Transfer 15(10):1787–1806, 1972.
* S. D. Porumbescu, B. Budge, L. Feng, K. I. Joy, *Shell maps*, ACM
  Transactions on Graphics 24(3):626–633, 2005.
