# cortical-bone-mapping-rs

Cortical bone mapping (CBM) in Rust, with Python bindings: model-based
measurement of cortical thickness, and automatic segmentation of the
periosteal and endosteal surfaces, from CT and HR-pQCT images.

CBM samples a density profile along the inward normal at every vertex of
the bone's outer surface and fits a blurred three-level step model (soft
tissue, cortex, trabecular bone). At every surface point the result is the
cortical **thickness** `t`, the position `m` of the cortical mid-line, the
blur `σ`, and the soft-tissue and trabecular densities `ρs`, `ρb`; the
cortical density `ρc` is fixed or estimated from the image.

![Profile fit](docs/images/profile_fit.png)

This crate implements

* the original method of Treece et al. (2010) (one fit per node), with fixed,
  PSF-based or estimated blur;
* the global **multigrid** and spatially **regularised** fits of Neeteson &
  Boyd (2025), with a much faster solver;
* three extensions:
  * a **curvature penalty** on the fitted surfaces;
  * a **volumetric model**, which fits the voxels of a band around the
    surface with a global PSF instead of one profile per node;
  * **segmentation refinement**, which moves a rough periosteal
    segmentation onto the fitted periosteum until it stops moving.

![A rough segmentation (red) tightening onto the periosteum](docs/images/refinement.gif)

*Refinement on a phantom whose input mask is dilated by 0.45 mm with 7-lobed
bumps. Left, a slice with the segmentation (red) and the true periosteum
(green, dashed); right, the surface coloured by its distance from the truth.*

## Install

```bash
pip install ./python          # builds the Rust extension (needs a Rust toolchain)
```

## Usage

From the image alone, an automatic segmentation and a thickness map
(see [docs/segmentation.md](docs/segmentation.md)):

```python
import cortical_bone_mapping_rs as cbm

start = cbm.hull_segmentation(image)             # threshold 320 + per-slice convex hull
options = cbm.Options.preset("radius_tibia", rho_c="estimate")
rr = cbm.refine_segmentation(image, start, options=options,
                             curvature="change", curvature_weight=1.0)
rr.periosteal_mask, rr.endosteal_mask, rr.cortical_mask
rr.result.thickness
```

Several bones in one image (e.g. a clinical knee CT; see
[docs/clinical_ct.md](docs/clinical_ct.md)). Each bone is its own closed
surface and its own fit:

```python
labels, sizes = cbm.label_bones(image, threshold="otsu", otsu_floor=-500)  # or a threshold in image units
# optional: name bones by landmarks; bones the threshold joined are split
labels, names, used = cbm.separate_bones(image, 300.0, {"femur": p1, "tibia": p2}, spacing)
starts = cbm.bone_hulls(labels, spacing, closing_radius=4.0)  # per-bone "local convex hulls", kept apart
options = cbm.Options(..., global_params=())         # per-node densities for a large field of view
refine = cbm.RefineOptions(level_set=True, freeze=True)  # remeshing, regional convergence
results = cbm.refine_bones(image, starts, spacing, options=options, refine=refine, names=names)  # smallest first
```

Thickness on a given segmentation (1 inside the periosteal surface):

```python
res = cbm.cortical_bone_mapping(
    image, mask, spacing=(0.0607, 0.0607, 0.0607),  # array-axis order (z, y, x)
    method="regularized",           # "local" | "multigrid" | "regularized" | "volumetric"
    normal_constraint="in_plane",   # transverse-plane normals for radius/tibia
    sub_mask=None,                  # optionally restrict where thickness is reported
)
res.points, res.normals          # (N, 3) surface nodes and inward normals
res.thickness, res.center        # (N,) fitted t and m
res.sigma, res.rho_s, res.rho_b  # other fitted parameters
res.periosteal_points, res.endosteal_points
res.save_vtk("thickness.vtk")    # or res.to_pyvista()
peri, endo, cortex = res.masks(image.shape)
```

Every option (`help(cbm.Options)`) can be passed as a keyword or through an
`Options` object:

```python
# Treece-style fixed blur, estimated as an axial PSF
cbm.cortical_bone_mapping(image, mask, spacing, sigma="estimate_psf", psf_model="axial", psf_axis=0)

# multigrid with control-point spacings in mm
cbm.cortical_bone_mapping(image, mask, spacing, method="multigrid", multigrid_spacings=(4, 2, 1))

# curvature penalty, alone or with the smoothing term
cbm.cortical_bone_mapping(image, mask, spacing, method="regularized",
                          reg_lambda=0.0, curvature="change", curvature_weight=1.0)

# volumetric model: voxels of the band, global PSF (axis 0 = scanner axis);
# result.volumetric holds the PSF, noise and per-node reliability
cbm.cortical_bone_mapping(image, mask, spacing, method="volumetric", vol_scanner_axis=0)
```

Images may be `float32/64`, `int8/16/32` or `uint8/16/32` and are not copied.
Also available: `cbm.buie_segmentation` (Buie et al. 2007, implemented from
the paper), and phantoms with exact ground truth: `cbm.Phantom` (smooth
shells), `cbm.BonePhantom` (realistic long bones, optionally with a
neighbouring fibula/ulna and cortical breaks), `cbm.JointPhantom` (the
contact region of a knee, optionally with the intercondylar notch),
`cbm.FemurPhantom` (a proximal femur with the acetabulum), and
`cbm.site_phantom(site)` for the distal radius and tibia (HR-pQCT) and the
proximal femur and knee (clinical CT) with parameters from the literature.

From Rust:

```rust
use cortical_bone_mapping::{run, Options, Site};

let opts = Options::preset(Site::RadiusTibia, 0.0607);
let res = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &opts)?;
let t = res.thickness();
```

`cargo run --release --example phantom` runs every method on a phantom,
`--example refine` shows refinement, and `--example benchmark` reports
timings and peak memory.

## Parameters

`cbm.Options.preset("radius_tibia")` and `preset("knee")` give the parameter
sets of Table 1 of Neeteson & Boyd (2025): Taubin smoothing (100 iterations,
pass-band 0.1 or 0.01), sampling 5 mm outside and 15 mm inside, profile
smoothing σ = 3 samples, normals in the transverse plane (radius & tibia) or
along the scan axis (knee), `λ = 1`, `σ_reg = 1.942 mm` (32 voxels), 500 or
1000 regularisation neighbours, multigrid spacings from 128 or 256 down to
16 voxels. Array axis 0 is the scan axis; overrides are keywords. The plain
defaults use the same model guesses and bounds with generic sampling (2 mm
outside, 4 mm inside).

**Cortical density.** `ρc` is never fitted. Give it (`rho_c=1000`, Table 1),
estimate it from the image (`rho_c="estimate"`), or take each node's profile
maximum (`rho_c="profile_max"`, what the original Python implementation does
when no density is given). On realistic phantoms the estimate is the most
robust: the per-node maximum overestimates thin-cortex thickness by ~0.2 mm.
See [method](docs/method.md#cortical-density).

## Results at a glance

* **Accuracy.** On smooth phantoms the spatially constrained methods reduce
  thickness error 1.5–5 times relative to the local fit
  ([examples](docs/examples.md)).
* **Volumetric model.** On phantoms of the distal radius and tibia at HR-pQCT
  resolution, the volumetric model is nearly unbiased, with a third of the
  regularised fit's thickness error and 5–11 times its precision; at
  clinical resolution it is as good as the best 1D fit, and its PSF is
  recovered to within about 10%. Refinement with it is not yet reliable
  ([volumetric validation](docs/volumetric_validation.md)).
* **Automatic segmentation.** On phantoms built from published HR-pQCT
  values, starting from a convex hull, CBM matches or beats Buie's
  dual-threshold method on every shaft: thickness RMSE 28–45 µm for the
  volumetric model on the healthy, osteoporotic and neighbour shafts, and
  40–85 µm for refinement with the curvature penalty where Buie is
  unbiased. It corrects Buie's −0.13 mm bias on a thin porous cortex (cortex
  Dice 0.81 → 0.93–0.96), and the curvature penalty bridges a 1.2 mm
  cortical break that makes Buie's marrow leak into the cortex. The knee's
  subchondral plate is still hard ([segmentation](docs/segmentation.md)).
* **Real data.** On a clinical knee CT, all four bones (femur, tibia,
  patella, fibula) are segmented and mapped from the image alone in 138 s,
  each converging in 5–7 moves with the femoral notch followed
  ([clinical CT](docs/clinical_ct.md),
  [refinement experiments](docs/refinement_experiments.md)).
* **Speed and memory.** Against the original Python implementation (Bonelab)
  on the same profiles, the local fit is about 420 times faster, multigrid
  45 times and the regularised fit 10–20 times (against a converged solve),
  with equal or better accuracy. At 134k nodes the 1D methods take 2–15 s
  and 0.1–0.8 GB, the volumetric model 41 s and 2 GB
  ([performance](docs/performance.md)).

![Methods on a noisy phantom](docs/images/methods_comparison.png)

## Documentation

* **[docs/method.md](docs/method.md)**: the model, every method and the
  solver, cortical density, the extensions, the starting segmentations, the
  memory/performance design, and the deliberate differences from the
  original implementations.
* **[docs/examples.md](docs/examples.md)**: accuracy of every method on
  smooth phantoms, noise sweeps, the curvature penalty, and refinement.
* **[docs/segmentation.md](docs/segmentation.md)**: automatic segmentation
  from the image alone; Buie's method as implemented; the experiment on
  realistic phantoms.
* **[docs/clinical_ct.md](docs/clinical_ct.md)**: real data, a clinical
  knee CT: threshold from the histogram, bone separation, closing starts and
  refinement one bone at a time.
* **[docs/refinement_experiments.md](docs/refinement_experiments.md)**: what
  the knee's femoral notch taught us about refinement on real data:
  remeshing, per-node densities, inexact inner fits, closing starts and
  regional convergence.
* **[docs/volumetric_model.md](docs/volumetric_model.md)**: the design note
  behind the volumetric (voxel-wise) model, which runs alongside the 1D one.
* **[docs/volumetric_validation.md](docs/volumetric_validation.md)**: the
  volumetric model against the 1D methods on phantoms of the distal radius
  and tibia (HR-pQCT) and the proximal femur and knee (clinical CT), and on
  the clinical knee.
* **[docs/volumetric_refinement.md](docs/volumetric_refinement.md)**: work in
  progress on making refinement with the volumetric model converge, phase by
  phase.
* **[docs/performance.md](docs/performance.md)**: run time and memory, and
  comparisons with the original Python implementation and with SciPy.

## Features

* **Methods**: `local` (Treece et al.), `multigrid` and `regularized`
  (Neeteson & Boyd), the latter optionally with the curvature penalty
  (`"change"`: curvature added by the fit; `"absolute"`: curvature above a
  threshold; `"variation"`: how fast the curvature changes), or with the
  curvature penalty alone; `volumetric` (extension),
  with a global PSF, the curvature penalty on the edges, smoothed density
  fields, and per-node reliability (local standard error, data share,
  misfit).
* **Neighbourhoods** by Euclidean distance or by distance along the surface
  (truncated Dijkstra), for every method.
* **Phantoms** with exact ground truth: smooth shells, realistic long bones
  (porous cortex, trabeculae, neighbouring bones, cortical breaks), a knee
  joint, a proximal femur with the acetabulum, and the four sites at their
  scanners' resolution with parameters from the literature
  (`cbm.site_phantom`).
* **Blur**: free, fixed, from a Gaussian PSF (direction-dependent), or a PSF
  estimated from the data (isotropic, axial or anisotropic).
* **Fast and frugal**: bounded Levenberg–Marquardt with analytic Jacobians
  per node; matrix-free Gauss–Newton/PCG with a work budget for the global
  fits; multithreaded; the image read in place; `f32` profiles and compact
  sparse operators.
* **Reproducible**: results are bitwise identical across runs and thread
  counts.
* **Tested**: analytic gradients against finite differences; the local fit
  against SciPy `least_squares`; every method for accuracy on phantoms with
  known geometry (one built independently with SciPy); property tests;
  refinement convergence; Buie's steps against brute force.

## References

> G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
> cortical bone thickness measurement from clinical CT data*, Medical Image
> Analysis 14(3):276–290, 2010.

> G. M. Treece, A. H. Gee, *Cortical Bone Mapping: Measurement and
> Statistical Analysis of Localised Skeletal Changes*, Curr. Osteoporos.
> Rep. 16:617–625, 2018,
> [doi:10.1007/s11914-018-0475-3](https://doi.org/10.1007/s11914-018-0475-3)

> N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
> optimization*, Computers in Biology and Medicine, 2025,
> [doi:10.1016/j.compbiomed.2025.110636](https://doi.org/10.1016/j.compbiomed.2025.110636)

> H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
> *Automatic segmentation of cortical and trabecular compartments based on a
> dual threshold technique for in vivo micro-CT bone analysis*, Bone
> 41:505–515, 2007,
> [doi:10.1016/j.bone.2007.07.007](https://doi.org/10.1016/j.bone.2007.07.007)

## Development

```bash
cargo test --workspace && cargo clippy --all-targets -- -D warnings
cd python && maturin develop --release && pytest tests
python docs/scripts/make_figures.py                 # docs/images
python docs/scripts/segmentation_experiment.py      # docs/segmentation.md results (~70 min)
python docs/scripts/ct_example.py DICOM_DIR ...     # docs/clinical_ct.md (image not included)
python docs/scripts/notch_experiment.py knee.pkl --out results run  # docs/refinement_experiments.md
cargo run --release --example benchmark -p cortical-bone-mapping -- --json docs/results/benchmark_results.json
python docs/scripts/compare_scipy.py
python docs/scripts/compare_bonelab.py /path/to/Bonelab
```

## License

MIT. If you use this software, please cite the papers above and this
implementation (see [CITATION.cff](CITATION.cff)).
