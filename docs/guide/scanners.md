# Scanners and parameters

Lengths are in the units of `spacing` (mm throughout this documentation),
and densities in the image's units: mg HA/cm³ for calibrated HR-pQCT,
Hounsfield units for clinical CT. The model's initial values and bounds
(`init`, `lower`, `upper`, in the order `m, t, σ, ρs, ρb`) are in the same
units, so they need to suit your scanner.

## HR-pQCT

{meth}`Options.preset <cortical_bone_mapping_rs.Options.preset>` gives the
parameter sets of Table 1 of our 2025 paper:

```python
options = cbm.Options.preset("radius_tibia")          # 60.7 µm voxels by default
options = cbm.Options.preset("knee", voxel=0.0607)    # subchondral bone of the knee
```

| | `"radius_tibia"` | `"knee"` |
|---|---|---|
| Taubin smoothing | 100 iterations, pass-band 0.1 | 100 iterations, pass-band 0.01 |
| sampling | 5 mm outside, 15 mm inside, every voxel | same |
| profile smoothing | σ = 3 samples | same |
| normals | in the plane of the slices | along the scanner axis |
| method | regularised, `λ = 1`, `σ_reg = 1.942 mm`, 500 neighbours | 1000 neighbours |
| multigrid spacings | 128 to 16 voxels | 256 to 16 voxels |

Any field can be overridden: `Options.preset("radius_tibia", rho_c="estimate",
method="volumetric")`. The cortical density defaults to 1000 mg HA/cm³, as
in Table 1; `rho_c="estimate"` measures it from the image instead (see
[choosing a method](methods.md#cortical-density)).

## Clinical CT

Clinical CT has voxels about ten times larger, densities in Hounsfield units
and no density calibration. These options worked on the clinical knee CT
of [the clinical example](../clinical_ct.md) (0.65 mm voxels):

```python
options = cbm.Options(
    spacing=spacing,
    outside=4.0, inside=8.0, resolution=0.25,        # mm
    taubin_iterations=50, taubin_passband=0.1,
    method="regularized", reg_sigma=3.0, reg_neighbours=500,
    global_params=(),                                # per-node ρs and ρb
    rho_c="estimate",
    init=(0.0, 2.0, 0.5, 0.0, 100.0),
    lower=(-4.0, 0.05, 0.05, -300.0, -300.0),
    upper=(8.0, 8.0, 2.0, 400.0, 1200.0),
)
```

- Normals are free (`normal_constraint="none"`, the default), since the
  bones' surfaces face every direction.
- `global_params=()` fits the soft-tissue and trabecular densities per
  node, because over a large field of view the trabecular density varies
  (dense epiphyses, fatty marrow in the shaft).
- The bounds on `ρs` and `ρb` are wide enough for Hounsfield units, and the
  lower bound on `m` matches `outside`.
- The cortex is often thinner than the blur, so treat thicknesses below
  about one voxel with care.

## Anisotropic voxels

`spacing` takes one value per array axis, e.g. `(0.625, 0.65, 0.65)` for
0.625 mm slices of 0.65 mm pixels. Profiles are sampled every `resolution`
(default: the smallest spacing). For PSF-based blur, `psf_model="axial"`
(default) has one blur in the plane of the slices and one along `psf_axis`.

## Physical position

`origin` places voxel `(0, 0, 0)`; set it when the image is a crop of a
larger one so that results are in the full image's coordinates.

## Image types

Images may be `float32`, `float64`, `int8/16/32` or `uint8/16/32` and are
read in place without copying. Masks are any integer or boolean array;
non-zero is inside.
