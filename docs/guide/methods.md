# Choosing a method

All methods fit the same profile model; they differ in how the nodes are
tied together. Set the method with `method=`.

| method | what it fits | use it for |
|---|---|---|
| `"local"` | each node's profile on its own (Treece et al. 2010) | a quick look, or a comparison with the original method |
| `"multigrid"` | `m, t, σ` on sparse control points, coarse to fine (Neeteson & Boyd 2025) | smooth thickness maps at low cost |
| `"regularized"` | `m, t, σ` per node with a spatial smoothness penalty (Neeteson & Boyd 2025) | the default; the presets use it |
| `"volumetric"` | every voxel of a band around the surface, with one PSF for the scan (extension) | the most accurate thickness, at a few times the cost |

On smooth phantoms the spatially constrained methods have 1.5–5 times less
thickness error than the local fit ([examples](../examples.md)). On phantoms
of the distal radius and tibia at HR-pQCT resolution the volumetric model is
nearly unbiased, with about a third of the regularised fit's error, and at
clinical resolution it is slightly more accurate than the best 1D fit
([volumetric validation](../volumetric_validation.md)). Its accuracy on real
HR-pQCT scans is still to be measured.

## Cost

On 134 000 surface nodes (a cylinder 8 mm in radius and 128 slices long at
HR-pQCT resolution, 16 threads), the 1D methods take 2–16 s and 0.1–0.8 GB,
and the volumetric model 51 s and 2 GB ([performance](../performance.md)).
Memory grows with `reg_neighbours`; `reg_neighbours=64` uses less than half
the memory of the paper's 500 at nearly the same accuracy. Refinement repeats the fit a few to a
dozen times, though later fits are cheaper.

## The regularised method

Smoothing weight `reg_lambda` (default 1) over a Gaussian of `reg_sigma`
(1.94 mm) across the `reg_neighbours` nearest nodes:

```python
res = cbm.cortical_bone_mapping(image, mask, spacing, method="regularized",
                                reg_lambda=1.0, reg_sigma=1.94, reg_neighbours=500)
```

`neighbour_distance="surface"` measures distances along the surface rather
than in a straight line, so the two walls of a narrow notch or the two
cortices of a thin plate are not smoothed together.

### Curvature penalty

An optional penalty on the curvature of the fitted periosteal and endosteal
surfaces (extension) keeps them from following noise:

| `curvature=` | penalises |
|---|---|
| `"change"` | curvature added relative to the input surface; uniform offsets are free |
| `"absolute"` | curvature above `curvature_kappa` (default 5 mm⁻¹) |
| `"variation"` | how fast the curvature changes along the surface |

```python
res = cbm.cortical_bone_mapping(image, mask, spacing, method="regularized",
                                curvature="change", curvature_weight=1.0)
```

With `reg_lambda=0` the curvature penalty is the only regulariser.
[Method: curvature penalty](../method.md#curvature-penalty) gives the
definitions.

## The volumetric model

The volumetric model fits the voxels of a band `[-outside, inside]` around
the surface rather than one profile per node. The blur is one Gaussian PSF
for the whole scan, in the plane of the slices (`sigma_xy`) and along the
scanner axis (`sigma_z`):

```python
res = cbm.cortical_bone_mapping(image, mask, spacing, method="volumetric",
                                vol_scanner_axis=0)
res.volumetric["sigma_xy"], res.volumetric["sigma_z"]   # the fitted PSF
res.volumetric["std_error_t"]                           # per-node standard error of t
```

- The blur along the scanner axis is only measurable where the surface faces
  that axis. For a straight shaft (all normals in the plane of the slices),
  tie it to the in-plane blur with `vol_sigma_z="tied"`, or fix it with a
  number.
- If you know the scanner's PSF, give it (`vol_sigma_xy=`, `vol_sigma_z=`)
  and it is not fitted.
- `m` and `t` are smoothed as in the regularised method (`vol_edge_lambda`,
  default 1) and carry a `"change"` curvature penalty
  (`vol_curvature_weight`, default 0.3).
- `res.volumetric` also holds the noise level and, per node, the share of
  the fit that comes from the data and the misfit relative to the noise, to
  find where a value is poorly determined.

[Method: volumetric model](../method.md#volumetric-model) and the
[design note](../volumetric_model.md) describe it in full.

## Blur in the 1D methods

| `sigma=` | |
|---|---|
| `"free"` | fitted per node (default) |
| `"fixed"` | `sigma_value` everywhere |
| `"psf"` | from a Gaussian PSF `psf` (per axis) along each normal |
| `"estimate_psf"` | estimate the PSF from a first fit, then fix it (as Treece et al.) |

## Cortical density

The cortical density `ρc` is never fitted per node. Give it if you know it
(`rho_c=1000.0`, Table 1 of our paper for calibrated HR-pQCT), or use
`rho_c="estimate"`, one value estimated from the nodes whose cortex is
thick enough to reach its density. `rho_c="profile_max"` takes each node's
profile maximum, as the original Python implementation does; a thin,
blurred cortex never reaches its density, so it overestimates thin cortices.
[Method: cortical density](../method.md#cortical-density) gives the details.
