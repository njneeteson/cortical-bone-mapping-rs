# Quick start

This page runs the whole pipeline on a phantom with a known cortex, so every
step can be checked against the truth. Each example takes a few seconds.

```python
import numpy as np
import cortical_bone_mapping_rs as cbm
```

## Coordinates

Images are NumPy arrays, and every coordinate is physical (e.g. mm) in
array-axis order: voxel `(i, j, k)` of an array indexed `(z, y, x)` is at
`origin + (i, j, k) * spacing`. Axis 0 is taken to be the scanner's axis
(the slice direction).

## Thickness on a segmentation

`Phantom` makes a cylindrical shell of known, varying thickness, blurred and
with noise, together with a mask of its outer surface:

```{doctest}
>>> ph = cbm.Phantom(noise=20.0, seed=1)
>>> image, mask = ph.generate()
>>> image.shape, ph.spacing
((24, 96, 96), (0.1, 0.1, 0.1))
```

{func}`~cortical_bone_mapping_rs.cortical_bone_mapping` extracts the mask's
surface, samples a profile along the inward normal at every vertex and fits
the model:

```{doctest}
>>> res = cbm.cortical_bone_mapping(
...     image, mask, spacing=ph.spacing,
...     outside=1.5, inside=2.5,           # profile extent, mm
...     normal_constraint="in_plane",      # normals in the plane of each slice
...     method="regularized")
>>> res.points.shape == (res.n, 3)
True
>>> err = res.thickness - ph.true_thickness(res.points)
>>> bool(np.sqrt(np.mean(err ** 2)) < 0.02)    # RMSE under 20 µm
True
```

The global fits print their progress to stderr; `progress_every=0` silences
them, and `verbose=False` silences refinement.

`res.thickness` and `res.center` are the fitted `t` and `m` per node,
`res.points` and `res.normals` the nodes and their inward normals. The
fitted surfaces and masks follow from them:

```{doctest}
>>> peri, endo, cortex = res.masks(image.shape)
>>> peri.dtype, bool(cortex.any())
(dtype('bool'), True)
>>> res.endosteal_points.shape == res.points.shape
True
```

[Reading the results](results.md) covers every output, including saving the
thickness map for a mesh viewer.

## From a rough segmentation

A segmentation is rarely exact. This phantom's mask lies 0.3 mm outside the
bone with lobed bumps; refinement fits, moves the surface onto the fitted
periosteum, and repeats until it stops moving:

```{doctest}
>>> ph = cbm.Phantom(noise=20.0, seed=1, mask_offset=0.3, mask_bump=0.15)
>>> image, rough = ph.generate()
>>> rr = cbm.refine_segmentation(
...     image, rough, spacing=ph.spacing, outside=1.5, inside=2.5,
...     normal_constraint="in_plane", method="regularized", verbose=False)
>>> rr.converged
True
>>> truth = ph.true_mask().astype(bool)
>>> def dice(a, b):
...     return 2 * (a & b).sum() / (a.sum() + b.sum())
>>> bool(dice(rough.astype(bool), truth) < 0.93 < 0.99 < dice(rr.periosteal_mask, truth))
True
```

`rr.periosteal_mask`, `rr.endosteal_mask` and `rr.cortical_mask` are the
refined segmentation, and `rr.result` is the fit on the refined surface.

## From the image alone

With no segmentation at all, a threshold and a per-slice convex hull make a
safe start, since the hull contains the bone and does not leak into the
marrow:

```{doctest}
>>> image, _ = cbm.Phantom(noise=20.0, seed=1).generate()
>>> start = cbm.hull_segmentation(image, threshold=500.0)
>>> rr = cbm.refine_segmentation(
...     image, start, spacing=ph.spacing, outside=1.5, inside=2.5,
...     normal_constraint="in_plane", method="regularized", verbose=False)
>>> rr.converged
True
```

The threshold is in image units: 320 mg HA/cm³ is the usual bone threshold
for calibrated HR-pQCT (the default), and this phantom's cortex is 1000.
[Automatic segmentation](refinement.md) covers the starting segmentations
and refinement on real images.

## Options

Every option can be passed as a keyword, as above, or collected in an
{class}`~cortical_bone_mapping_rs.Options`. Presets give the parameters of
Table 1 of our 2025 paper:

```{doctest}
>>> options = cbm.Options.preset("radius_tibia", rho_c="estimate")
>>> options.method, options.outside, options.inside
('regularized', 5.0, 15.0)
```

[Scanners and parameters](scanners.md) explains the presets and what to
change for other data.
