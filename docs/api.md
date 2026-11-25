# API reference

Everything is available from the top-level package:

```python
import cortical_bone_mapping_rs as cbm
```

```{eval-rst}
.. currentmodule:: cortical_bone_mapping_rs
```

## Measuring thickness

```{eval-rst}
.. autofunction:: cortical_bone_mapping
.. autofunction:: map_bones
.. autofunction:: shared_scanner
.. autofunction:: mask_surface
```

## Segmentation refinement

```{eval-rst}
.. autofunction:: refine_segmentation
.. autofunction:: refine_bones
```

## Options

```{eval-rst}
.. autoclass:: Options
   :no-members:

.. automethod:: Options.preset

.. autoclass:: RefineOptions
   :no-members:

.. autodata:: PARAM_NAMES
   :no-value:
```

## Results

```{eval-rst}
.. autoclass:: CbmResult
   :members:
   :exclude-members: points, normals, faces, params, residual, vertex_ids, surface_vertices, surface_faces, global_params, psf, rho_c, info, options, profile_x, profiles, rho_c_node, volumetric

.. autoclass:: RefineResult
   :members:
   :exclude-members: result, periosteal_mask, endosteal_mask, cortical_mask, endosteal_vertices, history, surfaces, converged, surfaces_faces, stalled, message, timings, scanner
```

## Starting segmentations

```{eval-rst}
.. autofunction:: hull_segmentation
.. autofunction:: otsu_thresholds
.. autofunction:: label_bones
.. autofunction:: separate_bones
.. autofunction:: name_bones
.. autofunction:: soft_tissue_threshold
.. autofunction:: grow_bones
.. autofunction:: bone_hulls
.. autofunction:: buie_segmentation
.. autoclass:: BuieOptions
   :no-members:
```

## Meshes

```{eval-rst}
.. autofunction:: write_vtk
.. autofunction:: voxelize
.. autofunction:: mesh_section
.. autofunction:: mesh_curvature
```

## The profile model

```{eval-rst}
.. autofunction:: profile_model
.. autofunction:: fit_profiles
```

## Phantoms

Synthetic images with exact ground truth, for testing.

```{eval-rst}
.. autoclass:: Phantom
   :no-members:

.. automethod:: Phantom.generate
.. automethod:: Phantom.true_thickness
.. automethod:: Phantom.true_mask
.. automethod:: Phantom.radial_offset

.. autoclass:: BonePhantom
   :no-members:

.. automethod:: BonePhantom.generate
.. automethod:: BonePhantom.labels
.. automethod:: BonePhantom.edges_along
.. automethod:: BonePhantom.true_thickness

.. autoclass:: FemurPhantom
   :no-members:

.. automethod:: FemurPhantom.generate
.. automethod:: FemurPhantom.edges_along
.. automethod:: FemurPhantom.true_thickness

.. autofunction:: site_phantom

.. autodata:: SITES
   :no-value:
```
