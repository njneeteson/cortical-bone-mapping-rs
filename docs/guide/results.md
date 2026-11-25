# Reading the results

## A fit

{func}`~cortical_bone_mapping_rs.cortical_bone_mapping` returns a
{class}`~cortical_bone_mapping_rs.CbmResult` with one row per measured
surface node:

```python
res.points, res.normals           # (N, 3) nodes and inward unit normals, physical coordinates
res.thickness                     # (N,) cortical thickness t
res.center                        # (N,) position m of the cortical mid-line along the normal
res.sigma, res.rho_s, res.rho_b   # (N,) blur, soft-tissue and trabecular densities
res.params                        # (N, 5) all of them: m, t, σ, ρs, ρb
res.residual                      # (N,) weighted RMS misfit of each profile
res.rho_c                         # the cortical density used
res.faces                         # (F, 3) triangles between the nodes
```

The periosteal and endosteal edges lie at `m - t/2` and `m + t/2` along each
normal (positive inwards):

```python
res.periosteal_points, res.endosteal_points            # (N, 3)
peri, endo, cortex = res.masks(image.shape)            # boolean masks of the fitted surfaces
```

Not every surface vertex is a node: vertices on caps where the image cuts
the bone (`exclude_border`) and outside a `sub_mask` are not fitted.
`res.surface_vertices` and `res.surface_faces` hold the whole surface, and
`res.vertex_ids` maps each node to its vertex.

With `keep_profiles=True`, `res.profiles` holds the sampled profiles at
positions `res.profile_x`, and `res.model(i)` the fitted model of node `i`,
to check a fit by eye.

`res.info` holds diagnostics: iterations, costs, convergence and timings.
For the volumetric model, `res.volumetric` holds the fitted PSF, the noise
level and per-node reliability (`std_error_t`, `data_share`, `misfit`).

## Refinement

{func}`~cortical_bone_mapping_rs.refine_segmentation` returns a
{class}`~cortical_bone_mapping_rs.RefineResult`:

```python
rr.periosteal_mask, rr.endosteal_mask, rr.cortical_mask   # boolean, image shape
rr.result                                                 # the CbmResult of the final fit
rr.periosteal_vertices, rr.endosteal_vertices             # the refined surfaces
rr.converged, rr.message, rr.history
```

## Saving and viewing

The thickness map as a mesh with per-node arrays, for ParaView or 3D Slicer:

```python
res.save_vtk("thickness.vtk")
mesh = res.to_pyvista()           # with PyVista installed
```

{func}`~cortical_bone_mapping_rs.write_vtk` writes any mesh with your own
arrays, and {func}`~cortical_bone_mapping_rs.mesh_section` cuts a mesh by an
image plane to draw it over a slice:

```python
import matplotlib.pyplot as plt

k = image.shape[0] // 2
seg = cbm.mesh_section(res.surface_vertices, res.surface_faces, axis=0, index=k, spacing=spacing)
plt.imshow(image[k], cmap="gray")
for s in seg:
    plt.plot(s[:, 1], s[:, 0], "r-", lw=0.8)
```

{func}`~cortical_bone_mapping_rs.voxelize` turns a closed mesh into a mask.

## Summaries

Thickness is per node. The nodes are the vertices of a mesh extracted on
the image grid, so they are spread roughly evenly over the surface, and
plain statistics over nodes are close to area-weighted ones:

```python
np.median(res.thickness), np.percentile(res.thickness, [5, 95])
```

To restrict a summary to a region, pass a `sub_mask` to the fit, or select
nodes by position.
