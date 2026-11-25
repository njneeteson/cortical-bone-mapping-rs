# Automatic segmentation

CBM needs a segmentation of the bone's outer surface, but it does not need
an exact one. Refinement starts from a rough segmentation that contains the
bone, fits, moves the surface onto the fitted periosteum, and repeats until
the surface stops moving. The endosteum comes from the fitted thickness.

```python
start = cbm.hull_segmentation(image)                     # threshold 320 + per-slice convex hull
options = cbm.Options.preset("radius_tibia", rho_c="estimate")
rr = cbm.refine_segmentation(image, start, options=options, method="volumetric")
```

## Starting segmentations

The start should contain the bone and should not leak into the marrow or a
neighbouring bone.

{func}`~cortical_bone_mapping_rs.hull_segmentation`
: Threshold, binary median to remove speckle, the largest 3D component, then
  the convex hull of every slice along axis 0. A hull cannot leak through a
  porous cortex into the marrow; its errors are bulges over concave parts of
  the periosteum, which refinement removes. For one bone, such as a distal
  radius or tibia; the largest component drops the ulna or fibula.

{func}`~cortical_bone_mapping_rs.bone_hulls`
: Per-bone starts for several bones, kept apart. With `closing_radius`
  (e.g. 4 mm) each bone's hull is replaced by a "local convex hull" that
  keeps wide concavities such as the femoral notch. See
  [several bones](several_bones.md).

Your own segmentation
: Any mask works as a start, e.g. one from a threshold or a manual contour.

{func}`~cortical_bone_mapping_rs.buie_segmentation` implements the
dual-threshold method of Buie et al. (2007) for comparison. It returns
periosteal, cortical and trabecular masks.

## Which fit to refine with

`refine_segmentation` takes every option of the fit. On phantoms built from
published HR-pQCT values, refinement converged from a convex hull with every
method; the volumetric model was the most accurate (thickness RMSE 34–47 µm,
against 70–147 µm for Buie's method) and was not drawn towards a
neighbouring bone 0.5 mm away
([automatic segmentation](../segmentation.md)). The regularised fit with
the curvature penalty is cheaper:

```python
rr = cbm.refine_segmentation(image, start, options=options,
                             curvature="change", curvature_weight=1.0)
```

These phantom results show that the pipeline works; its accuracy on real
HR-pQCT scans is still to be measured.

## Refinement options

Refinement options go in a {class}`~cortical_bone_mapping_rs.RefineOptions`
or as keywords:

```python
refine = cbm.RefineOptions(refine_iterations=50, relaxation=0.5)
rr = cbm.refine_segmentation(image, start, options=options, refine=refine)
```

| option | default | |
|---|---|---|
| `refine_iterations` | 20 | maximum number of moves |
| `relaxation` | 0.5 | fraction of the fitted displacement applied per move |
| `tolerance` | a tenth of a voxel | convergence: RMS displacement of a full fit |
| `level_set` | on | carry the surface as a level set and remesh every move, so it cannot fold |
| `freeze` | on | stop refitting regions that have converged |
| `keep_history` | off | keep every intermediate surface (`rr.surfaces`) |
| `verbose` | on | one line per move on stderr |

Level-set remeshing and freezing are on by default because they made
refinement converge on real data
([refinement experiments](../refinement_experiments.md)).

The scanner's parameters (an estimated `ρc`, the volumetric PSF) are held
fixed during a pass and re-estimated on the refined surface between passes
(`scanner_passes`, default up to 3).

## Convergence

```python
rr.converged          # a full fit moved the surface less than the tolerance
rr.stalled            # it stopped making progress
rr.message            # why it stopped, with a diagnosis if it stalled
rr.history            # one dict per fit: displacement, thickness, nodes still moving
```

Refinement has converged only when a full, converged fit moves the surface
by less than the tolerance overall and in every region. If it hits
`refine_iterations` first, the masks are still the last surface; check
`rr.message`.

## Excluding other structures

`exclude` marks voxels that belong to something else, such as another bone.
Profile samples there get no weight, and the surface never moves into them:

```python
rr = cbm.refine_segmentation(image, start, options=options, exclude=other_bones)
```

{func}`~cortical_bone_mapping_rs.refine_bones` does this for you, bone by bone.
