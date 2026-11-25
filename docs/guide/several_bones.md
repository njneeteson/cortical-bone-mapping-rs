# Several bones

A scan often holds several bones: a knee CT has the femur, tibia, patella
and fibula. Each bone is mapped as its own closed surface with its own fit,
since bones differ in size, cortex and convergence, but they share the
scanner, so they share its blur and cortical density. This is the pipeline
of [the clinical knee example](../clinical_ct.md), which maps all four bones
of a clinical knee CT from the image alone.

## 1. A threshold

Calibrated HR-pQCT has a known bone threshold (320 mg HA/cm³). Clinical CT
in Hounsfield units has none; Otsu's threshold over the voxels above a floor
(to leave out air) separates bone from soft tissue:

```python
threshold = cbm.otsu_thresholds(image, floor=-500.0)[0]
```

## 2. Label the bones

```python
labels, sizes = cbm.label_bones(image, threshold, min_size=min_voxels)
```

Labels are numbered by size (1 = largest). Bones that touch at the threshold
through blur or noise become one label. Given a landmark inside each bone
(physical coordinates), {func}`~cortical_bone_mapping_rs.separate_bones`
names the bones and splits those the threshold joined by raising it locally:

```python
landmarks = {"femur": p_femur, "tibia": p_tibia, "patella": p_patella, "fibula": p_fibula}
labels, names, used = cbm.separate_bones(image, threshold, landmarks, spacing)
```

Thin cortices can fall below the threshold through partial-volume blur.
{func}`~cortical_bone_mapping_rs.grow_bones` grows each bone into connected
voxels above a lower threshold, by default the soft-tissue peak of the
histogram plus four standard deviations:

```python
low = cbm.soft_tissue_threshold(image, threshold, floor=-500.0)
labels = cbm.grow_bones(image, labels, low)
```

## 3. Starting segmentations

```python
starts = cbm.bone_hulls(labels, spacing, closing_radius=4.0)
```

Each bone's start is a "local convex hull": the closing of the bone with a
4 mm ball, holes filled slice by slice. It fills pits and small gaps but
keeps wide concavities such as the femoral intercondylar notch, which a
per-slice convex hull would bridge. Where two bones' starts would overlap,
each voxel goes to the nearer bone.

## 4. Refine every bone

```python
refine = cbm.RefineOptions(refine_iterations=150)
results = cbm.refine_bones(image, starts, spacing, options=options, refine=refine, names=names)
for label, rr in results.items():
    print(names.get(label, label), rr.converged, np.median(rr.result.thickness))
```

{func}`~cortical_bone_mapping_rs.refine_bones` refines one bone at a time,
from the smallest to the largest:

- the scanner's parameters (an estimated `ρc`, the volumetric model's PSF)
  are estimated once on all the bones together and held in every bone's
  fit ({func}`~cortical_bone_mapping_rs.shared_scanner`; `shared=False` to
  estimate them per bone);
- every other bone's region is excluded, so a bone's surface never moves
  into its neighbour (`exclude_others=True`);
- each bone is cropped to its bounding box plus the sampling range
  (`crop=True`), with results in the full image's coordinates;
- `per_bone={label: {...}}` overrides options for one bone.

Masks come back at the full image's shape. On the clinical knee CT, all four
bones converged in 5–6 moves with the regularised fit (166 s in all) and in
4–7 moves with the volumetric model (252 s).

{func}`~cortical_bone_mapping_rs.map_bones` is the same without refinement:
one fit per bone on the given labels.
