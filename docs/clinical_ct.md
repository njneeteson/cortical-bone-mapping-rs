# Real data: a clinical knee CT

This is real data, not a phantom: a clinical CT of a knee, used with the
subject's permission. The image is not part of the repository. It is
non-calibrated (Hounsfield units), so there is no fixed bone threshold, and
it holds four bones: femur, tibia, patella and fibula. Each is segmented and
mapped separately, from the image alone.

| | |
|---|---|
| scanner | GE Revolution GSI, 140 kVp |
| voxels | 0.65 × 0.65 mm in-plane, 0.625 mm slices |
| region | the right knee, 300 slices (187 mm) |

How the pipeline got to this point, and why each step is there, is in
[refinement_experiments.md](refinement_experiments.md).

## Commands

Every step is a library call. The script reads the DICOM series, crops to
one knee, and draws the figures:

```bash
python docs/scripts/ct_example.py /path/to/DICOM --crop :,:,0:256 \
    --landmark femur=250,240,123 --landmark tibia=60,250,110 \
    --landmark patella=211,167,114 --landmark fibula=60,296,43 \
    --figures docs/images/knee_ct
```

which runs

```python
thr = cbm.otsu_thresholds(image, floor=-500)[0]                       # 267 HU
labels, names, _ = cbm.separate_bones(image, thr, landmarks, spacing, origin)
starts = cbm.bone_hulls(labels, spacing, closing_radius=4.0)          # local convex hulls
results = cbm.refine_bones(image, starts, spacing, options=OPTIONS, refine=REFINE,
                           names=names, origin=origin)
```

with, for clinical CT,

```python
OPTIONS = cbm.Options(outside=4, inside=8, resolution=0.25, method="regularized",
                      reg_sigma=3, reg_neighbours=500, global_params=(),  # per-node densities
                      rho_c="estimate", ...)                               # bounds in HU
REFINE = cbm.RefineOptions(refine_iterations=150, level_set=True, freeze=True)
```

1. **Threshold.** On a non-calibrated CT, the histogram of the body (the
   voxels above −500 HU) has a fat peak (about −105 HU) and a soft-tissue
   peak (about 55 HU). Bone forms no peak, only a long tail up to about
   1650 HU. The two-class Otsu threshold of that histogram, 267 HU, separates
   soft tissue from bone. A known threshold in image units can be given
   instead (`--threshold 300`).

   ![Histogram and threshold](images/knee_ct_threshold.png)

2. **Bones.** `label_bones` takes the connected components above the
   threshold and numbers them by volume (1 = the largest), discarding
   anything under 2 cm³. The landmarks are optional; each is a voxel inside a
   bone. With them, `separate_bones` names the bones and checks that no two
   landmarks fall on the same component. If two do, the threshold has
   joined two bones, and it is raised within that component until they
   split. Here the four bones were separate at 267 HU:

   | bone | name | thresholded volume |
   |---|---|---|
   | 1 | femur | 144 cm³ |
   | 2 | tibia | 81 cm³ |
   | 3 | patella | 19 cm³ |
   | 4 | fibula | 12 cm³ |

3. **Starting segmentations.** `bone_hulls(..., closing_radius=4)` closes each
   bone on its own, with a ball of radius 4 mm:
   1. dilate;
   2. fill the voids that do not reach the outside, and the holes of every
      slice in all three orientations;
   3. erode.

   This "local convex hull" fills concavities narrower than about 8 mm:
   pits, small cortical gaps, and the marrow wherever the trabecular density
   falls below the threshold. Wider concavities, such as the femoral
   intercondylar notch (about 20 mm), stay open. A per-slice convex hull
   bridges the notch, and refinement cannot recover from that (see the
   experiments). Where two bones' starts overlap, each voxel goes to the
   nearer bone and a gap is kept between them, so that every bone is its own
   closed surface.
4. **Refinement.** `refine_bones` refines one bone at a time, from the
   smallest to the largest, each with its own initialisation and its own
   cortical density estimate. Per bone:
   - **Per-node densities.** The soft-tissue and trabecular densities are
     regularised fields over the surface, like `m`, `t` and `σ`. In a knee
     the trabecular density ranges from about 70 HU (fatty marrow) to about
     500 HU (epiphyses), and a single global value pulls the surface into
     the bone.
   - **Remeshing.** The surface is carried as a level set and remeshed every
     iteration.
   - **Inexact inner fits.** They stop once the fitted edges have settled to
     a tenth of the last move.
   - **Regional convergence.** Regions that have stopped moving are frozen
     and no longer refitted.
   - **The finish.** A bone has converged when a full fit of its whole
     surface moves it by less than 0.1 voxel RMS, and by less than 0.2 voxel
     RMS in every region.

## Results

![Segmentation](images/knee_ct_segmentation.png)

*Sections through the knee: the starting segmentations (yellow, dashed), and
the refined periosteal (solid) and fitted endosteal (dotted) surfaces of each
bone. Everything more than 3 mm from the bones is blanked.*

![Refinement](images/knee_ct_refinement.gif)

*Refinement, one move per frame (coronal and sagittal sections).*

![Thickness](images/knee_ct_thickness.png)

*Fitted cortical thickness on the refined periosteal surfaces.*

| bone | nodes | moves | converged | `ρc` (HU) | thickness, median (5–95%) | time |
|---|---:|---:|---|---:|---|---:|
| patella | 13 268 | 6 | yes | 1379 | 1.17 mm (0.31–3.46) | 11 s |
| fibula | 16 481 | 5 | yes | 1488 | 0.88 mm (0.27–2.75) | 11 s |
| tibia | 65 410 | 6 | yes | 1493 | 0.89 mm (0.11–3.18) | 70 s |
| femur | 82 342 | 4 | yes | 1532 | 0.53 mm (0.18–2.55) | 49 s |

The whole knee takes 141 s. The first version of this pipeline
(per-slice convex hulls, global densities, the same mesh nodes moved every
iteration) took about 19 minutes, and its femur never converged. It folded
over the notch.

* **All four bones converge** in 4–6 moves, and their surfaces follow the
  periosteum, the femoral notch included.
* **The thickness pattern** is what one expects:
  - thin (≲ 0.5 mm) subchondral and metaphyseal shells over the epiphyses;
  - a thick anterior patella (up to about 3.5 mm);
  - diaphyseal cortices of 2–3.5 mm at the cut ends of the femur, tibia and
    fibula.
* **`ρc`** is estimated per bone, at 1380–1530 HU, from the thick
  diaphyseal cortex where there is one. It is an apparent density at this
  resolution.

There is no ground truth here: the thickness of a cortex thinner than the
in-plane resolution (0.65 mm) is a model-based estimate (Treece et al. 2010),
not a measurement that can be checked against the image by eye. In convex
regions the refined surfaces sit about 0.5 mm (0.7 voxel) inside the 267 HU
isosurface. For a cortex thinner than the blur that is what the model
predicts, but how far depends on `ρc`. That remains an open question (see
[refinement_experiments.md](refinement_experiments.md#open-questions)).
