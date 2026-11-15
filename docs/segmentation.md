# Automatic segmentation

Standard HR-pQCT analysis segments the periosteal and endosteal surfaces with
the dual-threshold method of Buie et al. (2007), or the Burghardt et al.
(2010) variant, and then an operator inspects and corrects the result slice
by slice. This page asks whether cortical bone mapping can take over that
correction, starting from the image alone. It covers:

* [the recommended pipeline](#the-pipeline);
* [the two starting segmentations](#starting-segmentations), including how
  Buie's method is implemented from the paper;
* [an experiment](#experiment) on realistic phantoms, including a nearby
  fibula/ulna, cortical breaks and a knee joint;
* [what it shows](#conclusions), and what remains to be tested on real data.

## The pipeline

```python
import cortical_bone_mapping_rs as cbm

start = cbm.hull_segmentation(image)                 # threshold 320, per-slice convex hull
options = cbm.Options.preset("radius_tibia", rho_c="estimate")
rr = cbm.refine_segmentation(image, start, options=options,
                             curvature="change", curvature_weight=1.0)
rr.periosteal_mask, rr.endosteal_mask, rr.cortical_mask   # the segmentation
rr.result.thickness                                       # and the thickness map
```

1. **Start from a hull, not from Buie.** The starting segmentation only has
   to be topologically right (one closed surface around the bone) and close
   enough for the profiles to cross the cortex. A convex hull of the
   thresholded bone has both properties by construction. Buie's method is
   usually closer, but when it fails it fails topologically (it leaks into
   the marrow through a wide cortical break), and refinement cannot repair
   topology.
2. **Let CBM place both surfaces.** One fit on the hull's surface gives the
   periosteal and endosteal surfaces; refinement then moves the surface to
   the fitted periosteum and refits until it stops moving.
3. **Do not assume the cortical density.** `rho_c="estimate"` measures it
   from the image (see [method](method.md#cortical-density)).

## Starting segmentations

### Threshold hull

`cbm.hull_segmentation(image, threshold=320)` (Rust:
`segmentation::hull_segmentation`):

1. threshold (`image ≥ 320`, the usual bone threshold for density-calibrated
   HR-pQCT);
2. binary median, kernel (3,3,1) as in Buie's step 3, against noise speckle;
3. keep the largest 3D component (drops a separate fibula/ulna and stray
   voxels);
4. the convex hull of every slice (along axis 0).

The hull always contains the thresholded bone. Its errors are outward bulges
over concave parts of the periosteum, which refinement pulls in. It fails
when a concavity is deeper than the inward sampling range, and when two
bones touch at the threshold: they become one component, and one hull per
slice. For a joint, the hull is taken per bone (the two largest components);
see the knee scenarios below.

### Buie et al. (2007)

`cbm.buie_segmentation(image)` (Rust: `segmentation::buie`) follows Fig. 1
and the Methods of

> H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
> *Automatic segmentation of cortical and trabecular compartments based on a
> dual threshold technique for in vivo micro-CT bone analysis*, Bone
> 41:505–515, 2007, doi:10.1016/j.bone.2007.07.007

| Fig. 1 step | paper | here |
|---|---|---|
| 2 threshold | user input 1 | `image ≥ periosteal_threshold` |
| 3 median | kernel (3,3,1) | binary majority, `[1, 3, 3]` (array order z, y, x) |
| 4 dilate | kernel (15,15,1) | ellipsoidal footprint (as `vtkImageDilateErode3D`), exact distance transform |
| 5 connectivity | select non-bone black region | background components touching the in-plane image border; bone = the rest |
| 6 erode | (15,15,1) | → periosteal mask |
| 7 threshold | user input 2 | `image < endosteal_threshold` |
| 8 mask | with step 1 region → marrow cavities | marrow = below T2 inside the periosteal mask |
| 9 dilate | (10,10,1) | dilate the marrow |
| 10 connectivity | select outer white region | non-marrow components touching the in-plane border |
| 11 erode | (10,10,1) | erode the marrow |
| 12 Gaussian smooth | σ = 3, support (3,3,1) | σ = 3 voxels, kernel radius ⌊σ·support⌋ = 9 in-plane, 3 along z (VTK semantics) |
| 13 threshold | by lower = 100 | marrow where the smoothed 0/127 mask ≤ 100 |
| 14 mask | combine | cortex = periosteal ∧ ¬marrow, trabecular = periosteal ∧ marrow |

Also from the paper:

* **Pre-processing.** Keep only the largest bone (removes the fibula/ulna;
  `largest_bone_only`). This is done on the thresholded image, before the
  step-4 dilation; after it, a neighbour closer than twice the dilation
  radius (~1.8 mm) would already be merged with the bone.
* **Padding.** The data are padded in-plane by the dilation radius, so that
  dilation cannot clip at the image edge (the paper's Discussion).
* **Connectivity.** Optionally per slice (`connectivity_2d`), which the paper
  reports is usually identical and faster.

Where the paper is silent:

* **Mask values.** 0/127 (Scanco) is assumed for step 13. This matters:
  smoothing with σ = 3 and re-thresholding at 100/127 moves the endosteal
  boundary into the cortex by about Φ⁻¹(100/127)·σ ≈ 0.8σ ≈ 2.4 voxels on a
  straight edge (2.1 voxels measured on a test ring). With 0/255 masks it
  would move ~0.8 voxels the other way.
* **Footprint shape.** Ellipsoidal, as in VTK (the paper's implementation).
* **Thresholds.** Inputs in image units. The paper used native scanner units
  (human XtremeCT: 3000 / 5000); the default 450 / 320 mg HA/cm³ is the
  common choice for density-calibrated HR-pQCT (Burghardt et al. 2010).

Bonelab's `autocontour.py` differs from the paper in its Gaussian (σ = 0.8
instead of 3, which reduces the step-13 shift to ~0.6 voxels and changes
Ct.Th by roughly 1.8 voxels), in using the *largest* instead of the *outer*
component in steps 5 and 10, and in having no largest-bone pre-processing,
padding or 2D option.

## Experiment

`python docs/scripts/segmentation_experiment.py` (about 50 minutes; results
in [`segmentation_experiment.json`](segmentation_experiment.json)).

### Phantoms

60.7 µm voxels, PSF 0.08 mm, noise σ = 80 on a 0–1000 mg HA/cm³ scale.
Long-bone shafts (`cbm.BonePhantom`, 40 slices, 200² or 240² in-plane) have
an irregular, axially varying periosteum and thickness, a trabecular network,
cortical pores, and an endocortical transition zone in which porosity rises
and trabeculae thicken towards the (smooth) true endosteum:

| scenario | thickness | tissue density | trabecular BV/TV deep / at endosteum | porosity bulk / extra at endosteum | other |
|---|---|---:|---|---|---|
| healthy | 0.9 mm | 1050 | 0.20 / 0.40 | 0.03 / 0.15 | |
| osteoporotic | 0.45 mm | 850 | 0.12 / 0.35 | 0.06 / 0.35 | |
| thick | 1.5 mm | 1100 | 0.28 / 0.50 | 0.02 / 0.10 | |
| neighbour 1.0 / 0.3 / 0 mm | as healthy | 1050 | | | a fibula/ulna (radius 1.5 mm) at that gap |
| breaks | as healthy | 1050 | | | four breaks, 0.3, 0.8, 1.6 and 2.4 mm wide, 1.2 mm long |

The true periosteum spans the breaks; lines through a break are excluded
from the thickness metrics (there is no cortex to measure).

The knee (`cbm.JointPhantom`, 140 × 200²) is the contact region of a
scaled-down joint: a tibial plateau (shallow dish) below a femoral condyle
(more curved) across a joint space of 1.0 mm ("knee") or 0.3 mm ("knee
narrow") at the contact point, each bone with a 0.15–0.6 mm subchondral plate,
a 0.3 mm side shell, and trabeculae thickening towards the plate (BV/TV 0.22 →
0.40); tissue density 950. CBM uses the knee preset (normals along the axis,
1000 regularisation neighbours) and measures the plates.

### Pipelines

* **buie**: the reference; `buie 2D` (per-slice connectivity) for the breaks,
  and `buie (no prepro.)` (without the largest-bone step, as in Bonelab's
  autocontour) for the neighbours.
* From the **hull**: one CBM fit with each method (`local`, `multigrid`,
  `regularized`, `reg+curvature`, and `curvature` alone, `λ = 0`), and
  refinement with `regularized` (`refine`) or `reg+curvature`
  (`refine+curvature`).

CBM uses the parameters of Table 1 of Neeteson & Boyd (2025), except that the
sampling is scaled to the phantoms (1 mm outside, 3 mm inside instead of 5 /
15 mm), and estimates `ρc` from the image.

### Metrics

* **Along the sampling lines**, against the exact crossings of the true
  surfaces: mean absolute error of the periosteal and endosteal edge, and
  bias and RMSE of the thickness. For CBM these are the fitted edges; for
  Buie the crossings of its masks along the regularised fit's lines.
* **Masks** (shafts only): Dice of the cortex, mean symmetric 3D distance of
  the periosteal and endosteal boundaries, and semi-derived Ct.Th (cortical
  area / periosteal perimeter per slice). For the knee only the line metrics
  are meaningful: with normals along the axis, the side shells are not
  fitted.

### Results

![Buie and hull → CBM refinement](images/segmentation.png)

Thickness RMSE along the lines (µm; bias in brackets):

| pipeline | healthy | osteoporotic | thick | neighbour 1.0 mm | neighbour 0.3 mm | neighbour 0 mm | breaks | knee | knee narrow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| buie | 78 (+5) | 144 (**−128**) | 126 (+67) | 84 (0) | 84 (0) | 87 (−5) | **397** (+121) | 204 (−3) | 294 (−17) |
| local | 162 (+60) | 157 (+75) | 202 (+90) | 157 (+54) | 202 (+60) | 148 (+45) | 157 (+51) | 448 (+181) | 469 (+178) |
| multigrid | 76 (+37) | 82 (+60) | **86** (+33) | 71 (+35) | 193 (+78) | 105 (+38) | 122 (+14) | 306 (+162) | 569 (+465) |
| regularized | 92 (+48) | 83 (+61) | 120 (+60) | 92 (+45) | 202 (+84) | 73 (+28) | 100 (+29) | 207 (+98) | 270 (+4) |
| reg+curvature | 62 (+36) | 69 (+57) | 101 (+33) | 63 (+35) | 123 (+59) | 95 (+1) | 107 (+6) | 214 (+73) | 294 (−49) |
| curvature | 63 (+38) | 78 (+65) | 102 (+36) | 64 (+37) | 126 (+61) | 270 (−97) | 78 (+42) | 251 (+143) | 279 (+78) |
| refine | 74 (+21) | **64** (+42) | 115 (+47) | 78 (+24) | 206 (+67) | **69** (+6) | 76 (+20) | **193** (+93) | 273 (−35) |
| refine+curvature | **61** (+36) | 71 (+60) | 100 (+34) | **63** (+37) | 200 (+81) | 77 (+12) | **65** (+26) | 221 (+103) | 280 (−2) |

Cortex Dice, and the mean distance of the periosteal / endosteal boundary
from the truth (µm):

| pipeline | healthy | osteoporotic | thick | neighbour 1.0 mm | neighbour 0.3 mm | neighbour 0 mm | breaks |
|---|---|---|---|---|---|---|---|
| buie | 0.956 · 21 / 59 | 0.816 · 16 / 116 | 0.970 · 26 / 69 | 0.954 · 21 / 62 | 0.954 · 21 / 62 | 0.877 · 244 / 329 | 0.912 · 602 / 380 |
| hull (start) | – · 77 / – | – · 58 / – | – · 80 / – | – · 75 / – | – · 75 / – | – · 527 / – | – · 75 / – |
| multigrid | 0.964 · 13 / 54 | 0.921 · 17 / 58 | **0.976** · 12 / 63 | 0.965 · 12 / 53 | 0.938 · 51 / 55 | 0.849 · 258 / 328 | 0.906 · 65 / 66 |
| regularized | 0.960 · 13 / 59 | 0.924 · 17 / 55 | 0.972 · 11 / 76 | 0.960 · 14 / 59 | 0.937 · 45 / 60 | 0.874 · 233 / 294 | **0.935** · 74 / 67 |
| reg+curvature | 0.966 · 13 / 52 | **0.932** · 14 / 51 | 0.965 · 16 / 97 | 0.963 · 13 / 58 | 0.947 · 33 / 60 | 0.846 · 251 / 304 | 0.911 · 46 / 68 |
| curvature | 0.965 · 13 / 53 | 0.925 · 15 / 56 | 0.965 · 16 / 98 | 0.962 · 13 / 58 | 0.947 · 33 / 61 | 0.795 · 302 / 436 | 0.909 · 27 / 61 |
| refine | 0.964 · 13 / 52 | 0.928 · 20 / 48 | 0.973 · 12 / 71 | 0.963 · 14 / 54 | **0.959** · 18 / 57 | 0.876 · 228 / 312 | 0.877 · 231 / 237 |
| refine+curvature | **0.970** · 10 / 47 | 0.930 · 16 / 52 | 0.968 · 12 / 91 | **0.967** · 10 / 52 | 0.940 · 32 / 70 | 0.864 · 244 / 358 | 0.894 · 108 / 147 |

Scenario-specific numbers:

* **Neighbours**, fraction of the fibula/ulna inside the periosteal mask:
  Buie 0 at 1.0 and 0.3 mm, but 1.0 without the pre-processing (as in
  Bonelab's autocontour, which then also has a 0.27–0.33 mm periosteal
  error); the hull 0. At 0 mm every pipeline includes the neighbour (≥ 98%).
* **Breaks**, mean distance of the periosteal boundary near the breaks from
  the true (bridging) periosteum: Buie 509 µm, Buie 2D 552 µm, the hull
  86 µm; `regularized` 400, `reg+curvature` 164, **`curvature` 81**,
  `refine` 601, `refine+curvature` 508 µm. Ct.Th: truth 0.97 mm, Buie 0.48,
  Buie 2D 0.83, `curvature` 1.11, `refine` 0.91 mm.
* **Knee**, Buie does not bridge the joint space in either knee; the per-bone
  hull does at 0.3 mm, where the bones touch at the threshold.

Cortical density (regularised fit on the hull; thickness bias / RMSE in µm):

| scenario (tissue density) | oracle | fixed 1000 (Table 1) | estimate | profile max (Bonelab's default) |
|---|---|---|---|---|
| healthy (1050) | +10 / 73 | +53 / 95 | 1006: +48 / 92 | 976: +79 / 112 |
| osteoporotic (850) | −13 / 46 | −83 / 90 | 736: +61 / 83 | 577: **+208 / 219** |
| thick (1100) | +30 / 104 | +121 / 167 | 1069: +60 / 120 | 1083: +51 / 115 |
| knee (950) | +15 / 188 | −27 / 195 | 867: +98 / 207 | 812: +183 / 277 |

## Conclusions

* **The hull is a safe start.** It contained the bone in every scenario,
  never leaked, and on the shafts without a close neighbour every
  regularised CBM pipeline started from it matched or beat Buie on the
  periosteum (10–20 µm against Buie's 16–26 µm).
* **Where Buie is biased, CBM corrects it.** On the thin porous cortex, Buie
  calls the porous inner cortex marrow (cortex Dice 0.82, thickness
  −128 µm); CBM from the hull reaches Dice 0.93 and a thickness RMSE of
  64–83 µm. This is the error that is usually corrected by hand.
* **Where Buie is right, CBM is about as good.** On healthy and thick
  cortices the gain is small (RMSE 61–86 against 78–126 µm). The thick
  cortex's dense transition zone is best handled by multigrid; the curvature
  penalty slightly hurts its endosteum.
* **Regularisation matters more than refinement.** The local fit is noisy
  (RMSE 150–200 µm); multigrid, regularised and curvature-penalised fits
  halve that. Refinement adds little where the hull is already close (one
  fit on the hull's surface is nearly as good) and helps most on the porous
  cortex.
* **The curvature penalty bridges cortical breaks.** Alone (`λ = 0`) it keeps
  the fitted periosteum on the true bridging surface (81 µm near the breaks),
  where the spatial smoothing term lets it sink into the gaps (400 µm).
  Refinement, by contrast, follows the soft tissue into the breaks, back
  towards Buie's answer; refining a broken cortex needs a way to recognise
  profiles without a cortex (they are currently only excluded when their
  fits hit the bounds).
* **Neighbouring bones within the outside sampling range corrupt the fit.**
  At a 0.3 mm gap the 1 mm outside part of each profile reaches into the
  neighbour's cortex, and the thickness RMSE triples (123–206 µm against
  Buie's 84 µm). Remedies to try: a shorter outside range, or truncating
  profiles where they enter another bone. Touching bones (0 mm) defeat every
  method; separating them needs a dedicated step.
* **The knee is hard for every method.** Buie and the regularised CBM
  pipelines all have thickness RMSEs of 190–300 µm on the subchondral plate
  (Buie 204 µm, local and multigrid worse). With a 1 mm joint space CBM finds
  the articular surface well (23–50 µm) but places the plate's endosteal edge
  among the dense subchondral trabeculae (130–240 µm). The narrow joint adds
  the opposing plate to the outside of the profiles and makes the hull bridge
  the bones. The knee needs its own treatment: sampling limited to the joint
  space, and probably a model with a joint-space (cartilage) level.
* **Do not assume the cortical density, but know what the estimate is.**
  Table 1's fixed 1000 is off by 5–15% on these phantoms and biases thickness
  by up to 120 µm. Bonelab's per-node profile maximum is worst on thin
  cortices (+208 µm), because a thin, blurred, smoothed cortex never reaches
  its density. The estimate is robust but measures the *apparent* density of
  the porous cortex (3–13% below the tissue density), which leaves a positive
  bias of 30–60 µm on these phantoms.
* **Run time.** One regularised fit takes 2–5 s on a 40-slice shaft (15–17 s
  for the knee with 1000 neighbours); refinement 10–70 s, more with the
  curvature penalty.

**Caveats.** Phantoms are not bone. The "true" endosteum of real
trabecularised cortex is itself a definition; real fibulae, breaks (erosions,
vessel canals) and joints differ from these geometric models; and the knee
phantom is scaled down. These results say where to look on real HR-pQCT
data, which is the next step.
