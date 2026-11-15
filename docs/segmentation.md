# Automatic segmentation

Standard HR-pQCT analysis segments the periosteal and endosteal surfaces with
the dual-threshold method of Buie et al. (2007), or the Burghardt et al.
(2010) variant, and then an operator inspects and corrects the result slice
by slice. This page asks whether cortical bone mapping can take over that
correction, starting from the image alone. It covers:

* [the recommended pipeline](#the-pipeline);
* [the two starting segmentations](#starting-segmentations), including how
  Buie's method is implemented from the paper;
* [an experiment](#experiment) on phantoms built from published HR-pQCT
  values, including a nearby ulna, cortical breaks and a knee joint;
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
   usually closer, but when it fails it fails topologically (its marrow
   leaks through a 1.2 mm cortical break, or it merges a nearby ulna or
   fibula without its pre-processing), and refinement cannot repair
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
bones are joined at the threshold (by blur, noise or motion across a thin
soft-tissue gap): they become one component, and one hull per slice. For a joint, the hull is taken per bone (the two largest components);
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

`python docs/scripts/segmentation_experiment.py` (about 70 minutes; results
in [`segmentation_experiment.json`](segmentation_experiment.json)).

### Phantoms

60.7 µm voxels (XtremeCT II), PSF 0.08 mm, noise σ = 80 on a 0–1000 mg
HA/cm³ scale. The parameters are chosen from published HR-pQCT values
(below). Long-bone shafts (`cbm.BonePhantom`, 40 slices, 360² in-plane,
470–500² with a neighbour) are distal-radius sized: periosteal radius 8.5 mm,
cross-section 228 mm². They have an irregular, axially varying periosteum and
thickness, a trabecular network, cortical pores (~0.17 mm), and an
endocortical transition zone in which porosity rises and trabeculae thicken
towards the (smooth) true endosteum:

| scenario | thickness (min) | tissue density | trabecular BV/TV deep / at endosteum | porosity bulk / extra at endosteum | other |
|---|---|---:|---|---|---|
| healthy | 0.9 mm (0.15) | 1050 | 0.20 / 0.40 | 0.03 / 0.15 | |
| osteoporotic | 0.45 mm (0.2) | 850 | 0.12 / 0.35 | 0.06 / 0.35 | |
| thick | 1.5 mm (0.15) | 1100 | 0.28 / 0.50 | 0.02 / 0.10 | |
| neighbour 2.0 / 1.0 / 0.5 mm | as healthy | 1050 | | | an ulna-like bone (radius 3.5 mm, cortex 0.8 mm) at that gap |
| breaks | as healthy | 1050 | | | four round breaks, 0.3, 0.5, 0.8 and 1.2 mm across |

The true periosteum spans the breaks; lines through a break are excluded
from the thickness metrics (there is no cortex to measure).

The knee (`cbm.JointPhantom`, 180–220 × 200²) is the contact region of a
scaled-down joint: a tibial plateau (shallow dish) below a femoral condyle
(more curved). At the contact point they are 4.5 mm apart ("knee", healthy) or
2.0 mm ("knee OA", moderate narrowing); the gap is filled with cartilage. Each
bone has a 0.15–0.6 mm subchondral plate, a 0.3 mm side shell, and
trabeculae thickening towards the plate (BV/TV 0.22 → 0.40); tissue density
950. CBM uses the knee preset (normals along the axis, 1000 regularisation
neighbours) and measures the plates.

#### What real images look like

The phantoms are meant to stay inside what HR-pQCT shows in vivo:

* **Cortex.**
  * Distal radius Ct.Th is 0.77–0.93 mm on average (XtremeCT, Burghardt et
    al. 2010) and up to ~1.2 mm in young adults on XtremeCT II; the distal
    tibia is 1.0–1.7 mm.
  * At the distal end of the radius region an osteoporotic cortex can be
    0.2–0.3 mm thick, hence the osteoporotic minimum of 0.2 mm.
  * Ct.Po is 1.5–2.5% at the radius and 3.5–6.6% at the tibia, with resolved
    pores of 0.16–0.19 mm mean diameter (Burghardt et al. 2010).
  * Ct.vBMD is ~800–975 mg HA/cm³. That density includes porosity, hence
    tissue densities of 850–1100.
* **Trabecular bone.** BV/TV 0.12–0.33, Tb.Th 0.22–0.29 mm, Tb.Sp
  0.6–1.0 mm (XtremeCT II normative data, e.g.
  [PMC9719438](https://pmc.ncbi.nlm.nih.gov/articles/PMC9719438)).
* **Size.** Distal radius Tt.Ar is 220–360 mm² and distal tibia 590–850 mm².
  The shafts here are a small radius; the default `BonePhantom` (radius
  4.5 mm) is scaled down for tests.
* **Neighbouring bones.** The ulna and fibula are always separated from the
  radius and tibia by soft tissue: the interosseous membrane, the
  syndesmosis (tibiofibular clear space ~2–6 mm on CT), or joint cartilage.
  They never touch in a normal scan. The realistic failure is a threshold
  segmentation that bridges a thin soft-tissue gap (blur, noise, motion) and
  captures both bones; that is the 0.5 mm case.
* **Cortical breaks.** Small.
  * HR-pQCT detects interruptions from ~0.25 mm across. In finger joints the
    median break volume is ~0.2 mm³ in healthy subjects and ~0.5 mm³ in
    rheumatoid arthritis (Peters et al. 2017, 2019).
  * Through a 0.5–1 mm cortex, that is ~0.3–1.2 mm across.
  * At the radius and tibia, the analogues are vascular foramina and the
    loss of a very thin shell. Erosions of several mm occur only in advanced
    erosive disease or at fractures.
* **Knee.** The healthy medial joint space is ~4.8 mm (women) and ~5.7 mm
  (men) radiographically, with ~1.5 mm of tibial cartilage. Narrowing to
  ~2 mm is moderate osteoarthritis; bone-on-bone contact is end-stage. With
  1 mm outside sampling, the opposing bone is out of reach in both knees.
  The 5 mm outside range of Table 1 would reach it in the OA knee.
* **Motion.** Motion is the commonest artefact: ~23% of exams are repeated,
  and a typical visual grade is 2–3 of 5 (Pialat et al. 2012). Streaks and
  cortical discontinuities appear from grade 4. It is not simulated here.

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

| pipeline | healthy | osteoporotic | thick | neighbour 2.0 mm | neighbour 1.0 mm | neighbour 0.5 mm | breaks | knee | knee OA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| buie | 70 (−13) | 147 (**−133**) | 148 (+81) | 79 (−14) | 78 (−3) | 75 (−2) | **372** (+95) | **135** (−15) | **140** (−16) |
| local | 154 (+41) | 144 (+71) | 260 (+123) | 159 (+42) | 155 (+50) | 149 (+43) | 162 (+39) | 363 (+160) | 333 (+148) |
| multigrid | 51 (+17) | 69 (+56) | 97 (+56) | 61 (+20) | 62 (+30) | 159 (+61) | 70 (+8) | 312 (+191) | 295 (+179) |
| regularized | 72 (+27) | 73 (+58) | 154 (+88) | 80 (+31) | 84 (+40) | 165 (+67) | 83 (+21) | 211 (+120) | 203 (+111) |
| reg+curvature | 42 (+18) | 64 (+56) | 87 (+50) | 46 (+20) | 49 (+30) | **96** (+47) | 53 (+10) | 212 (+93) | 212 (+88) |
| curvature | 43 (+20) | 72 (+63) | 88 (+51) | 48 (+23) | 50 (+31) | 103 (+50) | 46 (+20) | 257 (+157) | 259 (+153) |
| refine | 58 (+3) | **57** (+41) | 136 (+67) | 65 (+7) | 66 (+16) | 158 (+45) | 61 (+4) | 194 (+97) | 191 (+93) |
| refine+curvature | **40** (+19) | 67 (+59) | **85** (+54) | **45** (+22) | **48** (+31) | 155 (+62) | **45** (+14) | 219 (+105) | 222 (+106) |

Cortex Dice, and the mean distance of the periosteal / endosteal boundary
from the truth (µm):

| pipeline | healthy | osteoporotic | thick | neighbour 2.0 mm | neighbour 1.0 mm | neighbour 0.5 mm | breaks |
|---|---|---|---|---|---|---|---|
| buie | 0.954 · 21 / 59 | 0.808 · 17 / 116 | 0.961 · 26 / 104 | 0.951 · 21 / 64 | 0.953 · 22 / 61 | 0.953 · 22 / 61 | 0.878 · 1268 / 937 |
| hull (start) | – · 97 / – | – · 80 / – | – · 101 / – | – · 98 / – | – · 98 / – | – · 98 / – | – · 97 / – |
| local | 0.941 · 14 / 89 | 0.878 · 23 / 83 | 0.949 · 12 / 146 | 0.941 · 13 / 91 | 0.939 · 14 / 89 | 0.938 · 17 / 88 | 0.935 · 70 / 188 |
| multigrid | 0.972 · 11 / 40 | 0.929 · 15 / 51 | 0.973 · 10 / 69 | 0.968 · 12 / 47 | 0.968 · 11 / 48 | 0.950 · 39 / 48 | 0.960 · 18 / 45 |
| regularized | 0.963 · 14 / 52 | 0.926 · 17 / 52 | 0.963 · 13 / 94 | 0.961 · 14 / 56 | 0.959 · 14 / 59 | 0.942 · 39 / 60 | 0.958 · 21 / 55 |
| reg+curvature | 0.972 · 12 / 40 | **0.934** · 14 / 48 | 0.969 · 17 / 76 | 0.971 · 13 / 42 | 0.968 · 12 / 47 | 0.959 · 26 / 47 | 0.962 · 16 / 42 |
| curvature | 0.972 · 12 / 41 | 0.928 · 15 / 53 | 0.969 · 17 / 77 | 0.970 · 13 / 43 | 0.968 · 12 / 48 | 0.958 · 26 / 48 | 0.963 · 13 / 41 |
| refine | 0.966 · 14 / 46 | 0.928 · 21 / 46 | 0.965 · 14 / 88 | 0.963 · 14 / 51 | 0.962 · 15 / 53 | 0.960 · 16 / 54 | 0.949 · 41 / 89 |
| refine+curvature | **0.977** · 10 / 34 | 0.930 · 15 / 50 | **0.974** · 10 / 68 | **0.975** · 10 / 37 | **0.973** · 10 / 42 | **0.962** · 16 / 50 | **0.966** · 15 / 36 |

Scenario-specific numbers:

* **Neighbours.** Fraction of the ulna inside the periosteal mask:
  * Buie: 0 at every gap.
  * Without the largest-bone pre-processing (as in Bonelab's autocontour):
    1.0 at every gap, i.e. two bones in one mask. Cortex Dice falls to 0.81,
    and the periosteal error is 0.6–0.8 mm.
  * The hull: 0; every CBM pipeline: ≤ 2%.
* **Breaks.** Buie's result depends on break size. Run one break at a time,
  it is unaffected up to 0.8 mm (Ct.Th 1.12 against a true 1.14 mm). The
  1.2 mm break wrecks it:
  * Buie, 3D connectivity: the marrow mask leaks through the break and
    replaces most of the cortex (Ct.Th 0.35 mm against 1.13 mm, cortex Dice
    0.88).
  * Buie 2D: Ct.Th 0.98 mm, but a periosteal error of 1.4 mm.

  Mean distance of the periosteal boundary near the breaks from the true
  (bridging) periosteum: hull 101 µm, **`curvature` 45 µm**,
  `reg+curvature` 86, `multigrid` 115, `refine+curvature` 117, `regularized`
  224, `refine` 449, Buie 349 µm.
* **Knee.** No pipeline bridges the joint space in either knee. All CBM
  pipelines find the articular surface to within 21–37 µm. The plate's inner
  edge is off by 127–223 µm (Buie: 76 µm).

Cortical density (regularised fit on the hull; thickness bias / RMSE in µm):

| scenario (tissue density) | oracle | fixed 1000 (Table 1) | estimate | profile max (Bonelab's default) |
|---|---|---|---|---|
| healthy (1050) | −10 / 62 | +32 / 75 | 1005: +27 / 72 | 966: +66 / 95 |
| osteoporotic (850) | −20 / 41 | −88 / 94 | 729: +58 / 73 | 574: **+203 / 213** |
| thick (1100) | +50 / 129 | +147 / 202 | 1062: +88 / 154 | 1078: +77 / 146 |
| knee (950) | +42 / 178 | +2 / 177 | 868: +120 / 211 | 802: +203 / 276 |

## Conclusions

* **The hull is a safe start.** In every scenario it contained the bone,
  never leaked and never took in the neighbour. On every shaft, refinement
  with the curvature penalty, started from it, matched or beat Buie on the
  periosteum (10–16 µm against Buie's 17–26 µm).
* **Where Buie is biased, CBM corrects it.** On the thin porous cortex, Buie
  calls the porous inner cortex marrow (cortex Dice 0.81, thickness
  −133 µm). CBM from the hull reaches Dice 0.93 and a thickness RMSE of
  57–73 µm. This is the error that is usually corrected by hand.
* **Where Buie is right, CBM is better but not by much.** On healthy, thick
  and neighbour shafts with a gap of 1 mm or more, `refine+curvature` gives
  RMSEs of 40–85 µm against Buie's 70–148 µm, and cortex Dice of
  0.973–0.977 against 0.95–0.96.
* **The curvature penalty is the most useful extension.** With or without
  refinement, it is the best or near-best pipeline on every shaft. It
  bridges cortical breaks: alone (`λ = 0`) it keeps the periosteum within
  45 µm of the true surface at the breaks. Plain refinement instead follows
  the soft tissue into them (449 µm); with the curvature penalty it does not
  (117 µm). The local fit is never competitive (RMSE ~150 µm).
* **A realistic neighbour matters only when it is within the outside
  sampling range.** At 2.0 and 1.0 mm gaps the ulna has no effect (RMSE
  45–48 µm). At 0.5 mm the outside half of the profiles enters its cortex,
  and the RMSE roughly doubles (96–165 µm, against Buie's 75 µm), although
  no pipeline includes the ulna in the mask. Truncating profiles where they
  enter another bone, or a shorter outside range, should remove this.
  Without its largest-bone pre-processing, Buie merges the two bones at
  every gap tested, which is exactly the failure seen in practice.
* **Buie fails on breaks the size of an erosion.** It closes breaks up to
  0.8 mm. A single 1.2 mm break lets the marrow mask leak through and
  replace most of the cortex. The hull and the curvature-penalised CBM
  pipelines are unaffected.
* **The knee plate is where CBM is weakest.** Buie's thickness RMSE on the
  subchondral plate is 135–140 µm; the best CBM pipelines reach 191–212 µm
  with a +90–120 µm bias. CBM finds the articular surface as well as Buie,
  but puts the plate's inner edge too deep, among the dense subchondral
  trabeculae. With a realistic joint space (2–4.5 mm) the opposing bone is
  out of reach of 1 mm outside sampling, so this is a plate problem, not a
  joint problem. The knee also costs the most: the curvature fits exhaust
  their work budget, and refinement takes 4–10 minutes.
* **Do not assume the cortical density, but know what the estimate is.**
  * Table 1's fixed 1000 is off by up to 15% on these phantoms and biases
    thickness by −88 to +147 µm.
  * Bonelab's per-node profile maximum is worst on thin cortices (+203 µm),
    because a thin, blurred, smoothed cortex never reaches its density.
  * The estimate is robust but measures the *apparent* density of the
    porous cortex (3–14% below the tissue density). That leaves a positive
    bias of 30–120 µm.
* **Run time.** On a radius-sized, 40-slice shaft (~45 000 nodes):
  * one regularised fit takes 4–10 s, and 15–45 s with the curvature
    penalty;
  * refinement takes 20–60 s, and 1–5 minutes with the curvature penalty.

**Caveats.** Phantoms are not bone:
* the "true" endosteum of real trabecularised cortex is itself a definition;
* real neighbours, breaks and joints differ from these geometric models;
* the knee phantom is scaled down;
* motion artefacts are not simulated.

These results say where to look on real HR-pQCT data, which is the next
step.

### Sources for the phantom parameters

* A. J. Burghardt et al., *Reproducibility of direct quantitative measures
  of cortical bone microarchitecture of the distal radius and tibia by
  HR-pQCT*, Bone 47:519–528, 2010
  ([PMC2926164](https://pmc.ncbi.nlm.nih.gov/articles/PMC2926164)). Ct.Th,
  Ct.Po, pore diameter; trans-cortical foramina treated as background.
* XtremeCT II normative data, 863 adults
  ([PMC9719438](https://pmc.ncbi.nlm.nih.gov/articles/PMC9719438)). Ct.Th,
  Ct.Po, Ct.vBMD, trabecular structure, Tt.Ar.
* M. Peters et al., *An automated algorithm for the detection of cortical
  interruptions on HR-pQCT images of finger joints*, PLoS ONE 12:e0175829,
  2017 ([PMC5402632](https://pmc.ncbi.nlm.nih.gov/articles/PMC5402632)), and
  *Prospective follow-up of cortical interruptions …*, Calcif. Tissue Int.,
  2019 ([doi:10.1007/s00223-019-00523-2](https://doi.org/10.1007/s00223-019-00523-2)).
  Break detection limit and volumes.
* J. B. Pialat et al., *Visual grading of motion induced image degradation
  in high resolution peripheral computed tomography: impact of image quality
  on measures of bone density and micro-architecture*, Bone 50:111–118,
  2012. Motion artefacts.
* Knee joint space width and cartilage thickness in healthy adults: BMC
  Musculoskelet. Disord. 9:119, 2008
  ([doi:10.1186/1471-2474-9-119](https://doi.org/10.1186/1471-2474-9-119)).
