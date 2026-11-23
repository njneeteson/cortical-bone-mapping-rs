# Validation of the volumetric model

This page compares the volumetric model ([method.md](method.md#volumetric-model),
[volumetric_model.md](volumetric_model.md)) with the 1D methods on phantoms of
the sites CBM is used at, at their scanners' resolution, and on our clinical
knee CT. Run time and memory are not reported here; they are for a separate
benchmark.

* [Site phantoms](#site-phantoms)
* [Microarchitecture of the HR-pQCT phantoms, and Ct.Th](#microarchitecture-and-ctth)
* [Accuracy and precision](#accuracy-and-precision)
* [What limits the clinical sites](#what-limits-the-clinical-sites)
* [The curvature weight](#the-curvature-weight)
* [The clinical knee](#the-clinical-knee)
* [Conclusions](#conclusions)

## Site phantoms

`cbm.site_phantom(site, seed)` (Rust: `phantoms::sites`) gives each site at
its scanner's resolution, with parameters from the literature (the module
documentation lists the sources; values without one are typical, not
measured):

| site | phantom | voxels (mm) | PSF σ (mm) | noise | cortex |
|---|---|---|---|---|---|
| distal radius | shaft + ulna, 32 slices | 0.0607 | 0.08 | 40 mg HA/cm³ | 0.85 mm, Tt.Ar ≈ 254 mm² |
| distal tibia | shaft + fibula, 32 slices | 0.0607 | 0.08 | 40 mg HA/cm³ | 1.3 mm, Tt.Ar ≈ 707 mm² |
| proximal femur | head, neck, trochanter, shaft + acetabulum | 1.0 × 0.75 × 0.75 | 0.6 axial, 0.5 in-plane | 30 HU | neck 0.86 sup. – 2.79 inf., head and trochanter 0.8, shaft 3.5 mm |
| knee | tibia + femur with notch, 5–6.5 mm joint space | 0.625 × 0.65 × 0.65 | 0.45 axial, 0.4 in-plane | 22 HU | plates 0.78 (tibia), 0.32 (femur), side shell 0.8 mm |

![Mid-sections of the site phantoms](images/site_phantoms.png)

All have a porous cortex, trabeculae, and an endocortical transition zone in
which porosity and trabecular density rise towards the endosteum. The true
endosteum is the smooth surface behind it, so the image is ambiguous there,
as in real bone.

## Microarchitecture and Ct.Th

`docs/scripts/site_microarchitecture.py` measures the HR-pQCT phantoms with
the second-generation HR-pQCT definitions (Whittier et al. 2020). Tb.Th and
Tb.Sp are the mean local thickness of the trabecular bone and of the marrow
space within the trabecular compartment, Ct.Th is the mean local thickness
of the cortical compartment (Hildebrand & Rüegsegger 1997, computed with
[local-thickness-rs](https://github.com/njneeteson/local-thickness-rs)),
and BV/TV and Ct.Po are volume fractions. "Truth" is the phantom's exact
structure. "Scan" is the rendered image segmented as the standard analysis
does: a Gaussian filter (σ 0.8 voxels, support 1), 320 mg HA/cm³ for
trabecular bone and 450 mg HA/cm³ for the cortex, within the true
compartments. Mean of three phantoms each:

| | radius truth | radius scan | tibia truth | tibia scan | in vivo (HR-pQCT) |
|---|---|---|---|---|---|
| BV/TV | 0.21 | 0.27 | 0.26 | 0.38 | 0.12–0.33 |
| Tb.Th (mm) | 0.17 | 0.30 | 0.19 | 0.33 | 0.22–0.29 |
| Tb.Sp (mm) | 0.56 | 0.74 | 0.55 | 0.61 | 0.6–1.0 |
| Ct.Th (mm) | 0.867 | | 1.294 | | 0.77–0.93 / 1.0–1.7 |
| Ct.Po | 5.4% | 0.3% | 8.6% | 0.4% | 1.5–2.5% / 3.5–6.6% |

The cortex and the trabecular spacing are in the in vivo range. The scanned
trabeculae are a little thicker than in vivo, and the pores are too small to
be resolved, so the scanned porosity is far below in vivo values. Both are
limits of the phantoms' fine structure, not of the cortex that CBM measures.

**Ct.Th by local thickness and by CBM.** Local thickness and CBM measure
different things: local thickness is the diameter of the largest sphere
in the cortex, averaged over its *volume*; CBM's thickness is the distance
between the fitted surfaces along the normal, averaged over the *surface*.
For a cortex whose thickness varies, the volume average weights thick
regions more (by about `t̄ + var(t)/t̄`). Mean thickness, three phantoms
each:

| | radius | tibia |
|---|---|---|
| local thickness of the true cortex (Whittier Ct.Th) | 0.867 | 1.294 |
| true thickness along the normals (CBM's definition) | 0.817–0.827 | 1.216–1.264 |
| CBM, local | 0.874 (RMSE 0.147) | 1.307 (0.176) |
| CBM, regularised | 0.876 (0.131) | 1.311 (0.158) |
| CBM, volumetric | 0.826 (0.048) | 1.213 (0.056) |

The local thickness of the true cortex is 5–6% above the true mean
thickness along the normals, as the volume weighting predicts for this
phantom's ±35% thickness variation. The 1D fits' mean is close to local
thickness only because they overestimate thickness by about 0.05 mm here
(the endocortical transition reads as cortex); the volumetric model's mean
agrees with the truth on its own normals, and its per-node error is a third
of theirs. (These fits used 64 neighbours for the 1D regularisation; the
next section uses the paper's settings.)

## Accuracy and precision

`docs/scripts/volumetric_validation.py`: each site's geometry is rendered
once without noise, and five scans are that image plus independent noise
(exactly how the phantoms add noise), so all fits share the surface and the
nodes. Every method fits from the true periosteal mask with the same
sampling range (HR-pQCT 1 mm outside and 2.5 mm inside; femur 2.5 and 7 mm;
knee 3 and 6 mm), `rho_c="estimate"`, and its defaults otherwise. The
regularised method uses the smoothing of our 2025 paper (σ 1.94 mm; 500
neighbours at HR-pQCT, 128 at clinical resolution, where the nodes are
about ten times sparser), and the volumetric model the same operator for its
density fields and its default curvature weight (0.3). For each node the
thickness and the periosteal position are compared with the truth along the
method's own normal. Bias and RMSE are over nodes and scans, SD is the RMS
over nodes of the standard deviation over the five scans (precision), in mm.
At the clinical sites every method is also run with `ρc` fixed at the
phantom's true cortical density, to separate the error of the estimated
`ρc` from the method's (next section).

**HR-pQCT.**

| | t bias | t RMSE | t SD | edge RMSE | edge SD |
|---|---|---|---|---|---|
| radius: local | +0.048 | 0.154 | 0.0206 | 0.010 | 0.0039 |
| radius: local, estimated PSF | +0.042 | 0.145 | 0.0203 | 0.010 | 0.0038 |
| radius: regularised | +0.030 | 0.082 | 0.0044 | 0.010 | 0.0031 |
| radius: **volumetric** | −0.001 | **0.032** | **0.0009** | **0.006** | 0.0007 |
| tibia: local | +0.046 | 0.183 | 0.0272 | 0.015 | 0.0041 |
| tibia: local, estimated PSF | +0.041 | 0.177 | 0.0272 | 0.014 | 0.0040 |
| tibia: regularised | +0.031 | 0.114 | 0.0076 | 0.013 | 0.0036 |
| tibia: **volumetric** | −0.014 | **0.040** | **0.0006** | **0.006** | 0.0004 |

At HR-pQCT the volumetric model is unbiased and its thickness error is a
third of the regularised fit's (and a fifth of the local fit's), with
5–13× better precision; the periosteal position is about twice as accurate.
The 1D error is almost all structural: it is the same in every scan (SD
0.02 against RMSE 0.15 for the local fit). It sits at the endosteum, where
the porous, trabecularised transition makes single profiles ambiguous; the
volumetric model fits all voxels around a node, with the endosteal surface
regularised by its curvature, and reads it far better. The fitted PSF,
0.089 mm, is close to the effective blur (the PSF and the voxel
integration, `√(0.08² + 0.0607²/12)` = 0.082 mm).

**Clinical CT**, with `ρc` estimated / fixed at the truth:

| | t bias | t RMSE | t SD | edge RMSE | edge SD |
|---|---|---|---|---|---|
| femur: local | +0.166 / +0.155 | 0.219 / 0.211 | 0.035 | 0.060 / 0.062 | 0.024 |
| femur: local, estimated PSF | +0.138 / +0.132 | 0.173 / 0.169 | 0.028 | 0.066 / 0.067 | 0.023 |
| femur: regularised | +0.157 / +0.147 | 0.174 / 0.165 | 0.010 | 0.048 / 0.050 | 0.010 |
| femur: volumetric | +0.129 / +0.117 | 0.155 / 0.146 | 0.015 | 0.048 / 0.049 | 0.011 |
| knee tibia: local | +0.532 / +0.157 | 0.585 / 0.244 | 0.029 / 0.021 | 0.142 / 0.047 | 0.013 |
| knee tibia: local, estimated PSF | +0.480 / +0.110 | 0.523 / 0.167 | 0.024 / 0.018 | 0.134 / 0.053 | 0.012 |
| knee tibia: regularised | +0.486 / +0.118 | 0.505 / 0.177 | 0.007 / 0.004 | 0.141 / 0.055 | 0.007 / 0.004 |
| knee tibia: volumetric | +0.658 / +0.103 | 0.679 / 0.162 | 0.009 / 0.007 | 0.208 / 0.045 | 0.005 |
| knee femur: local | +0.591 / +0.199 | 0.676 / 0.308 | 0.053 / 0.037 | 0.159 / 0.128 | 0.024 / 0.027 |
| knee femur: local, estimated PSF | +0.483 / +0.109 | 0.549 / 0.162 | 0.028 / 0.016 | 0.144 / 0.086 | 0.016 / 0.017 |
| knee femur: regularised | +0.510 / +0.138 | 0.524 / 0.159 | 0.006 / 0.003 | 0.141 / 0.070 | 0.007 / 0.003 |
| knee femur: volumetric | +0.422 / +0.125 | 0.444 / 0.154 | 0.009 / 0.007 | 0.121 / 0.066 | 0.005 / 0.006 |

With the right `ρc`, the volumetric model is the most accurate method at
every clinical site, by a small margin (thickness RMSE 0.146–0.162 mm
against 0.159–0.177 mm for the regularised fit); its precision lies between
the local and the regularised fits'. Its fitted PSF is close to the
effective blur: 0.566 / 0.686 mm (in-plane / axial) on the femur against
0.545 / 0.666 mm, 0.461 / 0.536–0.566 mm on the knee against 0.442 /
0.485 mm. The remaining thickness bias of about +0.1 mm is shared by all
methods, and so is the dependence on `ρc`.

## What limits the clinical sites

Two effects dominate at clinical resolution, and neither is specific to a
method.

**The estimated `ρc`.** `rho_c="estimate"` takes the density plateau inside
cortices that are thick relative to the blur. The knee phantom has none:
its plates (0.32 and 0.78 mm) and side shell (0.8 mm) are all within
`4σ`. The only nodes that qualify are those fitted too thick, whose plateau
is diluted, so the estimate is low (1140–1190 HU against 1551 HU), and with
`ρc` too low every method makes the cortex thicker (the blurred mass
`(ρc − ρb) t` is what the data fix): +0.42 to +0.66 mm. With the true `ρc`
the bias falls to about +0.1 mm. The volumetric model is the most sensitive
to this: with its curvature penalty, the few thick nodes that set the
plateau also drag the fitted PSF down (0.30 / 0.36 mm against 0.46 /
0.54 mm with the true `ρc`). On the femur, whose shaft cortex is thick, the
estimate is close (about 1540 HU) and fixing `ρc` changes little. This is
the identifiability problem that led Treece et al. (2010) to constrain the
cortical density: **in a field of view without cortex much thicker than
about 4σ, fix `ρc`** (from a calibration, or from a scan that includes a
shaft). On the clinical knee, whose field of view includes the shafts, the
estimate was 1380–1530 HU ([clinical_ct.md](clinical_ct.md)).

**The endocortical transition.** With the true `ρc`, a thickness bias of
about +0.1 mm remains for every method. It is the dense trabecular bone
the phantoms put against the endosteum (BV/TV rising to 0.35 within 2–3 mm):
at clinical resolution it is indistinguishable from cortex. Without the
transition (the femur with `endo_bvtv = bvtv`, noise-free, true `ρc`), both
the regularised and the volumetric fits are unbiased (+0.000 and
−0.009 mm), with RMSE 0.048 and 0.059 mm; removing the pores changes
nothing. Where the volumetric error remains there, it is in the thick shaft
cortex (RMSE 0.08 mm against 0.06 mm for the regularised fit), where the
curvature penalty resists the real variation of the endosteal surface
(±0.5 mm over 25 mm), and in the bottom 2 mm of the image, where the band
is cut off.

## The curvature weight

The curvature penalty is the only regulariser of the volumetric model's edge
positions, and its weight sets the trade-off between noise and smoothing.
Thickness RMSE (mm) of the volumetric model on the cylindrical shell of the
tests (one scan) and on the site phantoms (two scans each, `ρc` fixed at
the truth):

| weight | shell, noise 30 | shell, noise 120 | distal radius | femur | femur, no transition | knee tibia |
|---|---|---|---|---|---|---|
| 0.01 | 0.0095 | 0.0280 | | | | |
| 0.03 | 0.0075 | 0.0162 | 0.058 | 0.156 | 0.065 | 0.160 |
| 0.1 | 0.0070 | 0.0100 | 0.042 | 0.150 | 0.063 | 0.158 |
| 0.3 | 0.0079 | 0.0088 | 0.032 | 0.145 | 0.063 | 0.162 |
| 1.0 | 0.0107 | 0.0106 | 0.026 | 0.143 | 0.065 | 0.166 |

The default, 0.3, is the best single compromise. HR-pQCT benefits from
heavier weights; at clinical resolution the choice matters little.

## The clinical knee

`docs/scripts/knee_volumetric.py`: the knee of [clinical_ct.md](clinical_ct.md)
(used with the subject's permission), from each bone's refined periosteal
surface, with the options of `ct_example.py`. There is no truth, so the
methods are compared with each other. Three fits:

* **regularised**: each bone on its own, the other bones excluded, with its
  own estimated `ρc`;
* **volumetric, joint**: all four bones in one fit. They share the scanner,
  so they share one PSF (and one estimated `ρc`);
* **volumetric, matched**: each bone on its own, with the PSF held at the
  joint fit's and `ρc` at the regularised fit's, so that the difference is
  that of the methods alone.

**The PSF.** Fitted bone by bone, the PSF came out at 0.53–0.80 mm in-plane
and 0.57–0.88 mm axially, the patella's with its anisotropy reversed. A
single bone constrains it poorly, and it absorbs whatever the model misses
locally. Fitted to all four bones together it is 0.58 mm in-plane and
0.85 mm along the scanner axis. The joint `ρc` is 1364 HU, against
1383–1535 HU for the bones' own (1D) estimates.

**Thickness and periosteum**, volumetric minus regularised, mean ± sd over
nodes (mm); the median thickness of the regularised fit is 0.54 (femur),
0.91 (tibia), 1.17 (patella) and 0.87 mm (fibula):

| | nodes | joint: thickness | joint: periosteum | matched: thickness | matched: periosteum |
|---|---|---|---|---|---|
| femur | 82323 | +0.06 ± 0.17 | −0.10 ± 0.11 | −0.07 ± 0.16 | −0.06 ± 0.16 |
| tibia | 65263 | −0.04 ± 0.42 | −0.13 ± 0.17 | −0.14 ± 0.37 | −0.09 ± 0.21 |
| patella | 13384 | −0.21 ± 0.37 | −0.18 ± 0.28 | −0.25 ± 0.43 | −0.20 ± 0.33 |
| fibula | 16679 | +0.02 ± 0.18 | −0.11 ± 0.07 | −0.09 ± 0.17 | −0.07 ± 0.12 |

(A negative periosteum difference puts the volumetric periosteum further
out.) With the same PSF and `ρc` (matched), the volumetric cortex is
thinner on every bone, by 0.07–0.25 mm, and its periosteum lies
0.06–0.20 mm further out, so its endosteum lies 0.12–0.44 mm further out:
the 1D fit counts more of the endocortical bone as cortex. That is the
direction of the 1D error on the phantoms. The joint fit, with its lower
`ρc`, reads thicker than the matched one.

**Reliability.** The noise in the soft tissue has a standard deviation of
28–40 HU and a lag-one correlation of 0.63–0.88 along the axes: the pixels
(0.65 mm) are smaller than the blur. The median local standard error of the
thickness is 0.09–0.17 mm (95th percentile 0.2–0.8 mm, more on the
fibula's thinnest nodes), the median misfit 1.0–1.4 (the model accounts for
the soft-tissue noise level at most nodes; the 95th percentile, 2.3–3.4,
marks where it does not). The data share of `t` is small (median 0.02–0.03):
at clinical resolution the curvature penalty carries much of each node's
value, as on the phantoms.

## Conclusions

* At HR-pQCT the volumetric model is clearly better than the 1D methods: an
  unbiased thickness with a third of the regularised fit's error and 5–13×
  its precision, and a periosteal position twice as accurate. It reads the
  porous, trabecularised endosteum that single profiles cannot.
* At clinical resolution it is as good as the best 1D fit, slightly better
  with the right `ρc`. Both are limited by the same two things: `ρc`, which
  cannot be estimated without some thick cortex in view, and the dense
  endocortical bone, which reads as cortex at that resolution.
* Its PSF is recovered globally, including its anisotropy: within 3–17% of
  the effective blur (PSF and voxel integration), mostly under 10%, when
  `ρc` is right. It is a property of the scan: fit it to all bones at once
  (and hold it for per-bone fits). Bone by bone it scatters (0.53–0.80 mm
  in-plane on the clinical knee) and takes thickness with it.
* On the clinical knee, with the same PSF and `ρc`, the volumetric cortex is
  0.07–0.25 mm thinner than the regularised fit's, its endosteum further
  out: the same direction as the 1D error on the phantoms.
* Open: an estimate of `ρc` that is robust when all of the cortex is thin
  (or a reliability flag when it is not identifiable), and a regulariser
  that lets the endosteum of thick cortices vary more freely.

## References

* D. E. Whittier, S. K. Boyd, A. J. Burghardt, J. Paccou, A. Ghasem-Zadeh,
  R. Chapurlat, K. Engelke, M. L. Bouxsein, *Guidelines for the assessment
  of bone density and microarchitecture in vivo using high-resolution
  peripheral quantitative computed tomography*, Osteoporosis International
  31(9):1607–1627, 2020, doi:10.1007/s00198-020-05438-5.
* T. Hildebrand, P. Rüegsegger, *A new method for the model-independent
  assessment of thickness in three-dimensional images*, Journal of
  Microscopy 185(1):67–75, 1997.
* G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
  cortical bone thickness measurement from clinical CT data*, Medical Image
  Analysis 14(3):276–290, 2010.
* N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
  optimization*, Computers in Biology and Medicine, 2025,
  doi:10.1016/j.compbiomed.2025.110636.
