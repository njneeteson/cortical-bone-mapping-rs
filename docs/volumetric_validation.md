# Validation of the volumetric model

The [volumetric model](method.md#volumetric-model) fits the voxels of a band
around the surface instead of one profile per node (the reasoning is in
[volumetric_model.md](volumetric_model.md)). This page asks whether it
measures cortical thickness and the periosteal surface more accurately and
more precisely than the 1D methods, on phantoms of the sites CBM is used at,
at their scanners' resolution, and how it compares on our clinical knee CT.
Run time and memory are in [performance.md](performance.md).

## Methods

### Site phantoms

`cbm.site_phantom(site, seed)` (Rust: `phantoms::sites`) builds each site at
its scanner's resolution with parameters from the literature; the module
documentation lists the sources, and values without one are typical rather
than measured.

| site | phantom | voxels (mm) | PSF σ (mm) | noise | cortex |
|---|---|---|---|---|---|
| distal radius | shaft + ulna, 32 slices | 0.0607 | 0.08 | 40 mg HA/cm³ | 0.85 mm, Tt.Ar ≈ 254 mm² |
| distal tibia | shaft + fibula, 32 slices | 0.0607 | 0.08 | 40 mg HA/cm³ | 1.3 mm, Tt.Ar ≈ 707 mm² |
| proximal femur | head, neck, trochanter, shaft + acetabulum | 1.0 × 0.75 × 0.75 | 0.6 axial, 0.5 in-plane | 30 HU | neck 0.86 superior to 2.79 inferior, head and trochanter 0.8, shaft 3.5 mm |
| knee | tibia + femur with notch, 5–6.5 mm joint space | 0.625 × 0.65 × 0.65 | 0.45 axial, 0.4 in-plane | 22 HU | plates 0.78 (tibia) and 0.32 (femur), side shell 0.8 mm |

![Mid-sections of the site phantoms](images/site_phantoms.png)

All have a porous cortex, trabeculae, and an endocortical transition zone in
which porosity and trabecular density rise towards the endosteum. The true
endosteum is the smooth surface behind that zone, so the image is ambiguous
there, as in real bone.

To check that the HR-pQCT phantoms are realistic,
`docs/scripts/site_microarchitecture.py` measures them with the
second-generation HR-pQCT definitions (Whittier et al. 2020): Tb.Th and Tb.Sp
are the mean local thickness of the trabecular bone and of the marrow space
in the trabecular compartment, and Ct.Th the mean local thickness of the
cortical compartment (Hildebrand & Rüegsegger 1997, computed with
[local-thickness-rs](https://github.com/njneeteson/local-thickness-rs));
BV/TV and Ct.Po are volume fractions. They are measured on the phantom's
exact structure ("truth") and on the rendered image segmented as the
standard analysis does ("scan": Gaussian filter σ 0.8 voxels with support 1,
320 mg HA/cm³ for trabecular bone, 450 mg HA/cm³ for the cortex, within the
true compartments). The same script compares Ct.Th by local thickness with
the mean CBM thickness.

### Accuracy and precision

`docs/scripts/volumetric_validation.py` renders each site's geometry once
without noise and makes five scans by adding independent noise (exactly how
the phantoms add noise), so that all fits share the surface and the nodes.
Every method fits from the true periosteal mask with the same sampling range
(HR-pQCT 1 mm outside and 2.5 mm inside, femur 2.5 and 7 mm, knee 3 and
6 mm) and `rho_c="estimate"`. The regularised method uses the smoothing of
our 2025 paper (σ 1.94 mm; 500 neighbours at HR-pQCT, 128 at clinical
resolution, where the nodes are about ten times sparser), and the volumetric
model the same operator for its density fields and its default curvature
weight (0.3). On the HR-pQCT shafts, all methods report along in-plane
normals. For each node, thickness and periosteal position are compared with
the truth along the reported direction. Bias and RMSE are over nodes and
scans; SD, the precision, is the RMS over nodes of the standard deviation
over the five scans. At the clinical sites every method is also run with
`ρc` fixed at the phantom's true cortical density, to separate the error of
the estimated `ρc` from that of the method.

### Clinical knee

`docs/scripts/knee_volumetric.py` fits the knee of
[clinical_ct.md](clinical_ct.md) (used with the subject's permission) from
each bone's refined periosteal surface, with the options of `ct_example.py`.
There is no truth, so the methods are compared with each other through three
fits: the regularised method on each bone with the other bones excluded and
its own estimated `ρc`; the volumetric model on all four bones together, so
that they share one PSF (they share the scanner) and one estimated `ρc`; and
the volumetric model on each bone with the PSF held at the joint fit's and
`ρc` at the regularised fit's ("matched"), so that the difference is that of
the methods alone.

## Results

### Realism of the HR-pQCT phantoms

Mean of three phantoms each:

| | radius truth | radius scan | tibia truth | tibia scan | in vivo (HR-pQCT) |
|---|---|---|---|---|---|
| BV/TV | 0.21 | 0.27 | 0.26 | 0.38 | 0.12–0.33 |
| Tb.Th (mm) | 0.17 | 0.30 | 0.19 | 0.33 | 0.22–0.29 |
| Tb.Sp (mm) | 0.56 | 0.74 | 0.55 | 0.61 | 0.6–1.0 |
| Ct.Th (mm) | 0.867 | | 1.294 | | 0.77–0.93 / 1.0–1.7 |
| Ct.Po | 5.4% | 0.3% | 8.6% | 0.4% | 1.5–2.5% / 3.5–6.6% |

The cortex and the trabecular spacing are in the in vivo range. The scanned
trabeculae are slightly thicker than in vivo, and the pores are too small to
be resolved, so the scanned porosity is far below in vivo. Both concern the
phantoms' fine structure, not the cortex that CBM measures.

### Ct.Th by local thickness and by CBM

| mean thickness (mm) | radius | tibia |
|---|---|---|
| local thickness of the true cortex (Whittier Ct.Th) | 0.867 | 1.294 |
| true thickness along the in-plane normals (CBM's definition) | 0.827 | 1.264 |
| CBM, local (RMSE) | 0.874 (0.147) | 1.307 (0.176) |
| CBM, regularised, 64 neighbours (RMSE) | 0.876 (0.131) | 1.311 (0.158) |
| CBM, volumetric (RMSE) | 0.835 (0.038) | 1.260 (0.058) |

The two definitions differ. Local thickness is the diameter of the largest
sphere in the cortex, averaged over its volume; CBM's thickness is the
distance between the fitted surfaces along the normal, averaged over the
surface. Volume weighting counts thick regions more (by about
`t̄ + var(t)/t̄`), and for this phantom's ±35% thickness variation it puts
local thickness 2–5% above the true mean along the normals. The 1D fits land
close to local thickness only because they overestimate thickness by about
0.05 mm here (the endocortical transition reads as cortex). The volumetric
model's mean agrees with the truth (+0.008 and −0.004 mm), and its per-node
error is about a third of theirs.

### HR-pQCT

| | t bias | t RMSE | t SD | edge RMSE | edge SD |
|---|---|---|---|---|---|
| radius: local | +0.048 | 0.154 | 0.0206 | 0.010 | 0.0039 |
| radius: local, estimated PSF | +0.042 | 0.145 | 0.0203 | 0.010 | 0.0038 |
| radius: regularised | +0.030 | 0.082 | 0.0044 | 0.010 | 0.0031 |
| radius: volumetric | +0.002 | 0.030 | 0.0009 | 0.006 | 0.0007 |
| tibia: local | +0.046 | 0.183 | 0.0272 | 0.015 | 0.0041 |
| tibia: local, estimated PSF | +0.041 | 0.177 | 0.0272 | 0.014 | 0.0040 |
| tibia: regularised | +0.031 | 0.114 | 0.0076 | 0.013 | 0.0036 |
| tibia: volumetric | −0.012 | 0.034 | 0.0007 | 0.006 | 0.0004 |

(mm.) The volumetric model is nearly unbiased, its thickness error is a
third of the regularised fit's and a fifth of the local fit's, its precision
is 5–11 times better, and its periosteal position is about twice as
accurate. The 1D error is almost all structural, the same in every scan (SD
0.02 against RMSE 0.15 for the local fit), and sits at the endosteum, where
the porous, trabecularised transition makes a single profile ambiguous. The
volumetric model fits all voxels around a node, with the endosteal surface
regularised by its curvature. Its fitted PSF, 0.089 mm, is close to the
effective blur (PSF and voxel integration, `√(0.08² + 0.0607²/12)` =
0.082 mm).

### Clinical CT

With `ρc` estimated / fixed at the truth (mm):

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

With the right `ρc` the volumetric model is the most accurate method at
every clinical site, by a small margin (thickness RMSE 0.146–0.162 mm
against 0.159–0.177 mm for the regularised fit), with a precision between
the local and regularised fits'. Its fitted PSF is close to the effective
blur: 0.566 / 0.686 mm (in-plane / axial) on the femur against
0.545 / 0.666 mm, and 0.461 / 0.536–0.566 mm on the knee against
0.442 / 0.485 mm.

Two effects shared by all methods dominate. The first is the estimated
`ρc`. The estimate takes the density plateau inside cortices that are thick
relative to the blur, and the knee phantom has none: its plates (0.32 and
0.78 mm) and side shell (0.8 mm) are all within `4σ`. The only nodes that
qualify are those fitted too thick, whose plateau is diluted, so the
estimate is low (1140–1190 HU against 1551 HU); since the data fix the
blurred mass `(ρc − ρb) t`, every method then makes the cortex thicker, by
+0.42 to +0.66 mm. With the true `ρc` the bias falls to about +0.1 mm. The
volumetric model is the most sensitive, because the few thick nodes that set
the plateau also pull its PSF down (0.30 / 0.36 mm against 0.46 / 0.54 mm).
On the femur, whose shaft cortex is thick, the estimate is close (about
1540 HU) and fixing `ρc` changes little.

The second is the endocortical transition. With the true `ρc` a bias of
about +0.1 mm remains for every method: the dense trabecular bone the
phantoms put against the endosteum (BV/TV rising to 0.35 within 2–3 mm) is
indistinguishable from cortex at clinical resolution. Without the transition
(the femur with `endo_bvtv = bvtv`, noise-free, true `ρc`), the regularised
and volumetric fits are both unbiased (+0.000 and −0.009 mm), with RMSE 0.048
and 0.059 mm; removing the pores changes nothing. The volumetric error left
there is in the thick shaft cortex (RMSE 0.08 against 0.06 mm), where the
curvature penalty resists the real variation of the endosteal surface
(±0.5 mm over 25 mm), and in the bottom 2 mm of the image, where the band is
cut off.

### Curvature weight

The curvature penalty is the only regulariser of the volumetric model's edge
positions, so its weight sets the trade-off between noise and smoothing.
Thickness RMSE (mm) on the cylindrical shell of the tests (one scan) and on
the site phantoms (two scans each, `ρc` fixed at the truth):

| weight | shell, noise 30 | shell, noise 120 | distal radius | femur | femur, no transition | knee tibia |
|---|---|---|---|---|---|---|
| 0.01 | 0.0095 | 0.0280 | | | | |
| 0.03 | 0.0074 | 0.0161 | 0.058 | 0.156 | 0.065 | 0.160 |
| 0.1 | 0.0069 | 0.0099 | 0.041 | 0.150 | 0.063 | 0.158 |
| 0.3 | 0.0077 | 0.0087 | 0.029 | 0.145 | 0.063 | 0.162 |
| 1.0 | 0.0106 | 0.0104 | 0.021 | 0.143 | 0.065 | 0.166 |

The default, 0.3, is the best single compromise. HR-pQCT benefits from
heavier weights; at clinical resolution the choice matters little.

### Clinical knee

Fitted bone by bone, the PSF came out at 0.53–0.80 mm in-plane and
0.57–0.88 mm axially, the patella's with its anisotropy reversed: a single
bone constrains it poorly, and it absorbs whatever the model misses locally.
Fitted to all four bones together it is 0.58 mm in-plane and 0.85 mm along
the scanner axis. The joint `ρc` is 1364 HU, against 1383–1535 HU for the
bones' own 1D estimates.

Volumetric minus regularised, mean ± SD over nodes (mm); the regularised
median thicknesses are 0.54 (femur), 0.91 (tibia), 1.17 (patella) and
0.87 mm (fibula), and a negative periosteum difference puts the volumetric
periosteum further out:

| | nodes | joint: thickness | joint: periosteum | matched: thickness | matched: periosteum |
|---|---|---|---|---|---|
| femur | 82323 | +0.06 ± 0.17 | −0.10 ± 0.11 | −0.07 ± 0.16 | −0.06 ± 0.16 |
| tibia | 65263 | −0.04 ± 0.42 | −0.13 ± 0.17 | −0.14 ± 0.37 | −0.09 ± 0.21 |
| patella | 13384 | −0.21 ± 0.37 | −0.18 ± 0.28 | −0.25 ± 0.43 | −0.20 ± 0.33 |
| fibula | 16679 | +0.02 ± 0.18 | −0.11 ± 0.07 | −0.09 ± 0.17 | −0.07 ± 0.12 |

With the same PSF and `ρc`, the volumetric cortex is thinner on every bone
by 0.07–0.25 mm and its periosteum lies 0.06–0.20 mm further out, so its
endosteum lies 0.12–0.44 mm further out: the 1D fit counts more of the
endocortical bone as cortex, which is the direction of its error on the
phantoms. The joint fit, with its lower `ρc`, reads thicker than the matched
one.

The soft-tissue noise has a standard deviation of 28–40 HU and a lag-one
correlation of 0.63–0.88 along the axes, because the pixels (0.65 mm) are
smaller than the blur. The median local standard error of the thickness is
0.09–0.17 mm (95th percentile 0.2–0.8 mm, more on the fibula's thinnest
nodes), and the median misfit is 1.0–1.4, so the model accounts for the
noise level at most nodes; the 95th percentile, 2.3–3.4, marks where it does
not. The data share of `t` is small (median 0.02–0.03): at clinical
resolution the curvature penalty carries much of each node's value.

## Discussion

At HR-pQCT the volumetric model is clearly better than the 1D methods: an
almost unbiased thickness with a third of the regularised fit's error and
5–11 times its precision, and a periosteal position twice as accurate,
because it reads the porous, trabecularised endosteum that single profiles
cannot. The same holds on the realistic shafts of
[segmentation.md](segmentation.md).

At clinical resolution it is as good as the best 1D fit, and slightly better
with the right `ρc`. Both are limited by `ρc`, which cannot be estimated
without some cortex thicker than about `4σ` in view, and by the dense
endocortical bone, which reads as cortex at that resolution. This is the
identifiability problem that led Treece et al. (2010) to constrain the
cortical density. In a field of view without thick cortex, `ρc` should be
fixed (from a calibration, or from a scan that includes a shaft); on the
clinical knee, whose field of view includes the shafts, the estimate was
1380–1530 HU.

The PSF is recovered globally with its anisotropy, within 3–17% of the
effective blur (mostly under 10%) when `ρc` is right. It is a property of
the scan, so it should be fitted to all bones at once and held for per-bone
fits; bone by bone it scatters and takes thickness with it.

Open questions: an estimate of `ρc` that is robust when all of the cortex is
thin (or a flag when it is not identifiable); a regulariser that lets the
endosteum of thick cortices vary more freely; and refinement with the
volumetric model, which plateaus at about a tenth of a voxel on phantoms
([examples.md](examples.md#segmentation-refinement)) and does not finish on
real data ([refinement_experiments.md](refinement_experiments.md),
[clinical_ct.md](clinical_ct.md)).

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
