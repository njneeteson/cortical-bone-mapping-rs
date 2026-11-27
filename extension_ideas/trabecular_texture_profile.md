# A profile model that knows trabecular bone is not uniform

*Idea for future work (not implemented). Recorded 2026-11-27, after the
HR-pQCT proximal tibia example.*

## The problem

The profile model (Treece et al. 2010; ours, Neeteson & Boyd 2025) is a
blurred three-level step: soft tissue `ρs`, cortex `ρc` over thickness `t`,
then one trabecular density `ρb` inwards. At clinical resolution the PSF
averages trabecular bone into something close to a constant, and the model
holds. At HR-pQCT resolution (61 µm voxels, PSF σ ≈ 0.12 mm) individual
trabeculae (≈ 0.1–0.3 mm thick, 0.5–1 mm apart) are resolved, and the
interior of the profile oscillates between bone and marrow.

Under the knee's articular surfaces this breaks the model. The subchondral
plate is thin (≈ 0.2–0.8 mm), and right beneath it lies dense subchondral
trabecular bone. A single `ρb` cannot be both the marrow-rich deep
trabecular bone and the dense bone just under the plate, so the fit
absorbs the dense trabeculae into the cortex. On TRIKNEE TR_1 the 1D
regularised fit gives a cortex two to three times deeper than the provided
subchondral plate (Ct.Th 2.21 mm against 0.80 mm in the periarticular
regions; examples page, "HR-pQCT proximal tibia"). The radius and tibia
shafts, with a thick compact cortex over sparse trabeculae, suffer much less,
so a site may need its own model.

## Ideas

### 1. Mean and variance: cortex is homogeneous, trabecular bone is not

The physical distinction that matters here is texture, not mean density:
compact cortex is nearly uniform (only scanner noise), trabecular bone
alternates between bone and marrow. Model both the mean and the variance of
the profile:

```text
y(x) ~ N( f(x; m, t, ρs, ρc, ρb, σ),  v(x) )
v(x) = σn² + σtb² · Φ((x − e) / s)        e = m + t/2, the endosteal edge
```

`σn²` the noise variance (measured in soft tissue, as the reliability
outputs already do), `σtb²` the trabecular texture variance after the PSF,
`s` a transition width (the PSF). Fit by maximum likelihood:

```text
−log L = Σ_x [ (y − f)² / (2 v(x)) + ½ log v(x) ]
```

The endosteal edge `e` is then where the variance switches on, which dense
subchondral trabeculae cannot fake: they raise the mean but also the
variance. `m` stays the periosteal edge of the mean model. This is a small
change (two parameters, `σtb` and `s`, possibly fixed per scan), it keeps
`m` and `t` as the outputs, and the Gauss–Newton machinery generalises
(iteratively reweighted least squares for the variance part).

### 2. A spectral profile model

Go further and model the interior's spatial spectrum. Along a profile (or,
better, in the 3D neighbourhood the volumetric model already uses), the
trabecular region's fluctuations have a power spectrum peaked near
`1/(Tb.Th + Tb.Sp)` (about 1–3 cycles/mm), shaped by the PSF's transfer
function `|H(k)|²`; the cortex's fluctuations are white noise at `σn²` only:

```text
S(x, k) = σn² + |H(k)|² · σtb² · P_tb(k; Tb.Th, Tb.Sp) · 1[x beyond e]
```

`P_tb` from a stationary random two-phase medium with the given thickness
and spacing (e.g. a Boolean or Gaussian-level-cut model). The local spectrum
(short-time Fourier or wavelet transform of the profile) is compared with
`S(x, k)`, jointly with the mean profile. This links the fit to physical
microarchitecture (`Tb.Th`, `Tb.Sp`, comparable with local thickness of the
masks, Whittier et al. 2020) and could estimate them as by-products. It is
heavier: spectra need windows of several trabecular periods (≈ 2–3 mm),
which blurs `e` unless the mean model still places the edges.

### 3. A depth-dependent trabecular density

A cheaper, purely mean-based alternative: let the trabecular density decay
with depth below the endosteal edge,

```text
ρb(x) = ρb∞ + (ρb0 − ρb∞) exp(−(x − e) / λ)
```

so dense subchondral trabecular bone is explained as trabecular bone near
the plate rather than as cortex. Two more parameters per node (or `λ` per
scan). It does not use the texture, so a dense, thin layer of trabeculae and
a thick plate remain hard to tell apart; idea 1 addresses that directly.

### 4. Site-specific models

The knee's subchondral plate and the radius/tibia shaft cortex differ
enough that a preset may need to choose the model, not only its parameters:
the three-level step where the cortex is thick and compact, the texture
model (1 or 2) under articular surfaces.

## Open questions

- Correlated noise (HR-pQCT: about 0.7 between neighbouring voxels) is
  itself texture; the variance model must separate it from trabeculae
  (`σn²` and its correlation measured in soft tissue).
- Identifiability of `σtb` and `s` per node from one profile; probably per
  scan or smoothed fields, as the PSF is.
- Whether the volumetric model's 3D neighbourhoods make the variance (or the
  spectrum) much better determined than 1D profiles.
- The reference: Buie et al. (2007)-style dual-threshold masks place the
  endosteal boundary by morphology; compare against those and against local
  thickness of the provided cortex near the plateaus.

## References

- G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
  cortical bone thickness measurement from clinical CT data*, Medical Image
  Analysis 14(3):276–290, 2010.
- G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
  thickness and cortical bone density using clinical CT*, Medical Image
  Analysis 20(1):249–264, 2015.
- H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
  *Automatic segmentation of cortical and trabecular compartments based on a
  dual threshold technique for in vivo micro-CT bone analysis*, Bone
  41(4):505–515, 2007.
- R. M. Haralick, K. Shanmugam, I. Dinstein, *Textural features for image
  classification*, IEEE Transactions on Systems, Man, and Cybernetics
  3(6):610–621, 1973.
- D. E. Whittier, S. K. Boyd, A. J. Burghardt, et al., *Guidelines for the
  assessment of bone density and microarchitecture in vivo using
  high-resolution peripheral quantitative computed tomography*, Osteoporosis
  International 31:1607–1627, 2020.
