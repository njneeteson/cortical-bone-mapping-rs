# cortical-bone-mapping-rs

Cortical bone mapping (CBM) in Rust, with Python bindings: model-based
measurement of cortical thickness, and automatic segmentation of the
periosteal and endosteal surfaces, from CT and HR-pQCT images.

At every vertex of the bone's outer surface, CBM samples a density profile
along the inward normal and fits a blurred three-level step model (soft
tissue, cortex, trabecular bone). The result at every surface point is the
cortical thickness `t`, the position `m` of the cortical mid-line, the blur
`σ`, and the soft-tissue and trabecular densities `ρs` and `ρb`.

![A profile and its fitted model](images/profile_fit.png)

The package implements the original method of Treece et al. (2010), the
global multigrid and spatially regularised fits of our 2025 paper (Neeteson
& Boyd, *Comput. Biol. Med.*), and three extensions: a curvature penalty on
the fitted surfaces, a volumetric model that fits every voxel of a band
around the surface, and segmentation refinement, which moves a rough
segmentation onto the fitted periosteum.

```python
import cortical_bone_mapping_rs as cbm

start = cbm.hull_segmentation(image)                        # from the image alone
options = cbm.Options.preset("radius_tibia", rho_c="estimate")
rr = cbm.refine_segmentation(image, start, options=options, method="volumetric")
rr.result.thickness                                         # mm, per surface node
rr.periosteal_mask, rr.endosteal_mask, rr.cortical_mask
```

![A rough segmentation (red) tightening onto the periosteum](images/refinement.gif)

## Where to start

- [Installation](guide/install.md), then the [quick start](guide/quickstart.md),
  which runs every step on a phantom.
- [Choosing a method](guide/methods.md) and
  [scanners and parameters](guide/scanners.md) for your data.
- [Automatic segmentation](guide/refinement.md) if you have no segmentation,
  and [several bones](guide/several_bones.md) for a scan with more than one.
- The [API reference](api.md) for every function and option.
- [Method](method.md) for the model, the solvers and the extensions, and
  the validation pages for how well each part works.

```{toctree}
:hidden:
:caption: User guide

guide/install
guide/quickstart
guide/methods
guide/scanners
guide/refinement
guide/several_bones
guide/results
guide/rust
```

```{toctree}
:hidden:
:caption: Reference

api
```

```{toctree}
:hidden:
:caption: Method and validation

method
examples
segmentation
clinical_ct
volumetric_model
volumetric_validation
volumetric_refinement
refinement_experiments
performance
```

## Citing

If you use this software, please cite our paper and this implementation
(see `CITATION.cff` in the repository):

> N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
> optimization*, Computers in Biology and Medicine, 2025,
> [doi:10.1016/j.compbiomed.2025.110636](https://doi.org/10.1016/j.compbiomed.2025.110636)

and the original method:

> G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
> cortical bone thickness measurement from clinical CT data*, Medical Image
> Analysis 14(3):276–290, 2010.
