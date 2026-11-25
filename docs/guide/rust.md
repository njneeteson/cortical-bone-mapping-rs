# Using the Rust crate

The Python package wraps the `cortical-bone-mapping` crate, which can be
used directly. Its API documentation, with the equations and sources of every
module, is on [docs.rs](https://docs.rs/cortical-bone-mapping) once the crate
is published; until then, build it locally with
`cargo doc -p cortical-bone-mapping --open`.

Images are `ndarray` 3D arrays of any of the supported voxel types (`u8`
to `f64`), read in place. Options are plain structs with defaults:

```rust
use cortical_bone_mapping::{run, refine_segmentation, Method, Options, RefineOptions, Site, VolumetricOptions};

// Table 1 of Neeteson & Boyd (2025) for the radius and tibia at 60.7 µm
let opts = Options::preset(Site::RadiusTibia, 0.0607);
let res = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &opts)?;
let thickness: Vec<f64> = res.thickness();

// refinement with the volumetric model
let opts = Options { method: Method::Volumetric(VolumetricOptions::default()), ..opts };
let rr = refine_segmentation(image.view(), start.view(), None::<ndarray::ArrayView3<u8>>, &opts,
                             &RefineOptions::default())?;
let cortex = &rr.cortical_mask;
```

| Python | Rust |
|---|---|
| `cbm.cortical_bone_mapping` | `run` |
| `cbm.refine_segmentation` | `refine_segmentation`, `refine_segmentation_excluding` |
| `cbm.Options` (flat) | `Options` with `geometry`, `surface`, `sampling`, `model`, `solver`, `method` |
| `method="volumetric"` and `vol_*` | `Method::Volumetric(VolumetricOptions)` |
| `cbm.hull_segmentation`, `cbm.buie_segmentation`, `cbm.label_bones` | `segmentation` module |
| phantoms | `phantoms` module |

The per-bone helpers (`map_bones`, `refine_bones`, `shared_scanner`) are
written in Python on top of these.

The examples in the repository run every method on a phantom, refine a
poor segmentation, and time the pipeline:

```bash
cargo run --release --example phantom -p cortical-bone-mapping
cargo run --release --example refine -p cortical-bone-mapping
cargo run --release --example benchmark -p cortical-bone-mapping
```
