//! Benchmarks of the volumetric fit's stages at HR-pQCT size.
//!
//! The input is the distal radius phantom (32 slices of 60.7 µm voxels,
//! about 30 000 nodes), fitted on its true periosteal mask with the
//! HR-pQCT volumetric options (paper sampling range, `σ_reg` = 1.942 mm,
//! normals unconstrained). Timing only: the phantom is not used to judge
//! results.
//!
//!     cargo bench -p cortical-bone-mapping --features bench --bench volumetric
//!     cargo bench ... -- --save-baseline before      # then, after a change:
//!     cargo bench ... -- --baseline before

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array3;

use cortical_bone_mapping::fit::volumetric::bench::Setup;
use cortical_bone_mapping::options::Site as Preset;
use cortical_bone_mapping::phantoms::sites::{Site, SitePhantom};
use cortical_bone_mapping::{Method, NormalConstraint, Options, VolumetricOptions};

/// The lag-one noise correlation measured on an XtremeCT II tibia scan.
const HRPQCT_NOISE_CORRELATION: f64 = 0.69;
/// The radius phantom's cortical density.
const RHO_C: f64 = 1000.0;

struct Input {
    image: Array3<f32>,
    mask: Array3<bool>,
    voxel: f64,
}

fn input() -> Input {
    let SitePhantom::Bone(b) = Site::DistalRadius.phantom(1) else {
        unreachable!("the distal radius is a shaft phantom")
    };
    let (image, mask, _) = b.generate();
    Input {
        image,
        mask,
        voxel: b.spacing[0],
    }
}

/// The HR-pQCT volumetric options with `neighbours` in the smoothing `ω`.
fn options(voxel: f64, neighbours: usize) -> Options {
    let mut o = Options::preset(Preset::Knee, voxel);
    o.sampling.normal_constraint = NormalConstraint::None;
    o.method = Method::Volumetric(VolumetricOptions {
        sigma: 1.9424,
        neighbours,
        ..VolumetricOptions::default()
    });
    o
}

fn slow(c: &mut Criterion) -> criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> {
    let mut g = c.benchmark_group("volumetric");
    g.sample_size(10)
        .measurement_time(Duration::from_secs(20))
        .warm_up_time(Duration::from_secs(2));
    g
}

fn stages(c: &mut Criterion) {
    let inp = input();
    let o = options(inp.voxel, 1000);
    let setup = Setup::new(inp.image.view(), inp.mask.view(), &o).expect("setup");
    let band = setup.band();
    eprintln!("{} nodes, {} band voxels", setup.nodes(), band.voxels());
    let mut g = slow(c);
    g.bench_function("band", |b| b.iter(|| setup.band().voxels()));
    setup.with_parts(&band, RHO_C, |p| {
        let v: Vec<f64> = (0..p.variables()).map(|i| ((i * 7919) % 1000) as f64 * 1e-3).collect();
        let mut out = vec![0.0; p.variables()];
        g.bench_function("linearise", |b| b.iter(|| p.linearise()));
        g.bench_function("product/whole", |b| b.iter(|| p.product(&v, &mut out)));
        g.bench_function("product/data", |b| b.iter(|| p.data_product(&v, &mut out)));
        g.bench_function("product/curvature", |b| b.iter(|| p.curvature_product(&v, &mut out)));
        for rho in [0.0, HRPQCT_NOISE_CORRELATION] {
            g.bench_with_input(BenchmarkId::new("node_moments/correlation", rho), &rho, |b, &r| {
                b.iter(|| p.node_moments([r; 3]))
            });
        }
    });
    g.finish();
}

fn smoothing(c: &mut Criterion) {
    let inp = input();
    let mut g = slow(c);
    for k in [64usize, 500, 1000] {
        let o = options(inp.voxel, k);
        let setup = Setup::new(inp.image.view(), inp.mask.view(), &o).expect("setup");
        let band = setup.band();
        setup.with_parts(&band, RHO_C, |p| {
            let v: Vec<f64> = (0..p.variables()).map(|i| ((i * 7919) % 1000) as f64 * 1e-3).collect();
            let mut out = vec![0.0; p.variables()];
            g.bench_with_input(BenchmarkId::new("product/smoothing/neighbours", k), &k, |b, _| {
                b.iter(|| p.smoothing_product(&v, &mut out))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, stages, smoothing);
criterion_main!(benches);
