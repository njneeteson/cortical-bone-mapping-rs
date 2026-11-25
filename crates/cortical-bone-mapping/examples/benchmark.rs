//! Run time and peak memory of every method on cylinder phantoms of
//! increasing size (number of surface nodes).
//!
//!     cargo run --release --example benchmark -p cortical-bone-mapping [--json out.json] [--volumetric]
//!
//! `--volumetric` instead runs ablations of the volumetric model on the
//! largest phantom (density smoothing over 64 neighbours, global densities,
//! no curvature penalty), to show where its time and memory go.
//!
//! Peak memory is the largest amount of heap allocated *during* the call,
//! above what was allocated before it (image + mask). The phantoms mimic a
//! distal-tibia-sized cortex at HR-pQCT resolution (60.7 µm voxels) with a
//! growing number of slices.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cortical_bone_mapping::phantoms::shell::ShellPhantom;
use cortical_bone_mapping::*;

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_path = args.iter().position(|a| a == "--json").map(|i| args[i + 1].clone());
    let ablation = args.iter().any(|a| a == "--volumetric");
    let slices: Vec<usize> = if ablation { vec![128] } else { vec![8, 32, 128] };
    let mut rows = Vec::new();
    for &d0 in &slices {
        let h = 0.0607;
        let ph = ShellPhantom {
            dims: (d0, 300, 300),
            spacing: [h; 3],
            radius: 8.0,
            thickness_mean: 1.2,
            thickness_amplitude: 0.5,
            blur: [0.1; 3],
            noise: 60.0,
            supersample: 1,
            ..ShellPhantom::default()
        };
        let (image, mask) = ph.generate();
        let methods = if ablation {
            volumetric_ablations()
        } else {
            all_methods()
        };
        for (name, method) in methods {
            let o = Options {
                geometry: ph.geometry(),
                sampling: SamplingOptions {
                    outside: 1.0,
                    inside: 3.0,
                    normal_constraint: NormalConstraint::InPlane(0),
                    ..Default::default()
                },
                method,
                ..Default::default()
            };
            let base = CURRENT.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            let t0 = Instant::now();
            let r = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &o).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            let peak = PEAK.load(Ordering::Relaxed) - base;
            let e: Vec<f64> = r
                .points
                .iter()
                .zip(r.thickness())
                .map(|(p, t)| t - ph.true_thickness(*p))
                .collect();
            let rmse = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
            let m = (o.sampling.outside + o.sampling.inside) / ph.spacing[0];
            println!(
                "{:<18} nodes={:>7} samples={:>3.0} time={:>7.3}s peak={:>7.1} MB ({:>5.0} B/node) rmse={:.4} mm  iters={} cg={}",
                name,
                r.n(),
                m + 1.0,
                dt,
                peak as f64 / 1e6,
                peak as f64 / r.n() as f64,
                rmse,
                r.info.iterations,
                r.info.cg_iterations
            );
            let timings: Vec<String> = r.info.timings.iter().map(|(k, v)| format!("\"{k}\": {v:.6}")).collect();
            rows.push(format!(
                "{{\"method\": \"{name}\", \"nodes\": {}, \"time\": {dt:.6}, \"peak_bytes\": {peak}, \"rmse\": {rmse:.6}, \"iterations\": {}, \"cg_iterations\": {}, \"timings\": {{{}}}}}",
                r.n(),
                r.info.iterations,
                r.info.cg_iterations,
                timings.join(", ")
            ));
        }
    }
    if let Some(p) = json_path {
        std::fs::write(&p, format!("[\n  {}\n]\n", rows.join(",\n  "))).unwrap();
        println!("wrote {p}");
    }
}

/// Every method, with the settings of the benchmarks page of the documentation.
fn all_methods() -> Vec<(&'static str, Method)> {
    vec![
        ("local", Method::Local),
        ("multigrid", Method::multigrid()),
        (
            "regularized k=64",
            Method::Regularized {
                lambda: 1.0,
                sigma: 1.9424,
                neighbours: 64,
                cutoff: 3.0,
                init_from_local: false,
                curvature: None,
            },
        ),
        ("regularized k=500", Method::regularized()),
        (
            "reg+curvature",
            Method::Regularized {
                lambda: 1.0,
                sigma: 1.9424,
                neighbours: 64,
                cutoff: 3.0,
                init_from_local: false,
                curvature: Some(CurvatureOptions::default()),
            },
        ),
        // a shaft: every normal is in-plane, so the axial blur is tied
        (
            "volumetric",
            Method::Volumetric(VolumetricOptions {
                sigma_z: AxialBlur::Tied,
                ..Default::default()
            }),
        ),
    ]
}

/// The volumetric model and variants without its costliest parts.
fn volumetric_ablations() -> Vec<(&'static str, Method)> {
    let tied = VolumetricOptions {
        sigma_z: AxialBlur::Tied,
        ..Default::default()
    };
    vec![
        ("volumetric", Method::Volumetric(tied.clone())),
        (
            "volumetric k=64",
            Method::Volumetric(VolumetricOptions {
                neighbours: 64,
                ..tied.clone()
            }),
        ),
        (
            "volumetric, global densities",
            Method::Volumetric(VolumetricOptions {
                global_densities: true,
                edge_lambda: 0.0,
                global_soft_tissue: false,
                adaptive_band: None,
                ..tied.clone()
            }),
        ),
        (
            "volumetric, no curvature",
            Method::Volumetric(VolumetricOptions {
                curvature: None,
                ..tied
            }),
        ),
    ]
}
