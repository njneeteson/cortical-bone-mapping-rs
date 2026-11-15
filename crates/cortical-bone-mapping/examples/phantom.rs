//! Fit every method to a synthetic cylinder phantom and report accuracy.
//!
//!     cargo run --release --example phantom -p cortical-bone-mapping

use std::time::Instant;

use cortical_bone_mapping::synthetic::ShellPhantom;
use cortical_bone_mapping::*;

fn main() {
    let ph = ShellPhantom {
        noise: 40.0,
        ..ShellPhantom::default()
    };
    let (image, mask) = ph.generate();
    let base = Options {
        geometry: ph.geometry(),
        sampling: SamplingOptions {
            outside: 1.5,
            inside: 2.5,
            normal_constraint: NormalConstraint::InPlane(0),
            ..Default::default()
        },
        ..Default::default()
    };
    let methods = vec![
        ("local", Method::Local, SigmaMode::Free),
        (
            "local+psf",
            Method::Local,
            SigmaMode::EstimatePsf {
                model: PsfModel::Isotropic,
                min_ratio: 3.0,
            },
        ),
        ("multigrid", Method::multigrid(), SigmaMode::Free),
        ("regularized", Method::regularized(), SigmaMode::Free),
        (
            "reg+curv",
            Method::Regularized {
                lambda: 1.0,
                sigma: 1.9424,
                neighbours: 100,
                cutoff: 3.0,
                init_from_local: false,
                curvature: Some(CurvatureOptions::default()),
            },
            SigmaMode::Free,
        ),
    ];
    for (name, method, sigma) in methods {
        let mut o = base.clone();
        o.method = method;
        o.model.sigma = sigma;
        let t0 = Instant::now();
        let r = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &o).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let err: Vec<f64> = r
            .points
            .iter()
            .zip(r.thickness())
            .map(|(p, t)| t - ph.true_thickness(*p))
            .collect();
        let bias = err.iter().sum::<f64>() / err.len() as f64;
        let rmse = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
        println!(
            "{name:<12} n={:<6} bias={bias:+.4} rmse={rmse:.4} time={dt:.2}s iters={} cg={} psf={:?} timings={:?}",
            r.n(),
            r.info.iterations,
            r.info.cg_iterations,
            r.psf,
            r.info.timings
        );
    }
}
