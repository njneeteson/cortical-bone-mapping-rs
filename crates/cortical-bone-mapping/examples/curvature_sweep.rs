//! Sweep the curvature penalty weight and scale on a noisy phantom.
//!
//!     cargo run --release --example curvature_sweep -p cortical-bone-mapping

use cortical_bone_mapping::phantoms::shell::ShellPhantom;
use cortical_bone_mapping::*;

fn main() {
    let ph = ShellPhantom {
        noise: 150.0,
        seed: 4,
        ..ShellPhantom::default()
    };
    let (img, mask) = ph.generate();
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
    let report = |name: String, o: &Options| {
        let r = run(img.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, o).unwrap();
        let e: Vec<f64> = r
            .points
            .iter()
            .zip(r.thickness())
            .map(|(p, t)| t - ph.true_thickness(*p))
            .collect();
        let bias = e.iter().sum::<f64>() / e.len() as f64;
        let rmse = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
        println!(
            "{name:<34} bias {bias:+.4}  rmse {rmse:.4}  outer {:>3}  cg {:>5}",
            r.info.iterations, r.info.cg_iterations
        );
    };
    report(
        "local".into(),
        &Options {
            method: Method::Local,
            ..base.clone()
        },
    );
    for (scale, weights) in [
        (LaplacianScale::Physical(0.25), vec![0.1, 1.0, 10.0]),
        (LaplacianScale::Physical(0.5), vec![1.0, 10.0, 100.0]),
        (LaplacianScale::Physical(1.0), vec![1.0, 10.0, 100.0, 1000.0]),
        (LaplacianScale::Cotangent, vec![0.01, 0.1, 1.0]),
    ] {
        for w in weights {
            let o = Options {
                method: Method::Regularized {
                    lambda: 0.0,
                    sigma: 1.0,
                    neighbours: 1,
                    cutoff: 3.0,
                    init_from_local: false,
                    curvature: Some(CurvatureOptions {
                        weight: w,
                        scale,
                        ..Default::default()
                    }),
                },
                ..base.clone()
            };
            report(format!("curvature scale={scale:?} weight={w}"), &o);
        }
    }
    for lambda in [0.1, 1.0, 10.0] {
        let o = Options {
            method: Method::Regularized {
                lambda,
                sigma: 1.0,
                neighbours: 64,
                cutoff: 3.0,
                init_from_local: false,
                curvature: None,
            },
            ..base.clone()
        };
        report(format!("neeteson lambda={lambda} k=64"), &o);
    }
}
