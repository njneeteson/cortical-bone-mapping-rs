//! Refine a deliberately poor segmentation of a phantom.
//!
//!     cargo run --release --example refine -p cortical-bone-mapping

use cortical_bone_mapping::synthetic::ShellPhantom;
use cortical_bone_mapping::*;

fn dice(a: &ndarray::Array3<bool>, b: &ndarray::Array3<u8>) -> f64 {
    let (mut inter, mut sa, mut sb) = (0usize, 0usize, 0usize);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let y = y != 0;
        inter += (x && y) as usize;
        sa += x as usize;
        sb += y as usize;
    }
    2.0 * inter as f64 / (sa + sb) as f64
}

fn main() {
    let ph = ShellPhantom {
        mask_offset: 0.35,
        mask_bump: 0.25,
        noise: 30.0,
        ..ShellPhantom::default()
    };
    let (image, bad_mask) = ph.generate();
    let truth_mask = ShellPhantom {
        mask_offset: 0.0,
        mask_bump: 0.0,
        ..ph.clone()
    }
    .generate()
    .1;
    let opts = Options {
        geometry: ph.geometry(),
        sampling: SamplingOptions {
            outside: 1.5,
            inside: 2.5,
            normal_constraint: NormalConstraint::InPlane(0),
            ..Default::default()
        },
        method: Method::Regularized {
            lambda: 1.0,
            sigma: 1.0,
            neighbours: 100,
            cutoff: 3.0,
            init_from_local: false,
            curvature: None,
        },
        ..Default::default()
    };
    let r = refine_segmentation(
        image.view(),
        bad_mask.view(),
        None::<ndarray::ArrayView3<u8>>,
        &opts,
        &RefineOptions::default(),
    )
    .unwrap();
    let bad_as_bool = bad_mask.mapv(|v| v != 0);
    println!("Dice(bad, truth)     = {:.4}", dice(&bad_as_bool, &truth_mask));
    println!("Dice(refined, truth) = {:.4}", dice(&r.periosteal_mask, &truth_mask));
    for (i, h) in r.history.iter().enumerate() {
        println!(
            "iter {i:>2}: rms move {:.4}  max move {:.4}  mean t {:.4}  residual {:.2}",
            h.rms_displacement, h.max_displacement, h.mean_thickness, h.mean_residual
        );
    }
    // radial error of the fitted (non-cap) vertices, before and after
    let res = &r.result;
    let err = |m: &Mesh| -> (f64, f64) {
        let off: Vec<f64> = res
            .vertex_ids
            .iter()
            .map(|&v| ph.radial_offset(m.vertices[v as usize]))
            .collect();
        let rms = (off.iter().map(|o| o * o).sum::<f64>() / off.len() as f64).sqrt();
        let max = off.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        (rms, max)
    };
    let (rms, max) = err(&r.periosteal);
    println!(
        "converged={} radial error of refined surface: rms {rms:.4} mm, max {max:.4} mm",
        r.converged
    );
    // where are the worst vertices?
    let mut worst: Vec<(f64, f64)> = res
        .vertex_ids
        .iter()
        .map(|&v| {
            let p = r.periosteal.vertices[v as usize];
            (ph.radial_offset(p).abs(), p[0])
        })
        .collect();
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("worst (|err|, axial pos): {:?}", &worst[..8]);
}
