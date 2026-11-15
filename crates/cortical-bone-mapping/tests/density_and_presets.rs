//! Cortical density choices (fixed, estimated, per-node profile maximum) and
//! the Table 1 presets.

mod common;

use common::*;
use cortical_bone_mapping::synthetic::ShellPhantom;
use cortical_bone_mapping::*;

fn phantom() -> ShellPhantom {
    // a cortical density the default (1000) gets wrong
    ShellPhantom {
        rho_c: 1300.0,
        ..cylinder(40.0, 3)
    }
}

fn fit(ph: &ShellPhantom, mode: RhoCMode) -> CbmResult {
    let (img, mask) = ph.generate();
    let mut o = options(ph, Method::Local);
    o.model.rho_c_mode = mode;
    run(img.view(), mask.view(), NO_SUB, &o).unwrap()
}

#[test]
fn estimated_density_is_close_to_the_true_one_and_beats_a_wrong_fixed_one() {
    let ph = phantom();
    let est = fit(&ph, RhoCMode::Estimate);
    assert!((est.rho_c - 1300.0).abs() < 0.04 * 1300.0, "estimate {}", est.rho_c);
    assert!(est.rho_c_node.is_none());
    let (b_est, r_est) = thickness_error(&ph, &est);
    let (b_fix, _) = thickness_error(&ph, &fit(&ph, RhoCMode::Fixed));
    assert!(b_est.abs() < 0.03 && r_est < 0.06, "estimate bias {b_est} rmse {r_est}");
    // ρc = 1000 < 1300: the fit widens the cortex to hold the same mass
    assert!(b_fix > 0.08, "fixed bias {b_fix}");
}

#[test]
fn per_node_profile_maximum_scales_densities_back() {
    let ph = phantom();
    let r = fit(&ph, RhoCMode::ProfileMax);
    let node = r.rho_c_node.as_ref().expect("per-node densities");
    assert_eq!(node.len(), r.n());
    assert_eq!(r.rho_c_at(0), node[0]);
    // densities are reported in image units, not in the scaled ones
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let rs = med(r.param(model::RHO_S));
    let rb = med(r.param(model::RHO_B));
    assert!((rs - ph.rho_s).abs() < 40.0, "rho_s {rs}");
    assert!((rb - ph.rho_b).abs() < 60.0, "rho_b {rb}");
    // the reported common density is the median of the per-node ones
    let mx = med(node.clone());
    assert!((r.rho_c - mx).abs() < 1e-3 * mx, "{} vs {mx}", r.rho_c);
    // per-node maxima are noisy (the maximum of noisy samples), and
    // thickness follows them: worse than with an estimated common density
    let (_, rmse) = thickness_error(&ph, &r);
    let (_, rmse_est) = thickness_error(&ph, &fit(&ph, RhoCMode::Estimate));
    assert!(rmse > rmse_est, "profile max {rmse} vs estimate {rmse_est}");
}

#[test]
fn table1_presets() {
    let o = Options::preset(Site::RadiusTibia, 0.0607);
    assert_eq!(o.sampling.normal_constraint, NormalConstraint::InPlane(0));
    assert_eq!(
        (o.sampling.outside, o.sampling.inside, o.sampling.smoothing),
        (5.0, 15.0, 3.0)
    );
    assert_eq!(o.surface.taubin_passband, 0.1);
    match o.method {
        Method::Regularized {
            lambda,
            sigma,
            neighbours,
            ..
        } => {
            assert_eq!((lambda, neighbours), (1.0, 500));
            assert!((sigma - 32.0 * 0.0607).abs() < 1e-3);
        }
        _ => panic!("regularized expected"),
    }
    let k = Options::preset(Site::Knee, 0.0607);
    assert_eq!(k.sampling.normal_constraint, NormalConstraint::AlongAxis(0));
    assert_eq!(k.surface.taubin_passband, 0.01);
    assert!(matches!(k.method, Method::Regularized { neighbours: 1000, .. }));
    match Method::multigrid_for(Site::Knee, 0.0607) {
        Method::Multigrid {
            spacings, neighbours, ..
        } => {
            let vox: Vec<f64> = spacings.iter().map(|s| (s / 0.0607).round()).collect();
            assert_eq!(vox, vec![256.0, 128.0, 64.0, 32.0, 16.0]);
            assert_eq!(neighbours, 5);
        }
        _ => unreachable!(),
    }
    match Method::multigrid_for(Site::RadiusTibia, 0.0607) {
        Method::Multigrid { spacings, .. } => assert_eq!(spacings.len(), 4),
        _ => unreachable!(),
    }
}

#[test]
fn global_solves_respect_the_work_budget() {
    let ph = cylinder(40.0, 4);
    let (img, mask) = ph.generate();
    let mut o = options(&ph, regularized(64, None));
    let free = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    assert!(!free.info.budget_exhausted && free.info.converged == 1.0);
    o.solver.cg_max_total = 5;
    let capped = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    assert!(capped.info.budget_exhausted && capped.info.converged == 0.0);
    // the global initialisation and the regularised solve each get the budget
    assert!(capped.info.cg_iterations <= 5, "{}", capped.info.cg_iterations);
}
