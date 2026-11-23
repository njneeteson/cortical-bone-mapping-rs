//! The volumetric model: recovery of thickness, PSF and noise on phantoms,
//! its reliability outputs, and segmentation refinement with it.

mod common;
use common::*;
use cortical_bone_mapping::phantoms::shell::ShellPhantom;
use cortical_bone_mapping::*;

fn volumetric(v: VolumetricOptions) -> Method {
    Method::Volumetric(v)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn recovers_thickness_psf_and_noise_on_a_noisy_cylinder() {
    let ph = cylinder(40.0, 1);
    let (img, mask) = ph.generate();
    // a shaft: every normal is in-plane, so the axial blur is not identified
    let v = VolumetricOptions {
        sigma_z: AxialBlur::Tied,
        ..Default::default()
    };
    let r = run(img.view(), mask.view(), NO_SUB, &options(&ph, volumetric(v))).unwrap();
    let (bias, rmse) = thickness_error(&ph, &r);
    assert!(bias.abs() < 0.01 && rmse < 0.02, "bias {bias} rmse {rmse}");
    assert!(periosteal_error(&ph, &r) < 0.02);
    let info = r.volumetric.as_ref().expect("volumetric outputs");
    // the phantom's PSF (0.12) plus the partial-volume blur of the voxels
    assert!((info.sigma_xy - 0.12).abs() < 0.02, "sigma_xy {}", info.sigma_xy);
    assert_eq!(info.sigma_xy, info.sigma_z);
    assert!((info.noise_sd / 40.0 - 1.0).abs() < 0.1, "noise {}", info.noise_sd);
    let n = r.n();
    assert!(info.std_error_t.len() == n && info.data_share.len() == n && info.misfit.len() == n);
    let misfit = median(info.misfit.clone());
    assert!((misfit - 1.0).abs() < 0.15, "misfit {misfit}");
    for s in info.data_share.iter().flatten() {
        assert!((0.0..=1.0).contains(s), "data share {s}");
    }
    // the reported error of t is of the order of the actual error
    let se = median(info.std_error_t.clone());
    assert!(se > 0.2 * rmse && se < 5.0 * rmse, "se {se} vs rmse {rmse}");
}

#[test]
fn estimates_an_anisotropic_psf_on_a_sphere() {
    let ph = ShellPhantom {
        blur: [0.2, 0.1, 0.1],
        noise: 10.0,
        ..sphere(0.0)
    };
    let (img, mask) = ph.generate();
    let r = run(
        img.view(),
        mask.view(),
        NO_SUB,
        &options(&ph, volumetric(Default::default())),
    )
    .unwrap();
    let info = r.volumetric.as_ref().unwrap();
    assert!(
        info.sigma_z > info.sigma_xy + 0.05,
        "psf {} {}",
        info.sigma_xy,
        info.sigma_z
    );
    let (bias, rmse) = thickness_error(&ph, &r);
    assert!(bias.abs() < 0.02 && rmse < 0.04, "bias {bias} rmse {rmse}");
    // a fixed axial blur is kept
    let v = VolumetricOptions {
        sigma_z: AxialBlur::Fixed(0.21),
        ..Default::default()
    };
    let f = run(img.view(), mask.view(), NO_SUB, &options(&ph, volumetric(v))).unwrap();
    assert_eq!(f.volumetric.unwrap().sigma_z, 0.21);
}

#[test]
fn regularisation_improves_precision_under_heavy_noise() {
    let ph = cylinder(120.0, 2);
    let (img, mask) = ph.generate();
    let local = run(img.view(), mask.view(), NO_SUB, &options(&ph, Method::Local)).unwrap();
    let v = VolumetricOptions {
        sigma_z: AxialBlur::Tied,
        ..Default::default()
    };
    let vol = run(img.view(), mask.view(), NO_SUB, &options(&ph, volumetric(v))).unwrap();
    let (e_local, e_vol) = (thickness_error(&ph, &local).1, thickness_error(&ph, &vol).1);
    assert!(e_vol < 0.6 * e_local, "rmse {e_vol} vs local {e_local}");
}

#[test]
fn refinement_with_the_volumetric_model_snaps_to_the_periosteum() {
    let ph = ShellPhantom {
        mask_offset: 0.35,
        mask_bump: 0.25,
        noise: 30.0,
        ..ShellPhantom::default()
    };
    let (img, bad) = ph.generate();
    let truth = ShellPhantom {
        mask_offset: 0.0,
        mask_bump: 0.0,
        ..ph.clone()
    }
    .generate()
    .1;
    let v = VolumetricOptions {
        sigma_z: AxialBlur::Tied,
        ..Default::default()
    };
    let o = options(&ph, volumetric(v));
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &RefineOptions::default()).unwrap();
    assert!(r.converged);
    let (before, after) = (dice(&bad.mapv(|v| v != 0), &truth), dice(&r.periosteal_mask, &truth));
    assert!(before < 0.92 && after > 0.995, "dice {before} -> {after}");
    assert!(periosteal_error(&ph, &r.result) < 0.02);
    assert!(thickness_error(&ph, &r.result).1 < 0.03);
    assert!(r.result.volumetric.is_some());
}
