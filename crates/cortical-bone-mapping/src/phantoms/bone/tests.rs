use super::*;

#[test]
fn erfinv_is_inverse() {
    for &x in &[-0.9, -0.3, 0.0, 0.2, 0.6, 0.95, 0.999] {
        assert!((libm::erf(erfinv(x)) - x).abs() < 1e-6, "{x}");
    }
}

#[test]
fn phantom_statistics_and_truth() {
    let ph = BonePhantom {
        dims: (12, 160, 160),
        noise: 0.0,
        ..BonePhantom::default()
    };
    let (img, peri, endo) = ph.generate();
    // deep marrow (away from the transition zone)
    let c0 = ph.coeffs();
    let g0 = ph.geometry();
    let deep = Array3::from_shape_fn(ph.dims, |(i, j, k)| {
        let (r, _, big_r, t) = ph.polar(&c0, g0.to_physical([i as f64, j as f64, k as f64]));
        r < big_r - t - 4.0 * ph.transition_width
    });
    let marrow: Vec<f32> = img
        .iter()
        .zip(deep.iter())
        .filter(|(_, &e)| e)
        .map(|(&v, _)| v)
        .collect();
    assert!(endo.iter().filter(|&&b| b).count() > marrow.len());
    // marrow-space mean ≈ ρm + BV/TV (ρb − ρm)
    let mean = marrow.iter().map(|&v| v as f64).sum::<f64>() / marrow.len() as f64;
    let want = ph.rho_marrow + ph.bvtv * (ph.rho_bone - ph.rho_marrow);
    assert!((mean - want).abs() < 0.08 * want, "marrow mean {mean} vs {want}");
    // the cortex exists everywhere and is not a circle
    assert!(peri.iter().filter(|&&b| b).count() > endo.iter().filter(|&&b| b).count());
    let c = ph.coeffs();
    let g = ph.geometry();
    let mid = [6.0 * g.spacing[0], 0.0, 0.0];
    let rs: Vec<f64> = (0..36)
        .map(|i| {
            let th = i as f64 * 10f64.to_radians();
            let cc = ph.center_at(&c, mid[0]);
            ph.polar(&c, [mid[0], cc[0] + th.cos(), cc[1] + th.sin()]).2
        })
        .collect();
    let (lo, hi) = rs
        .iter()
        .fold((f64::INFINITY, 0.0f64), |a, &r| (a.0.min(r), a.1.max(r)));
    assert!(hi - lo > 0.3, "radius range {lo}..{hi}");
    // edges along a radial line from outside agree with the implicit surfaces
    let cc = ph.center_at(&c, mid[0]);
    let p = [mid[0], cc[0] + 7.0, cc[1]];
    let e = ph.edges_along(&[p], &[[0.0, -1.0, 0.0]], 0.0, 7.0)[0];
    let (_, _, big_r, t) = ph.polar(&c, [mid[0], cc[0] + 4.0, cc[1]]);
    assert!(((7.0 - e.0) - big_r).abs() < 1e-6, "{e:?} R {big_r}");
    assert!(((e.1 - e.0) - t).abs() < 1e-6);
}

#[test]
fn neighbour_and_breaks() {
    let gap = 0.5;
    let ph = BonePhantom {
        dims: (6, 200, 200),
        noise: 0.0,
        neighbour: Some(NeighbourBone {
            radius: 1.2,
            thickness: 0.4,
            gap,
            angle: std::f64::consts::FRAC_PI_4,
        }),
        breaks: vec![CorticalBreak {
            angle: -2.0,
            width: 1.0,
            z_center: 0.15,
            length: 0.2,
        }],
        ..BonePhantom::default()
    };
    let (_, peri, _) = ph.generate();
    let lab = ph.labels();
    let c = ph.coeffs();
    let g = ph.geometry();
    // the neighbour is outside the main bone, at the requested gap
    assert!(lab.iter().zip(peri.iter()).all(|(&l, &p)| !(l == 1 && p)));
    let z = 3.0 * g.spacing[0];
    let a = std::f64::consts::FRAC_PI_4;
    let cc = ph.center_at(&c, z);
    let r = ph.radius_at(&c, a, z);
    let at = |d: f64| [z, cc[0] + d * a.cos(), cc[1] + d * a.sin()];
    assert!(ph.neighbour_distance(&c, at(r + 0.5 * gap)).unwrap().0 > 1.2);
    assert!(ph.neighbour_distance(&c, at(r + gap + 0.05)).unwrap().0 < 1.2);
    // breaks: soft tissue through the cortex in the labelled slices only
    let nb = lab.iter().filter(|&&l| l == 2).count();
    assert!(nb > 0);
    // |z − 0.15| < 0.1 mm: slices 1–4 at 60.7 µm
    for (idx, &l) in lab.indexed_iter() {
        if l == 2 {
            assert!((1..=4).contains(&idx.0), "break at slice {}", idx.0);
        }
    }
    let mid = 2.0 * g.spacing[0];
    let cc = ph.center_at(&c, mid);
    let (_, _, big_r, t) = ph.polar(&c, [mid, cc[0] + (-2f64).cos(), cc[1] + (-2f64).sin()]);
    let p = [
        mid,
        cc[0] + (big_r - 0.5 * t) * (-2f64).cos(),
        cc[1] + (big_r - 0.5 * t) * (-2f64).sin(),
    ];
    assert_eq!(ph.density(&c, p), ph.rho_soft);
}
