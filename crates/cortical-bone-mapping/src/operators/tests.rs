use super::*;
use crate::surface::mesh::Mesh;
use crate::util::{scale, Rng, V3};

#[test]
fn many_vector_products_match_single_ones_bitwise() {
    let mut rng = Rng::new(3);
    let pts: Vec<V3> = (0..20000)
        .map(|_| [rng.uniform() * 50.0, rng.uniform() * 50.0, rng.uniform() * 5.0])
        .collect();
    let op = SmoothingOperator::new(&pts, 40, 2.0, 3.0);
    let xs: Vec<Vec<f64>> = (0..3)
        .map(|_| {
            (0..pts.len())
                .map(|i| if i % 7 == 0 { 0.0 } else { rng.normal() })
                .collect()
        })
        .collect();
    let refs: Vec<&[f64]> = xs.iter().map(|x| x.as_slice()).collect();
    let mut many = vec![vec![0.0; pts.len()]; 3];
    let mut many_t = vec![vec![0.0; pts.len()]; 3];
    op.apply_many(&refs, &mut many);
    op.apply_t_many(&refs, &mut many_t);
    for c in 0..3 {
        let mut one = vec![0.0; pts.len()];
        op.apply(&xs[c], &mut one);
        assert_eq!(one, many[c]);
        op.apply_t(&xs[c], &mut one);
        assert_eq!(one, many_t[c]);
    }
}

fn grid_points(n: usize, h: f64) -> Vec<V3> {
    let mut v = Vec::new();
    for i in 0..n {
        for j in 0..n {
            v.push([i as f64 * h, j as f64 * h, 0.0]);
        }
    }
    v
}

#[test]
fn sparse_transpose_matches_dense() {
    let pts = grid_points(90, 0.1); // 8100 rows, multiple chunks
    let ctrl_idx = select_control_points(&pts, 0.5, 3);
    let ctrl: Vec<V3> = ctrl_idx.iter().map(|&i| pts[i as usize]).collect();
    let a = interpolation_matrix(&pts, &ctrl, 5, 0.5);
    let mut rng = Rng::new(1);
    let x: Vec<f64> = (0..ctrl.len()).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
    let mut ax = vec![0.0; pts.len()];
    let mut aty = vec![0.0; ctrl.len()];
    a.apply(&x, &mut ax);
    a.apply_t(&y, &mut aty);
    let lhs: f64 = ax.iter().zip(&y).map(|(a, b)| a * b).sum();
    let rhs: f64 = aty.iter().zip(&x).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0));
    // rows are convex combinations
    for i in 0..a.nrows() {
        let s: f64 = (a.offsets[i]..a.offsets[i + 1]).map(|e| a.vals[e]).sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}

#[test]
fn control_points_cover_and_separate() {
    let pts = grid_points(40, 0.1);
    let s = 0.45;
    let c = select_control_points(&pts, s, 9);
    for (a, &i) in c.iter().enumerate() {
        for &j in &c[a + 1..] {
            assert!(crate::util::dist2(pts[i as usize], pts[j as usize]) > s * s);
        }
    }
    for p in &pts {
        assert!(c.iter().any(|&i| crate::util::dist2(pts[i as usize], *p) <= s * s));
    }
    assert_eq!(c, select_control_points(&pts, s, 9));
}

#[test]
fn change_mode_ignores_uniform_offsets_but_not_bumps() {
    let mesh = Mesh::uv_sphere(3.0, 48, 32);
    let normals: Vec<V3> = mesh.vertex_normals().iter().map(|v| scale(*v, -1.0)).collect();
    let op = CurvatureOperator::new(
        SurfaceLaplacian::cotangent(&mesh),
        &mesh.vertices,
        &normals,
        CurvatureMode::Change,
        1.0,
        [true, false],
    );
    let n = mesh.vertices.len();
    let mut st = op.new_state();
    let (mut gm, mut gt) = (vec![0.0; n], vec![0.0; n]);
    let flat = op.eval(&vec![0.7; n], &vec![0.0; n], &mut st, &mut gm, &mut gt);
    assert!(flat < 1e-20, "uniform offset cost {flat}");
    let bumpy: Vec<f64> = mesh.vertices.iter().map(|v| 0.7 + 0.1 * (4.0 * v[0]).sin()).collect();
    let c = op.eval(&bumpy, &vec![0.0; n], &mut st, &mut gm, &mut gt);
    assert!(c > 1e-3);
    // the excess mode sees the curvature 1/R of the sphere itself
    for lap in [
        SurfaceLaplacian::cotangent(&mesh),
        SurfaceLaplacian::gaussian(&mesh.vertices, 40, 0.4, 3.0),
    ] {
        let ex = CurvatureOperator::new(
            lap,
            &mesh.vertices,
            &normals,
            CurvatureMode::Excess { kappa: 0.2 },
            1.0,
            [true, false],
        );
        let h = ex.mean_curvature(&vec![0.0; n]);
        let mid = h
            .iter()
            .zip(&mesh.vertices)
            .filter(|(_, v)| v[2].abs() < 2.0)
            .map(|(h, _)| *h);
        for hv in mid {
            assert!((hv * 3.0 - 1.0).abs() < 0.08, "H*R = {}", hv * 3.0);
        }
    }
    // the Gaussian change operator also ignores uniform offsets
    let g = CurvatureOperator::new(
        SurfaceLaplacian::gaussian(&mesh.vertices, 40, 0.4, 3.0),
        &mesh.vertices,
        &normals,
        CurvatureMode::Change,
        1.0,
        [true, false],
    );
    assert!(g.eval(&vec![0.7; n], &vec![0.0; n], &mut st, &mut gm, &mut gt) < 1e-20);
}

#[test]
fn smoothing_operator_annihilates_constants() {
    let pts = grid_points(20, 0.1);
    let w = SmoothingOperator::new(&pts, 12, 0.2, f64::INFINITY);
    let mut out = vec![0.0; pts.len()];
    w.apply(&vec![3.0; pts.len()], &mut out);
    assert!(out.iter().all(|v| v.abs() < 1e-6));
    // transposed product is the adjoint
    let mut rng = Rng::new(5);
    let x: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
    let mut wx = vec![0.0; pts.len()];
    let mut wty = vec![0.0; pts.len()];
    w.apply(&x, &mut wx);
    w.apply_t(&y, &mut wty);
    let l: f64 = wx.iter().zip(&y).map(|(a, b)| a * b).sum();
    let r: f64 = wty.iter().zip(&x).map(|(a, b)| a * b).sum();
    assert!((l - r).abs() < 1e-5 * l.abs().max(1.0));
    // radius cut-off keeps at least one neighbour
    let w2 = SmoothingOperator::new(&pts, 12, 0.01, 1.0);
    assert!((0..pts.len()).all(|i| w2.rows.offsets[i + 1] > w2.rows.offsets[i]));
}
