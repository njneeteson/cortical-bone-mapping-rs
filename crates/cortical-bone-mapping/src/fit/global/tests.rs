use super::*;
use crate::model::{eval, residual_weights, Params, M, RHO_B, RHO_S, SIGMA, T};
use crate::operators::{interpolation_matrix, select_control_points, CurvatureMode};
use crate::util::{Rng, V3};

fn synth(n: usize, seed: u64, noise: f64) -> (Profiles, Vec<f64>, Vec<Params>, Vec<V3>) {
    let x: Vec<f64> = (0..61).map(|j| -1.0 + j as f64 * 0.05).collect();
    let mut rng = Rng::new(seed);
    let mut values = Vec::new();
    let mut truth = Vec::new();
    let mut pts = Vec::new();
    for i in 0..n {
        let s = i as f64 / n as f64;
        let p = [0.5 + 0.1 * (6.0 * s).sin(), 0.6 + 0.3 * s, 0.15, 50.0, 300.0];
        for &xj in &x {
            values.push((eval(xj, &p, 1000.0) + noise * rng.normal()) as f32);
        }
        truth.push(p);
        pts.push([s * 10.0, 0.0, 0.0]);
    }
    (
        Profiles {
            n,
            m: x.len(),
            values,
            used: None,
        },
        x,
        truth,
        pts,
    )
}

fn bounds() -> (Params, Params) {
    ([-3.0, 0.05, 0.01, -200.0, -200.0], [6.0, 12.0, 2.0, 600.0, 900.0])
}

#[test]
fn all_global_fit_recovers_mean_model() {
    let (prof, x, _, _) = synth(200, 1, 0.0);
    let w = residual_weights(&x, 1.0);
    let (lo, hi) = bounds();
    let pr = Problem {
        profiles: &prof,
        x: &x,
        weights: &w,
        rho_c: 1000.0,
        roles: [Role::Global; 5],
        fixed_per_node: [None; 5],
        interp: None,
        frozen: None,
        smoothing: None,
        curvature: None,
        lower: lo,
        upper: hi,
    };
    let mut st = State {
        field: vec![],
        scalars: [0.0, 1.0, 0.1, 0.0, 0.0],
    };
    let info = solve(&pr, &mut st, &Settings::default());
    assert!(info.status.converged(), "{info:?}");
    assert!((st.scalars[RHO_S] - 50.0).abs() < 5.0, "{:?}", st.scalars);
    assert!((st.scalars[RHO_B] - 300.0).abs() < 10.0);
    assert!((st.scalars[T] - 0.75).abs() < 0.1);
}

#[test]
fn field_fit_recovers_per_node_parameters() {
    let (prof, x, truth, _) = synth(300, 2, 0.0);
    let w = residual_weights(&x, 2.0);
    let (lo, hi) = bounds();
    let pr = Problem {
        profiles: &prof,
        x: &x,
        weights: &w,
        rho_c: 1000.0,
        roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
        fixed_per_node: [None; 5],
        interp: None,
        frozen: None,
        smoothing: None,
        curvature: None,
        lower: lo,
        upper: hi,
    };
    let mut st = State {
        field: vec![vec![0.4; 300], vec![0.8; 300], vec![0.2; 300]],
        scalars: [0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let info = solve(
        &pr,
        &mut st,
        &Settings {
            max_iterations: 200,
            ftol: 1e-12,
            gtol: 1e-10,
            ..Default::default()
        },
    );
    assert!(info.status.converged(), "{info:?}");
    for i in 0..300 {
        assert!(
            (st.field[1][i] - truth[i][T]).abs() < 2e-3,
            "node {i}: {} vs {}",
            st.field[1][i],
            truth[i][T]
        );
        assert!((st.field[0][i] - truth[i][M]).abs() < 2e-3);
    }
    assert!((st.scalars[RHO_S] - 50.0).abs() < 0.5);
    let _ = SIGMA;
}

#[test]
fn gradient_matches_finite_differences_with_all_terms() {
    // tiny curved surface patch (UV sphere) so curvature is meaningful
    let mesh = crate::surface::mesh::Mesh::uv_sphere(2.0, 10, 8);
    let n = mesh.vertices.len();
    let normals: Vec<V3> = mesh
        .vertex_normals()
        .iter()
        .map(|v| crate::util::scale(*v, -1.0))
        .collect();
    let x: Vec<f64> = (0..31).map(|j| -1.0 + j as f64 * 0.1).collect();
    let mut rng = Rng::new(3);
    let values: Vec<f32> = (0..n * x.len())
        .map(|_| (300.0 + 200.0 * rng.normal()) as f32)
        .collect();
    let prof = Profiles {
        n,
        m: x.len(),
        values,
        used: None,
    };
    let w = residual_weights(&x, 3.0);
    let near = crate::operators::Neighbours::euclidean(&mesh.vertices);
    let sm = SmoothingOperator::new(&near, 6, 0.8, f64::INFINITY);
    for mode in [CurvatureMode::Change, CurvatureMode::Excess { kappa: 0.3 }] {
        for lap_kind in 0..2 {
            let lap = if lap_kind == 0 {
                crate::operators::SurfaceLaplacian::cotangent(&mesh)
            } else {
                crate::operators::SurfaceLaplacian::gaussian(&near, 8, 0.8, 3.0)
            };
            let cv = CurvatureOperator::new(lap, &mesh.vertices, &normals, mode, 5.0, [true, true]);
            for use_interp in [false, true] {
                let ctrl = select_control_points(&near, 1.0, 1);
                let a = interpolation_matrix(&near, &ctrl, 4, 1.0);
                let ndof = if use_interp { ctrl.len() } else { n };
                let (lo, hi) = bounds();
                let pr = Problem {
                    profiles: &prof,
                    x: &x,
                    weights: &w,
                    rho_c: 1000.0,
                    roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
                    fixed_per_node: [None; 5],
                    interp: use_interp.then_some(&a),
                    frozen: None,
                    smoothing: Some(Smoothing {
                        op: &sm,
                        coef: [3.0, 2.0, 1.0, 0.0, 0.0],
                    }),
                    curvature: Some(&cv),
                    lower: lo,
                    upper: hi,
                };
                let st = State {
                    field: vec![
                        (0..ndof).map(|_| 0.3 + 0.2 * rng.normal()).collect(),
                        (0..ndof).map(|_| 0.8 + 0.2 * rng.normal()).collect(),
                        (0..ndof).map(|_| 0.3 + 0.05 * rng.uniform()).collect(),
                    ],
                    scalars: [0.0, 0.0, 0.0, 60.0, 320.0],
                };
                let g = gradient(&pr, &st);
                let lay = pr.layout();
                for v in (0..lay.nvars()).step_by(7) {
                    let h = 1e-6;
                    let mut sp = st.clone();
                    let mut sm_ = st.clone();
                    let bump = |s: &mut State, d: f64| {
                        if v < lay.fields.len() * ndof {
                            s.field[v / ndof][v % ndof] += d;
                        } else {
                            s.scalars[lay.globals[v - lay.fields.len() * ndof]] += d;
                        }
                    };
                    bump(&mut sp, h);
                    bump(&mut sm_, -h);
                    let fd = (cost(&pr, &sp) - cost(&pr, &sm_)) / (2.0 * h);
                    assert!(
                        (fd - g[v]).abs() <= 1e-4 * (1.0 + fd.abs()),
                        "{mode:?} interp={use_interp} var {v}: fd {fd} vs {}",
                        g[v]
                    );
                }
            }
        }
    }
}

#[test]
fn smoothing_reduces_noise_in_thickness() {
    let (prof, x, truth, pts) = synth(400, 4, 60.0);
    let w = residual_weights(&x, 2.0);
    let (lo, hi) = bounds();
    let sm = SmoothingOperator::new(&crate::operators::Neighbours::euclidean(&pts), 10, 0.2, 3.0);
    let run = |lambda: f64| {
        let coef = 1000.0f64.powi(2) / (400.0 * 0.8f64.powi(2)) * lambda;
        let pr = Problem {
            profiles: &prof,
            x: &x,
            weights: &w,
            rho_c: 1000.0,
            roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
            fixed_per_node: [None; 5],
            interp: None,
            frozen: None,
            smoothing: Some(Smoothing {
                op: &sm,
                coef: [coef, coef, coef * 16.0, 0.0, 0.0],
            }),
            curvature: None,
            lower: lo,
            upper: hi,
        };
        let mut st = State {
            field: vec![vec![0.5; 400], vec![0.75; 400], vec![0.15; 400]],
            scalars: [0.0, 0.0, 0.0, 50.0, 300.0],
        };
        solve(&pr, &mut st, &Settings::default());
        let err: f64 = (0..400).map(|i| (st.field[1][i] - truth[i][T]).powi(2)).sum::<f64>() / 400.0;
        err.sqrt()
    };
    let e0 = run(0.0);
    let e1 = run(1.0);
    assert!(e1 < 0.7 * e0, "rmse without {e0}, with {e1}");
}
