use ndarray::Array3;

use super::band::{Band, BandSpec};
use super::problem::{Prepared, VolumetricProblem, M, NV, RHO_B, RHO_S, SXY, SZ, T};
use crate::fit::global::{LeastSquares, Role};
use crate::operators::{CurvatureMode, CurvatureOperator, Distance, Neighbours, SmoothingOperator, SurfaceLaplacian};
use crate::surface::mesh::Mesh;
use crate::util::{add, scale, Rng, V3};
use crate::volume::Geometry;

/// A sphere of radius 4 in a 20³ image at 0.5 spacing, its band, and its
/// operators.
struct Setup {
    mesh: Mesh,
    band: Band,
    smoothing: SmoothingOperator,
    curvature: CurvatureOperator,
}

fn setup() -> Setup {
    let mut mesh = Mesh::uv_sphere(4.0, 24, 16);
    for v in &mut mesh.vertices {
        *v = add(*v, [5.0; 3]);
    }
    let geom = Geometry::new([0.5; 3]);
    let img = Array3::<f32>::zeros((20, 20, 20));
    let fitted: Vec<Option<u32>> = (0..mesh.faces.len() as u32).map(Some).collect();
    let spec = BandSpec {
        geom: &geom,
        outside: 1.0,
        inside: 1.5,
        boost: 3.0,
        axis: 0,
        exclude: None,
        stride: 1,
    };
    let band = Band::build(&img.view(), &mesh, &fitted, mesh.faces.len(), &spec);
    let near = Neighbours::new(&mesh, Distance::Surface);
    let smoothing = SmoothingOperator::new(&near, 8, 1.0, 3.0);
    let normals: Vec<V3> = mesh.vertex_normals().iter().map(|n| scale(*n, -1.0)).collect();
    let lap = SurfaceLaplacian::gaussian(&near, 12, 1.0, 3.0);
    let curvature = CurvatureOperator::new(lap, &mesh.vertices, &normals, CurvatureMode::Change, 3.0, [true, true]);
    Setup {
        mesh,
        band,
        smoothing,
        curvature,
    }
}

fn problem<'a>(s: &'a Setup, roles: [Role; NV], tied: bool) -> VolumetricProblem<'a> {
    let mut fixed = [0.0; NV];
    fixed[RHO_S] = 40.0;
    fixed[RHO_B] = 200.0;
    fixed[SZ] = 0.4;
    VolumetricProblem {
        band: &s.band,
        faces: &s.mesh.faces,
        nodes: s.mesh.vertices.len(),
        rho_c: 1000.0,
        roles,
        tied,
        fixed,
        frozen: None,
        smoothing: Some((&s.smoothing, [2e-4, 3e-4])),
        curvature: Some(&s.curvature),
        lower: [-3.0, 0.05, -500.0, -500.0, 0.05, 0.05],
        upper: [3.0, 3.0, 1000.0, 1000.0, 2.0, 2.0],
    }
}

/// A point of the variables: smooth fields with some structure.
fn point(p: &Prepared, rng: &mut Rng, jitter: f64) -> Vec<f64> {
    let lay = &p.lay;
    let n = lay.nodes;
    let mut x = Vec::new();
    for &c in &lay.fields {
        let base = match c {
            M => 0.4,
            T => 0.8,
            RHO_S => 50.0,
            _ => 250.0,
        };
        let amp = if c >= RHO_S { 20.0 } else { 0.1 };
        x.extend((0..n).map(|v| base + amp * ((v as f64) * 0.37).sin() + jitter * amp * rng.normal()));
    }
    for &c in &lay.globals {
        let base = match c {
            M => 0.4,
            T => 0.8,
            RHO_S => 50.0,
            RHO_B => 250.0,
            SXY => 0.3,
            _ => 0.45,
        };
        x.push(base * (1.0 + 0.1 * jitter * rng.normal()));
    }
    x
}

/// Replace the band's values by the model's at `x` (noise-free data).
fn synthesise(s: &mut Setup, roles: [Role; NV], tied: bool, x: &[f64]) {
    let p = problem(s, roles, tied);
    let prep = Prepared::new(&p);
    let values = prep.model_values(x);
    s.band.value = values.iter().map(|&v| v as f32).collect();
}

const ROLE_SETS: [([Role; NV], bool); 4] = [
    // fields m, t, ρs, ρb; fitted PSF
    (
        [
            Role::Field,
            Role::Field,
            Role::Field,
            Role::Field,
            Role::Global,
            Role::Global,
        ],
        false,
    ),
    // global densities, tied PSF
    (
        [
            Role::Field,
            Role::Field,
            Role::Global,
            Role::Global,
            Role::Global,
            Role::Global,
        ],
        true,
    ),
    // fixed densities and σ_z
    (
        [
            Role::Field,
            Role::Field,
            Role::Fixed,
            Role::Fixed,
            Role::Global,
            Role::Fixed,
        ],
        false,
    ),
    // the global initialisation
    (
        [
            Role::Global,
            Role::Global,
            Role::Global,
            Role::Global,
            Role::Global,
            Role::Global,
        ],
        false,
    ),
];

#[test]
fn gradient_matches_finite_differences() {
    let mut s = setup();
    let mut rng = Rng::new(1);
    for (roles, tied) in ROLE_SETS {
        let truth = {
            let p = problem(&s, roles, tied);
            point(&Prepared::new(&p), &mut rng, 0.0)
        };
        synthesise(&mut s, roles, tied, &truth);
        let p = problem(&s, roles, tied);
        let prep = Prepared::new(&p);
        let x = point(&prep, &mut rng, 0.3); // away from the data: residuals ≠ 0
        let (_, g, _) = prep.linearise(&x);
        let nv = x.len();
        for k in (0..nv).step_by((nv / 40).max(1)).chain(nv.saturating_sub(2)..nv) {
            let h = 1e-6 * x[k].abs().max(1.0);
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            let fd = (prep.cost(&xp) - prep.cost(&xm)) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() <= 1e-5 * (fd.abs().max(g[k].abs()) + 1e-6),
                "roles {roles:?} variable {k}: fd {fd} analytic {}",
                g[k]
            );
        }
    }
}

#[test]
fn gauss_newton_product_is_the_hessian_on_exact_data() {
    let mut s = setup();
    let mut rng = Rng::new(2);
    for (roles, tied) in ROLE_SETS {
        let truth = {
            let p = problem(&s, roles, tied);
            point(&Prepared::new(&p), &mut rng, 0.0)
        };
        synthesise(&mut s, roles, tied, &truth);
        let p = problem(&s, roles, tied);
        let prep = Prepared::new(&p);
        let (_, _, lin) = prep.linearise(&truth);
        let v: Vec<f64> = (0..truth.len())
            .map(|k| rng.normal() * truth[k].abs().max(0.1) * 1e-3)
            .collect();
        let mut hv = vec![0.0; truth.len()];
        prep.hess_vec(&lin, &v, &mut hv);
        let eps = 1e-3;
        let xp: Vec<f64> = truth.iter().zip(&v).map(|(a, b)| a + eps * b).collect();
        let xm: Vec<f64> = truth.iter().zip(&v).map(|(a, b)| a - eps * b).collect();
        let (gp, gm) = (prep.linearise(&xp).1, prep.linearise(&xm).1);
        let scale_ = hv.iter().map(|a| a.abs()).fold(0.0, f64::max);
        for k in 0..truth.len() {
            let fd = (gp[k] - gm[k]) / (2.0 * eps);
            assert!(
                (fd - hv[k]).abs() <= 1e-4 * scale_ + 1e-12,
                "roles {roles:?} variable {k}: fd {fd} product {}",
                hv[k]
            );
        }
    }
}

#[test]
fn preconditioner_blocks_are_the_product_s_diagonal_blocks() {
    let s = setup();
    let (roles, tied) = ROLE_SETS[0];
    let mut p = problem(&s, roles, tied);
    p.smoothing = None;
    p.curvature = None;
    let prep = Prepared::new(&p);
    let mut rng = Rng::new(3);
    let x = point(&prep, &mut rng, 0.2);
    let (_, _, lin) = prep.linearise(&x);
    let (field, glob) = prep.diag_blocks(&lin);
    let (nfc, n, ng) = prep.shape();
    let nv = x.len();
    let column = |k: usize| {
        let mut e = vec![0.0; nv];
        e[k] = 1.0;
        let mut out = vec![0.0; nv];
        prep.hess_vec(&lin, &e, &mut out);
        out
    };
    for v in [0, n / 3, n - 1] {
        for a in 0..nfc {
            let col = column(a * n + v);
            for b in 0..nfc {
                let blk = field[v * nfc * nfc + b * nfc + a];
                assert!(
                    (col[b * n + v] - blk).abs() <= 1e-12 * blk.abs().max(1e-12),
                    "node {v} ({b}, {a})"
                );
            }
        }
    }
    for a in 0..ng {
        let col = column(nfc * n + a);
        for b in 0..ng {
            assert!((col[nfc * n + b] - glob[b * ng + a]).abs() <= 1e-12 * glob[b * ng + a].abs().max(1e-12));
        }
    }
}
