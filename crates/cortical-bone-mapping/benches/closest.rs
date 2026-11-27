//! Benchmarks of the closest-point queries that build the volumetric band:
//! the 4-nearest-vertex search alone and the whole query (the search plus
//! the faces around those vertices), on the surface of the distal radius
//! phantom (HR-pQCT voxels), for the voxels of a few slices within the
//! band's reach, in the band's (row-major) order. Timing only.
//!
//!     cargo bench -p cortical-bone-mapping --features bench --bench closest

use std::num::NonZeroUsize;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use kiddo::{ImmutableKdTree, SquaredEuclidean};

use cortical_bone_mapping::options::Site as Preset;
use cortical_bone_mapping::phantoms::sites::{Site, SitePhantom};
use cortical_bone_mapping::pipeline::prepare_surface;
use cortical_bone_mapping::surface::closest::MeshPoints;
use cortical_bone_mapping::util::dist2;
use cortical_bone_mapping::Options;

/// Slices of queries (every voxel of the mesh's box in them).
const SLICES: usize = 4;
/// How far the band reaches (mm), as in the HR-pQCT runs.
const REACH: f64 = 5.0;

fn closest(c: &mut Criterion) {
    let SitePhantom::Bone(b) = Site::DistalRadius.phantom(1) else {
        unreachable!("the distal radius is a shaft phantom")
    };
    let (_, mask, _) = b.generate();
    let o = Options::preset(Preset::Knee, b.spacing[0]);
    let (mesh, _) = prepare_surface(&mask.view(), &o);
    let (d0, d1, d2) = mask.dim();
    let h = b.spacing;
    let mid = d0 / 2;
    let points = MeshPoints::new(&mesh);
    let queries: Vec<[f64; 3]> = (mid..mid + SLICES)
        .flat_map(|i| {
            (0..d1).flat_map(move |j| (0..d2).map(move |k| [i as f64 * h[0], j as f64 * h[1], k as f64 * h[2]]))
        })
        .filter(|&q| points.closest(q, REACH).is_some())
        .collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&mesh.vertices).expect("a tree");
    eprintln!("{} vertices, {} queries", mesh.vertices.len(), queries.len());
    let mut g = c.benchmark_group("closest");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    g.bench_function("whole", |bch| {
        bch.iter(|| queries.iter().filter(|&&q| points.closest(q, REACH).is_some()).count())
    });
    // along each row, each query bounded by the previous one's (as the band)
    g.bench_function("whole, row-bounded", |bch| {
        bch.iter(|| {
            let (mut bound, mut prev): (Option<f64>, [f64; 3]) = (None, [0.0; 3]);
            queries
                .iter()
                .filter(|&&q| {
                    let (c, far) = points.closest_bounded(q, REACH, bound.map(|b| b + dist2(q, prev).sqrt()));
                    (bound, prev) = (Some(far), q);
                    c.is_some()
                })
                .count()
        })
    });
    g.bench_function("nearest vertices", |bch| {
        bch.iter(|| queries.iter().map(|&q| points.nearest_vertices(q).len()).sum::<usize>())
    });
    g.bench_function("kiddo nearest_n", |bch| {
        bch.iter(|| {
            queries
                .iter()
                .map(|q| {
                    tree.query(q)
                        .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(6).unwrap())
                        .execute()
                        .len()
                })
                .sum::<usize>()
        })
    });
    g.finish();
}

criterion_group!(benches, closest);
criterion_main!(benches);
