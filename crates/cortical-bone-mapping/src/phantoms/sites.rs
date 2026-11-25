//! Site phantoms: the anatomical sites CBM is applied to, at the resolution
//! they are scanned at, with parameters from the literature.
//!
//! | site | scanner | voxels (mm) | PSF σ (mm, axial / in-plane) | noise |
//! |---|---|---|---|---|
//! | distal radius | HR-pQCT (XtremeCT II) | 0.0607 | 0.08 | 40 |
//! | distal tibia | HR-pQCT (XtremeCT II) | 0.0607 | 0.08 | 40 |
//! | proximal femur | clinical CT (hip) | 1.0 × 0.75 × 0.75 | 0.6 / 0.5 | 30 HU |
//!
//! * **HR-pQCT** ([`BonePhantom`]): densities in mg HA/cm³. The distal
//!   radius has Tt.Ar 220–360 mm², Ct.Th 0.77–0.93 mm and Ct.Po 1.5–2.5%;
//!   the distal tibia Tt.Ar 590–850 mm², Ct.Th 1.0–1.7 mm and Ct.Po
//!   3.5–6.6% (Burghardt et al. 2010); trabecular BV/TV is 0.12–0.33 and
//!   Tb.Sp 0.6–1.0 mm. The ulna and fibula lie beside them across 2–6 mm of
//!   soft tissue. The slab is 32 slices (1.9 mm), a fifth of a standard
//!   scan, to keep the tests and experiments small.
//! * **Proximal femur** ([`FemurPhantom`], see its module): a hip CT with
//!   0.75 mm pixels and 1 mm slices (the AGES-Reykjavik hip scans had
//!   0.98 mm pixels and 1 mm slices; Johannesdottir et al. 2011).
//! * **Clinical PSF.** A Gaussian with a 10% modulation transfer at
//!   frequency `f` has `σ = √(ln 10 / 2)/(π f)`. With standard kernels the
//!   10% MTF of clinical scanners is 6.98 ± 0.40 lp/cm (five scanners; Li et
//!   al. 2022), i.e. σ ≈ 0.49 mm in-plane; bone kernels are sharper, so we
//!   take 0.4–0.5 mm. Along the axis the slice sensitivity profile of a
//!   helical scan is somewhat wider than the nominal slice. The voxel
//!   integration adds `h²/12` to each variance.
//!
//! The values not given a source (neck and shaft radii, the trochanter and
//! acetabulum) are typical, not measured.
//!
//! # References
//!
//! * A. J. Burghardt, H. R. Buie, A. Laib, S. Majumdar, S. K. Boyd,
//!   *Reproducibility of direct quantitative measures of cortical bone
//!   microarchitecture of the distal radius and tibia by HR-pQCT*, Bone
//!   47(3):519–528, 2010.
//! * F. Johannesdottir et al., *Distribution of cortical bone in the femoral
//!   neck and hip fracture: a prospective case-control analysis of 143
//!   incident hip fractures; the AGES-REYKJAVIK Study*, Bone
//!   48(6):1268–1276, 2011.
//! * Y. Li, Y. Jiang, H. Liu, X. Yu, S. Chen, D. Ma, J. Gao, Y. Wu, *A
//!   phantom study comparing low-dose CT physical image quality from five
//!   different CT scanners*, Quantitative Imaging in Medicine and Surgery
//!   12(1):766–780, 2022, doi:10.21037/qims-21-245.

use super::bone::{BonePhantom, NeighbourBone};
use super::femur::FemurPhantom;

/// An anatomical site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    DistalRadius,
    DistalTibia,
    ProximalFemur,
}

/// A site's phantom.
#[derive(Clone, Debug)]
pub enum SitePhantom {
    Bone(BonePhantom),
    Femur(FemurPhantom),
}

impl Site {
    pub const ALL: [Site; 3] = [Site::DistalRadius, Site::DistalTibia, Site::ProximalFemur];

    pub fn name(self) -> &'static str {
        match self {
            Site::DistalRadius => "distal_radius",
            Site::DistalTibia => "distal_tibia",
            Site::ProximalFemur => "proximal_femur",
        }
    }

    pub fn from_name(name: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The site's phantom with noise realisation (and random geometry) `seed`.
    pub fn phantom(self, seed: u64) -> SitePhantom {
        match self {
            Site::DistalRadius => SitePhantom::Bone(distal_radius(seed)),
            Site::DistalTibia => SitePhantom::Bone(distal_tibia(seed)),
            Site::ProximalFemur => SitePhantom::Femur(FemurPhantom {
                seed,
                ..FemurPhantom::default()
            }),
        }
    }
}

/// A shaft of mean periosteal radius `radius` and cortical thickness
/// `thickness`, with its neighbour across `gap` along array axis 2, in a
/// 32-slice HR-pQCT slab wide enough for the shaft and the near half of the
/// neighbour.
fn hrpqct_shaft(
    radius: f64,
    thickness: f64,
    porosity: f64,
    bvtv: f64,
    neighbour: NeighbourBone,
    seed: u64,
) -> BonePhantom {
    let h = 0.0607;
    let reach = 1.25 * radius + 1.0;
    let side = radius + neighbour.gap + neighbour.radius + 1.0;
    let n = |half: f64| (2.0 * half / h).ceil() as usize + 1;
    BonePhantom {
        dims: (32, n(reach), n(side)),
        spacing: [h; 3],
        radius,
        thickness_mean: thickness,
        thickness_min: 0.2,
        porosity,
        bvtv,
        neighbour: Some(neighbour),
        seed,
        ..BonePhantom::default()
    }
}

fn distal_radius(seed: u64) -> BonePhantom {
    let ulna = NeighbourBone {
        radius: 6.5,
        thickness: 1.0,
        gap: 3.0,
        angle: std::f64::consts::FRAC_PI_2,
    };
    hrpqct_shaft(9.0, 0.85, 0.02, 0.2, ulna, seed)
}

fn distal_tibia(seed: u64) -> BonePhantom {
    let fibula = NeighbourBone {
        radius: 7.0,
        thickness: 1.0,
        gap: 3.0,
        angle: std::f64::consts::FRAC_PI_2,
    };
    hrpqct_shaft(15.0, 1.3, 0.05, 0.25, fibula, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hrpqct_sizes_follow_the_literature() {
        for (site, area, thick) in [
            (Site::DistalRadius, (220.0, 360.0), (0.77, 0.93)),
            (Site::DistalTibia, (590.0, 850.0), (1.0, 1.7)),
        ] {
            let SitePhantom::Bone(b) = site.phantom(1) else {
                panic!()
            };
            let tt_ar = std::f64::consts::PI * b.radius * b.radius;
            assert!(tt_ar > area.0 && tt_ar < area.1, "{site:?} Tt.Ar {tt_ar}");
            assert!(b.thickness_mean > thick.0 && b.thickness_mean < thick.1);
            // the volume holds the shaft, and the neighbour's near side
            let w = (b.dims.2 - 1) as f64 * b.spacing[2];
            let nb = b.neighbour.as_ref().unwrap();
            assert!(0.5 * w > b.radius + nb.gap + nb.radius);
        }
        assert_eq!(Site::from_name("proximal_femur"), Some(Site::ProximalFemur));
    }
}
