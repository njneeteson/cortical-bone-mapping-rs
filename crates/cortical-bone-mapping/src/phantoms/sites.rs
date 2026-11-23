//! Site phantoms: the anatomical sites CBM is applied to, at the resolution
//! they are scanned at, with parameters from the literature.
//!
//! | site | scanner | voxels (mm) | PSF σ (mm, axial / in-plane) | noise |
//! |---|---|---|---|---|
//! | distal radius | HR-pQCT (XtremeCT II) | 0.0607 | 0.08 | 40 |
//! | distal tibia | HR-pQCT (XtremeCT II) | 0.0607 | 0.08 | 40 |
//! | proximal femur | clinical CT (hip) | 1.0 × 0.75 × 0.75 | 0.6 / 0.5 | 30 HU |
//! | knee | clinical CT | 0.625 × 0.65 × 0.65 | 0.45 / 0.4 | 22 HU |
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
//! * **Knee** ([`JointPhantom`]): the voxel size and the soft-tissue level
//!   and noise (57 and 22 HU) of our clinical knee CT
//!   (`docs/clinical_ct.md`); a tibial plateau 74 mm across; subchondral
//!   plates 0.78 mm (tibia) and 0.32 mm (femur) and a joint space of 5–6 mm,
//!   as measured in healthy young adults by weight-bearing CT with 0.3 mm
//!   voxels (Waungana et al. 2025); an intercondylar notch 20 mm wide;
//!   trabecular densities of about 330 HU, rising to about 540 HU under the
//!   plates (70–500 HU on the clinical knee).
//! * **Clinical PSF.** A Gaussian with a 10% modulation transfer at
//!   frequency `f` has `σ = √(ln 10 / 2)/(π f)`. With standard kernels the
//!   10% MTF of clinical scanners is 6.98 ± 0.40 lp/cm (five scanners; Li et
//!   al. 2022), i.e. σ ≈ 0.49 mm in-plane; bone kernels are sharper, so we
//!   take 0.4–0.5 mm. Along the axis the slice sensitivity profile of a
//!   helical scan is somewhat wider than the nominal slice. The voxel
//!   integration adds `h²/12` to each variance.
//!
//! The values not given a source (neck and shaft radii, the trochanter and
//! acetabulum, notch, side-shell thickness, curvatures) are typical, not
//! measured.
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
//! * T. H. Waungana, R. Mangat, C. Chen, A. C. J. Smith, Y. Al-Khoury, M. T.
//!   Kuczynski, T. Turmezei, D. D. Anderson, W. B. Edwards, S. L. Manske,
//!   *Repeatability of objective bone and joint measures in the knee using
//!   weight-bearing computed tomography: three-dimensional quantification of
//!   bone mineral density, joint space width, and subchondral bone plate
//!   thickness*, JBMR Plus 10(2):ziaf194, 2026, doi:10.1093/jbmrpl/ziaf194.
//! * Y. Li, Y. Jiang, H. Liu, X. Yu, S. Chen, D. Ma, J. Gao, Y. Wu, *A
//!   phantom study comparing low-dose CT physical image quality from five
//!   different CT scanners*, Quantitative Imaging in Medicine and Surgery
//!   12(1):766–780, 2022, doi:10.21037/qims-21-245.

use super::bone::{BonePhantom, NeighbourBone};
use super::femur::FemurPhantom;
use super::joint::{JointPhantom, Notch};

/// An anatomical site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    DistalRadius,
    DistalTibia,
    ProximalFemur,
    Knee,
}

/// A site's phantom.
#[derive(Clone, Debug)]
pub enum SitePhantom {
    Bone(BonePhantom),
    Femur(FemurPhantom),
    Joint(JointPhantom),
}

impl Site {
    pub const ALL: [Site; 4] = [Site::DistalRadius, Site::DistalTibia, Site::ProximalFemur, Site::Knee];

    pub fn name(self) -> &'static str {
        match self {
            Site::DistalRadius => "distal_radius",
            Site::DistalTibia => "distal_tibia",
            Site::ProximalFemur => "proximal_femur",
            Site::Knee => "knee",
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
            Site::Knee => SitePhantom::Joint(knee(seed)),
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

fn knee(seed: u64) -> JointPhantom {
    JointPhantom {
        dims: (113, 144, 144),
        spacing: [0.625, 0.65, 0.65],
        radius: 37.0,
        femur_scale: 1.05,
        joint_level: 0.43,
        joint_space: 5.0,
        contact_offset: [2.0, 0.0],
        tibia_curvature: 120.0,
        femur_curvature: 60.0,
        bump_amplitude: 0.4,
        bump_wavelength: 20.0,
        plate_thickness: 0.78,
        femur_plate_thickness: Some(0.32),
        plate_variation: 0.3,
        plate_min: 0.15,
        shell_thickness: 0.8,
        rim_rounding: 3.0,
        bvtv: 0.22,
        endo_bvtv: 0.35,
        transition_width: 3.0,
        trabecular_wavelength: 1.6,
        porosity: 0.03,
        rho_soft: 57.0,
        rho_cartilage: 80.0,
        rho_bone: 1600.0,
        rho_marrow: -30.0,
        blur: [0.45, 0.4, 0.4],
        noise: 22.0,
        seed,
        supersample: 3,
        notch: Some(Notch {
            width: 20.0,
            height: 22.0,
            anterior: -5.0,
        }),
        ..JointPhantom::default()
    }
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
        assert_eq!(Site::from_name("knee"), Some(Site::Knee));
    }

    #[test]
    fn knee_joint_space_is_healthy() {
        let SitePhantom::Joint(k) = Site::Knee.phantom(1) else {
            panic!()
        };
        // at the notch edges and at the condyle centres (±19 mm): 5–6.5 mm
        for x in [10.0, 19.0] {
            let q2 = x * x + k.contact_offset[0].powi(2);
            let gap = k.joint_space + q2 * (1.0 / (2.0 * k.femur_curvature) - 1.0 / (2.0 * k.tibia_curvature));
            assert!(gap > 5.0 && gap < 6.6, "gap {gap} at {x}");
        }
    }
}
