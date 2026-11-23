"""Phantoms with exact ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from . import _core
from ._util import Triple, _triple
from .results import CbmResult


@dataclass
class Phantom:
    """A synthetic cylinder (along axis 0) or sphere with a known cortex.

    Outer radius ``R(theta) = radius + radius_bump cos(radius_lobes theta)``;
    thickness ``t(theta) = thickness_mean + thickness_amplitude
    cos(thickness_lobes theta)`` with ``theta`` the angle in the axis-1/axis-2
    plane (cylinder) or the polar angle from axis 0 (sphere). Densities are
    integrated over voxels (``supersample``), blurred with a Gaussian PSF
    ``blur`` (physical standard deviation per axis) and given Gaussian noise.
    The returned mask can be perturbed (``mask_offset``, ``mask_bump``,
    ``mask_lobes``) to emulate a poor segmentation.
    """

    dims: Tuple[int, int, int] = (24, 96, 96)
    spacing: Triple = (0.1, 0.1, 0.1)
    shape: Literal["cylinder", "sphere"] = "cylinder"
    radius: float = 3.5
    radius_bump: float = 0.0
    radius_lobes: float = 3.0
    thickness_mean: float = 0.8
    thickness_amplitude: float = 0.3
    thickness_lobes: float = 2.0
    rho_s: float = 50.0
    rho_c: float = 1000.0
    rho_b: float = 250.0
    blur: Triple = (0.12, 0.12, 0.12)
    noise: float = 20.0
    seed: int = 0
    supersample: int = 3
    mask_offset: float = 0.0
    mask_bump: float = 0.0
    mask_lobes: float = 5.0

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(image float32, mask uint8)``."""
        d = asdict(self)
        d["dims"] = tuple(int(x) for x in self.dims)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["blur"] = _triple(self.blur, "blur")
        return _core.phantom(d)

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.dims, float) - 1) * 0.5 * np.asarray(_triple(self.spacing, "spacing"))

    def polar(self, points) -> Tuple[np.ndarray, np.ndarray]:
        """Radial distance and angle of physical points."""
        d = np.atleast_2d(points) - self.center
        if self.shape == "cylinder":
            return np.hypot(d[:, 1], d[:, 2]), np.arctan2(d[:, 2], d[:, 1])
        r = np.linalg.norm(d, axis=1)
        return r, np.arccos(np.clip(d[:, 0] / np.maximum(r, 1e-300), -1, 1))

    def outer_radius(self, theta) -> np.ndarray:
        return self.radius + self.radius_bump * np.cos(self.radius_lobes * np.asarray(theta))

    def thickness(self, theta) -> np.ndarray:
        return self.thickness_mean + self.thickness_amplitude * np.cos(self.thickness_lobes * np.asarray(theta))

    def true_thickness(self, points) -> np.ndarray:
        return self.thickness(self.polar(points)[1])

    def radial_offset(self, points) -> np.ndarray:
        """Radial distance of points outside the true periosteal surface."""
        r, th = self.polar(points)
        return r - self.outer_radius(th)

    def true_mask(self) -> np.ndarray:
        """The unperturbed segmentation."""
        return replace(self, mask_offset=0.0, mask_bump=0.0, noise=0.0, supersample=1).generate()[1]


@dataclass
class BonePhantom:
    """A realistic long-bone phantom (shaft along axis 0) with known geometry.

    * periosteal cross-section: random star-shaped curve
      ``R(theta, z) = radius * (1 + sum_k a_k cos(k theta + phi_k + w_k z))``
      (``shape_harmonics`` terms from k = 2, first amplitude ~``shape_amplitude``,
      decaying as k^-1.5, phases drifting over ``axial_wavelength``), centre
      wandering by ``center_wander``;
    * radial cortical thickness: independent random harmonics
      (``thickness_variation``) plus an axial modulation, clipped at
      ``thickness_min``;
    * trabecular network in the marrow space (thin level-set shell of a
      random field, bone volume fraction ``bvtv``, pattern wavelength
      ``trabecular_wavelength``, elongated along the axis);
    * cortical pores (volume fraction ``porosity``);
    * an endocortical transition zone: porosity rising by ``endo_porosity``
      towards the endosteum and trabecular BV/TV rising to ``endo_bvtv``
      within ~``transition_width`` of it (the true endosteal surface stays the
      smooth ``r = R - t``);
    * voxel integration (``supersample``), Gaussian PSF ``blur`` (physical
      SD per axis) and Gaussian ``noise``.

    * optionally a ``neighbour`` bone (fibula / ulna) given as
      ``(radius, cortical thickness, gap, angle)``: circular, solid cortex,
      same trabecular pattern, placed so that the gap between the periosteal
      surfaces along direction ``angle`` (radians, 0 = array axis 1) is
      ``gap`` in every slice. Real neighbours are always separated by some
      soft tissue (typically 1-5 mm), so ``gap`` should be positive;
    * optionally cortical ``breaks``, each ``(angle, width, z_center, length)``:
      a channel of soft tissue through the full cortical depth, elliptical on
      the periosteal surface with axes ``width`` (along the arc, centred on
      ``angle``) and ``length`` (along the axis, centred on ``z_center``).
      Realistic HR-pQCT breaks are ~0.3-1.2 mm across. The true periosteal
      surface spans the break.

    Densities: ``rho_soft`` outside, ``rho_bone`` for cortical and
    trabecular tissue, ``rho_marrow`` in marrow and pores. The truth masks
    are of the main bone only; :meth:`labels` marks the neighbour and the
    breaks.
    """

    dims: Tuple[int, int, int] = (32, 192, 192)
    spacing: Triple = (0.0607, 0.0607, 0.0607)
    radius: float = 4.5
    shape_harmonics: int = 6
    shape_amplitude: float = 0.12
    axial_wavelength: float = 8.0
    center_wander: float = 0.2
    thickness_mean: float = 0.8
    thickness_harmonics: int = 6
    thickness_variation: float = 0.35
    thickness_min: float = 0.15
    bvtv: float = 0.2
    trabecular_wavelength: float = 1.0
    porosity: float = 0.03
    pore_wavelength: float = 0.35
    endo_porosity: float = 0.15
    endo_bvtv: float = 0.4
    transition_width: float = 0.3
    rho_soft: float = 50.0
    rho_bone: float = 1000.0
    rho_marrow: float = 30.0
    blur: Triple = (0.08, 0.08, 0.08)
    noise: float = 40.0
    seed: int = 1
    supersample: int = 2
    neighbour: Optional[Tuple[float, float, float, float]] = None
    breaks: Sequence[Tuple[float, float, float, float]] = ()

    def _spec(self) -> dict:
        d = asdict(self)
        d["dims"] = tuple(int(x) for x in self.dims)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["blur"] = _triple(self.blur, "blur")
        d["neighbour"] = None if self.neighbour is None else tuple(float(v) for v in self.neighbour)
        d["breaks"] = [tuple(float(v) for v in b) for b in self.breaks]
        return d

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(image float32, periosteal mask, endosteal mask)``; the true
        cortical mask is ``periosteal & ~endosteal``."""
        return _core.bone_phantom(self._spec())

    def labels(self) -> np.ndarray:
        """uint8 labels at voxel centres: 1 = neighbouring bone, 2 = cortical
        break (soft tissue inside the cortex shell), 0 otherwise."""
        return _core.bone_phantom_labels(self._spec())

    def edges_along(self, points, normals, x0: float, x1: float) -> np.ndarray:
        """True ``(x_periosteal, x_endosteal)`` crossings along each line
        ``p + x n`` for ``x`` in ``[x0, x1]`` (NaN where missed), shape (N, 2)."""
        p = np.ascontiguousarray(points, dtype=np.float64)
        n = np.ascontiguousarray(normals, dtype=np.float64)
        return _core.bone_phantom_edges(self._spec(), p, n, float(x0), float(x1))

    def true_thickness(self, result: "CbmResult") -> np.ndarray:
        """Reference thickness along each sampling line of a result."""
        o = result.options
        e = self.edges_along(result.points, result.normals, -o.outside, o.inside)
        return e[:, 1] - e[:, 0]


@dataclass
class JointPhantom:
    """A knee-joint phantom: the articular ends of the tibia (below) and the
    femur (above, along array axis 0) across a cartilage-filled joint space.

    Each bone is a star-shaped cylinder (footprint ``radius``, femur
    ``femur_scale`` larger) cut by its articular surface, with the rim
    rounded over ``rim_rounding``. The tibial plateau is a shallow dish
    (curvature radius ``tibia_curvature``) with random bumps; the femoral
    condyle is more curved (``femur_curvature``), so the joint space is
    ``joint_space`` at the contact point (``contact_offset`` from the axis)
    and widens towards the rim. The cortex is a subchondral plate
    (``plate_thickness``, random relative variation ``plate_variation``,
    minimum ``plate_min``) and a thin side shell (``shell_thickness``);
    trabecular BV/TV rises from ``bvtv`` to ``endo_bvtv`` towards the plate;
    the plate has pores. The joint space holds cartilage / fluid
    (``rho_cartilage``).

    Optionally the femoral plate is thinner (``femur_plate_thickness``),
    and the femur has an intercondylar ``notch`` given as
    ``(width, height, anterior)``: a slot ``width`` across (array axis 2,
    centred on the axis), its roof ``height`` above the femoral surface at
    the contact point, open posteriorly from ``anterior`` along axis 1, with
    cortex of ``shell_thickness`` on its walls.

    The default is a scaled-down joint (a real tibial plateau is ~70 mm
    wide): it reproduces the local geometry of the contact region in a small
    volume; ``site_phantom("knee")`` is a full-size knee at clinical CT
    resolution.
    The default joint space, 4 mm, is a healthy bone-to-bone distance (medial
    radiographic joint space width ~4.8 mm in women, ~5.7 mm in men); ~2 mm
    is moderate osteoarthritis.
    """

    dims: Tuple[int, int, int] = (180, 200, 200)
    spacing: Triple = (0.0607, 0.0607, 0.0607)
    radius: float = 5.0
    femur_scale: float = 1.05
    shape_harmonics: int = 5
    shape_amplitude: float = 0.08
    joint_level: float = 0.35
    joint_space: float = 4.0
    contact_offset: Tuple[float, float] = (0.8, -0.5)
    tibia_curvature: float = 20.0
    femur_curvature: float = 9.0
    bump_amplitude: float = 0.08
    bump_wavelength: float = 4.0
    plate_thickness: float = 0.45
    plate_variation: float = 0.3
    plate_min: float = 0.15
    shell_thickness: float = 0.3
    rim_rounding: float = 0.4
    bvtv: float = 0.22
    endo_bvtv: float = 0.4
    transition_width: float = 0.4
    trabecular_wavelength: float = 1.0
    porosity: float = 0.03
    pore_wavelength: float = 0.35
    rho_soft: float = 50.0
    rho_cartilage: float = 80.0
    rho_bone: float = 1000.0
    rho_marrow: float = 30.0
    blur: Triple = (0.08, 0.08, 0.08)
    noise: float = 40.0
    seed: int = 1
    supersample: int = 2
    femur_plate_thickness: Optional[float] = None
    notch: Optional[Tuple[float, float, float]] = None

    TIBIA = 1
    FEMUR = 2

    def _spec(self) -> dict:
        d = asdict(self)
        d["dims"] = tuple(int(x) for x in self.dims)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["blur"] = _triple(self.blur, "blur")
        d["contact_offset"] = tuple(float(v) for v in self.contact_offset)
        d["notch"] = None if self.notch is None else tuple(float(v) for v in self.notch)
        return d

    def joint_height(self) -> float:
        """Height of the tibial plateau at the contact point (physical, axis 0)."""
        return self.joint_level * (self.dims[0] - 1) * _triple(self.spacing, "spacing")[0]

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(image float32, periosteal labels, endosteal labels)``: uint8,
        0 outside, 1 tibia, 2 femur."""
        return _core.joint_phantom(self._spec())

    def edges_along(self, points, normals, x0: float, x1: float) -> np.ndarray:
        """True ``(x_periosteal, x_endosteal, bone)`` along each line ``p + x n``
        (first entry into either bone, then into its marrow space), shape (N, 3)."""
        p = np.ascontiguousarray(points, dtype=np.float64)
        n = np.ascontiguousarray(normals, dtype=np.float64)
        return _core.joint_phantom_edges(self._spec(), p, n, float(x0), float(x1))

    def true_thickness(self, result: "CbmResult", bone: Optional[int] = None) -> np.ndarray:
        """Reference thickness along each sampling line of a result (NaN where
        the line first enters a bone other than ``bone``, if given)."""
        o = result.options
        e = self.edges_along(result.points, result.normals, -o.outside, o.inside)
        t = e[:, 1] - e[:, 0]
        if bone is not None:
            t[e[:, 2] != bone] = np.nan
        return t


@dataclass
class FemurPhantom:
    """A proximal femur at clinical CT scale, with the acetabulum across the
    hip joint space (array axis 0 = scanner axis, superior).

    The periosteal surface is a smooth union (fillet width ``rounding``) of
    the shaft (``shaft_radius``), the neck (elliptical, ``neck_radii``
    superoinferior and anteroposterior, at ``neck_shaft_angle`` and
    ``anteversion`` degrees), the head (``head_radius``, its centre
    ``offset`` from the shaft axis) and the greater trochanter (an ellipsoid
    with semi-axes ``trochanter``). The cortex is ``neck_thickness``
    (superior, inferior) around the neck, ``head_thickness``,
    ``trochanter_thickness`` and ``shaft_thickness``, blended between the
    parts, varied by ``thickness_variation`` over ``thickness_wavelength``
    and clipped at ``thickness_min``. ``acetabulum`` is
    ``(gap, thickness, plate, coverage, tilt)``: a spherical shell across the
    joint space ``gap``, with bone plates on both faces, covering the head
    within ``coverage`` degrees of an axis tilted ``tilt`` degrees medially;
    it is bone 2 in the labels. Trabeculae and pores as in
    :class:`BonePhantom`. Parameters from measured femora (see the Rust
    module ``phantoms::femur``).
    """

    dims: Tuple[int, int, int] = (110, 104, 140)
    spacing: Triple = (1.0, 0.75, 0.75)
    head_radius: float = 23.15
    neck_shaft_angle: float = 126.4
    anteversion: float = 17.5
    offset: float = 37.9
    neck_radii: Tuple[float, float] = (15.0, 12.5)
    shaft_radius: float = 15.0
    trochanter: Triple = (16.0, 14.0, 11.0)
    rounding: float = 4.0
    neck_thickness: Tuple[float, float] = (0.86, 2.79)
    head_thickness: float = 0.8
    trochanter_thickness: float = 0.8
    shaft_thickness: float = 3.5
    thickness_variation: float = 0.15
    thickness_wavelength: float = 25.0
    thickness_min: float = 0.3
    acetabulum: Optional[Tuple[float, float, float, float, float]] = (4.0, 8.0, 1.0, 75.0, 45.0)
    bvtv: float = 0.18
    endo_bvtv: float = 0.35
    transition_width: float = 2.0
    trabecular_wavelength: float = 1.6
    porosity: float = 0.03
    pore_wavelength: float = 0.35
    rho_soft: float = 57.0
    rho_cartilage: float = 80.0
    rho_bone: float = 1600.0
    rho_marrow: float = -30.0
    blur: Triple = (0.6, 0.5, 0.5)
    noise: float = 30.0
    seed: int = 1
    supersample: int = 3

    FEMUR = 1
    PELVIS = 2

    def _spec(self) -> dict:
        d = asdict(self)
        d["dims"] = tuple(int(x) for x in self.dims)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["blur"] = _triple(self.blur, "blur")
        d["trochanter"] = _triple(self.trochanter, "trochanter")
        d["neck_radii"] = tuple(float(v) for v in self.neck_radii)
        d["neck_thickness"] = tuple(float(v) for v in self.neck_thickness)
        d["acetabulum"] = None if self.acetabulum is None else tuple(float(v) for v in self.acetabulum)
        return d

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(image float32, periosteal labels, endosteal labels)``: uint8,
        0 outside, 1 femur, 2 pelvis (acetabulum)."""
        return _core.femur_phantom(self._spec())

    def edges_along(self, points, normals, x0: float, x1: float) -> np.ndarray:
        """True ``(x_periosteal, x_endosteal, bone)`` along each line ``p + x n``
        (first entry into either bone, then into its marrow space), shape (N, 3)."""
        p = np.ascontiguousarray(points, dtype=np.float64)
        n = np.ascontiguousarray(normals, dtype=np.float64)
        return _core.femur_phantom_edges(self._spec(), p, n, float(x0), float(x1))

    def true_thickness(self, result: "CbmResult", bone: Optional[int] = 1) -> np.ndarray:
        """Reference thickness along each sampling line of a result (NaN where
        the line first enters a bone other than ``bone``, if given)."""
        o = result.options
        e = self.edges_along(result.points, result.normals, -o.outside, o.inside)
        t = e[:, 1] - e[:, 0]
        if bone is not None:
            t[e[:, 2] != bone] = np.nan
        return t


SITES = ("distal_radius", "distal_tibia", "proximal_femur", "knee")


def site_phantom(site: str, seed: int = 1):
    """The phantom of an anatomical site at its scanner's resolution, with
    parameters from the literature (Rust module ``phantoms::sites``).

    ``"distal_radius"`` and ``"distal_tibia"`` (HR-pQCT, 61 um voxels, with
    the ulna / fibula beside them) are :class:`BonePhantom`;
    ``"proximal_femur"`` (hip CT) is a :class:`FemurPhantom`; ``"knee"``
    (clinical CT, with an intercondylar notch) is a :class:`JointPhantom`.
    ``seed`` sets the noise and the random geometry.
    """
    kind, spec = _core.site_phantom(str(site), int(seed))
    cls = {"bone": BonePhantom, "femur": FemurPhantom, "joint": JointPhantom}[kind]
    return cls(**spec)
