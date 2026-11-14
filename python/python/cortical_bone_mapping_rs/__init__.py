"""Cortical bone mapping in Rust.

Model-based measurement of cortical thickness from CT / HR-pQCT images
(Treece et al. 2010, 2012, 2015), with the global multigrid and spatially
regularised fits of Neeteson & Boyd (2025, *Comput. Biol. Med.*,
https://doi.org/10.1016/j.compbiomed.2025.110636), a curvature penalty on the
fitted surfaces, and model-driven refinement of the outer-surface
segmentation.

At every vertex of the outer bone surface a density profile is sampled along
the inward normal and fitted with a blurred three-level step model::

    f(x) = rho_s + (rho_c - rho_s)/2 [1 + erf((x - m + t/2) / (sigma sqrt 2))]
                 + (rho_b - rho_c)/2 [1 + erf((x - m - t/2) / (sigma sqrt 2))]

giving the cortical thickness ``t``, the position ``m`` of the cortical
mid-line along the normal, the blur ``sigma`` and the soft tissue and
trabecular densities, with the cortical density ``rho_c`` fixed.

Coordinates are physical and in array-axis order (axis 0, 1, 2), i.e.
``(z, y, x)`` for a NumPy array: voxel ``(i, j, k)`` is at
``origin + (i, j, k) * spacing``.

Example
-------
>>> import cortical_bone_mapping_rs as cbm
>>> ph = cbm.Phantom(noise=0.0)
>>> image, mask = ph.generate()
>>> res = cbm.cortical_bone_mapping(
...     image, mask, spacing=ph.spacing, outside=1.5, inside=2.5,
...     normal_constraint="in_plane")
>>> err = res.thickness - ph.true_thickness(res.points)
>>> bool(abs(err.mean()) < 0.01)
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Real
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from . import _core

__all__ = [
    "Options",
    "RefineOptions",
    "CbmResult",
    "RefineResult",
    "Phantom",
    "cortical_bone_mapping",
    "refine_segmentation",
    "voxelize",
    "write_vtk",
    "fit_profiles",
    "profile_model",
    "PARAM_NAMES",
    "__version__",
]

__version__: str = _core.__version__

#: Column order of :attr:`CbmResult.params`.
PARAM_NAMES = ("center", "thickness", "sigma", "rho_s", "rho_b")

Triple = Tuple[float, float, float]
Quint = Tuple[float, float, float, float, float]


@dataclass
class Options:
    """All options of the cortical bone mapping pipeline.

    Lengths are in the physical units of ``spacing`` (e.g. mm), densities in
    image units. Defaults for the model follow Table 6.1 of Neeteson & Boyd.

    **Geometry**

    spacing, origin
        Voxel size and position of voxel (0, 0, 0) per array axis.

    **Surface** (extraction from the mask by surface nets, then Taubin smoothing)

    mask_smoothing
        Gaussian blur of the mask (voxels) before extraction; 0 = none.
    taubin_iterations, taubin_passband, taubin_lambda
        Taubin lambda|mu smoothing with ``1/lambda + 1/mu = passband``.
    exclude_border
        Do not fit vertices on the caps where the mask is cut by the image edge.

    **Sampling**

    outside, inside
        Profile extent outside and inside the surface.
    resolution
        Sample spacing (``None`` = the smallest voxel spacing).
    profile_smoothing
        1D Gaussian smoothing of each profile (standard deviation in samples).
    normal_constraint, normal_axis
        ``"none"``, ``"in_plane"`` (project normals into the plane
        perpendicular to ``normal_axis``, e.g. transverse plane of a long bone)
        or ``"along_axis"`` (sample along +/- ``normal_axis``, e.g. the
        subchondral plate).

    **Model**

    rho_c
        Fixed cortical density.
    residual_boost
        ``gamma_0``: weight of residuals at the surface relative to the inner
        end of the profile (Treece et al. 2010).
    init, lower, upper
        Initial guess and bounds of ``(m, t, sigma, rho_s, rho_b)``.
    sigma
        ``"free"`` (fitted per node), ``"fixed"`` (``sigma_value``), ``"psf"``
        (``sigma`` along each normal from the Gaussian PSF ``psf`` per axis) or
        ``"estimate_psf"`` (estimate the PSF from a first fit with free
        ``sigma`` using nodes with ``t >= psf_min_ratio * sigma``, then fix it;
        Treece-style). ``psf_model`` is ``"isotropic"``, ``"axial"`` (in-plane
        + along ``psf_axis``) or ``"anisotropic"``.

    **Solver**

    max_iterations, ftol, gtol, xtol
        Levenberg-Marquardt stopping criteria (per node for the local method,
        per problem for the global methods).
    cg_max_iterations, cg_tolerance
        Conjugate-gradient inner solves of the global methods.

    **Method**

    method
        ``"local"``: independent fit per node (Treece et al.).
        ``"multigrid"``: ``m, t, sigma`` on control points at
        ``multigrid_spacings`` (coarse to fine) interpolated to the nodes from
        ``multigrid_neighbours`` control points (Neeteson & Boyd).
        ``"regularized"``: per-node ``m, t, sigma`` with the spatial
        regularisation of Neeteson & Boyd (``reg_lambda``, Gaussian
        ``reg_sigma`` over ``reg_neighbours`` nearest nodes, truncated at
        ``reg_cutoff * reg_sigma``), optionally plus a curvature penalty.
    init_from_global
        Initialise from one model fitted to all profiles.
    init_from_local
        (regularized) Start from the local fit instead.

    **Curvature penalty** (``method="regularized"`` only; extension)

    curvature
        ``None``, ``"change"`` (penalise curvature added to the fitted
        periosteal/endosteal surfaces relative to the parallel surface of the
        input, ~ Laplacian of the offset; uniform offsets are free) or
        ``"excess"`` (penalise mean curvature above ``curvature_kappa``,
        1/length).
    curvature_weight
        Weight ``mu``. The residual is
        ``sqrt(mu / N) * rho_c * curvature_length * curvature``.
    curvature_length
        Length scale (``None`` = globally fitted thickness).
    curvature_scale, curvature_neighbours
        Gaussian point-cloud Laplacian at this scale over this many neighbours;
        ``curvature_scale=None`` uses the cotangent mesh Laplacian (mesh
        resolution; slower to solve).
    curvature_surfaces
        ``"both"``, ``"periosteal"`` or ``"endosteal"``.

    **Other**

    keep_profiles
        Return the sampled profiles.
    threads
        Worker threads (``None`` = all cores).
    """

    spacing: Triple = (1.0, 1.0, 1.0)
    origin: Triple = (0.0, 0.0, 0.0)
    # surface
    mask_smoothing: float = 0.0
    taubin_iterations: int = 100
    taubin_passband: float = 0.1
    taubin_lambda: float = 0.5
    exclude_border: bool = True
    # sampling
    outside: float = 2.0
    inside: float = 4.0
    resolution: Optional[float] = None
    profile_smoothing: float = 0.0
    normal_constraint: Literal["none", "in_plane", "along_axis"] = "none"
    normal_axis: int = 0
    # model
    rho_c: float = 1000.0
    residual_boost: float = 10.0
    init: Quint = (0.0, 1.0, 0.1, 0.0, 0.0)
    lower: Quint = (-3.0, 0.0607, 0.01, -200.0, -200.0)
    upper: Quint = (6.0, 12.0, 2.0, 400.0, 400.0)
    sigma: Literal["free", "fixed", "psf", "estimate_psf"] = "free"
    sigma_value: float = 0.1
    psf: Triple = (0.1, 0.1, 0.1)
    psf_model: Literal["isotropic", "axial", "anisotropic"] = "axial"
    psf_axis: int = 0
    psf_min_ratio: float = 3.0
    # solver
    max_iterations: int = 100
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-10
    cg_max_iterations: int = 200
    cg_tolerance: float = 1e-3
    # method
    method: Literal["local", "multigrid", "regularized"] = "local"
    multigrid_spacings: Sequence[float] = (7.77, 3.885, 1.942, 0.971)
    multigrid_neighbours: int = 5
    seed: int = 0
    reg_lambda: float = 1.0
    reg_sigma: float = 1.924
    reg_neighbours: int = 500
    reg_cutoff: float = 3.0
    init_from_local: bool = False
    init_from_global: bool = True
    # curvature
    curvature: Optional[Literal["change", "excess"]] = None
    curvature_weight: float = 1.0
    curvature_kappa: float = 0.5
    curvature_length: Optional[float] = None
    curvature_scale: Optional[float] = 0.25
    curvature_neighbours: int = 64
    curvature_surfaces: Literal["both", "periosteal", "endosteal"] = "both"
    # other
    keep_profiles: bool = False
    threads: Optional[int] = None

    def _to_core(self) -> dict:
        d = asdict(self)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["origin"] = _triple(self.origin, "origin")
        d["psf"] = _triple(self.psf, "psf")
        for k in ("init", "lower", "upper"):
            v = tuple(float(x) for x in d[k])
            if len(v) != 5:
                raise ValueError(f"{k} must have 5 values (m, t, sigma, rho_s, rho_b)")
            d[k] = v
        d["multigrid_spacings"] = [float(s) for s in self.multigrid_spacings]
        if self.threads is not None and (not isinstance(self.threads, int) or self.threads < 1):
            raise ValueError("threads must be a positive integer or None")
        return d


@dataclass
class RefineOptions:
    """Options of :func:`refine_segmentation`.

    refine_iterations
        Maximum number of move/refit iterations.
    relaxation
        Under-relaxation ``alpha`` in (0, 1]: each iteration moves vertices by
        ``alpha * (m - t/2)`` along the sampling normal.
    tolerance
        Stop when the RMS displacement is below this (``None`` = a tenth of the
        smallest voxel spacing).
    smoothing_iterations
        Taubin iterations after every move.
    max_step
        Optional cap on the displacement per iteration.
    keep_history
        Keep the surface of every iteration (:attr:`RefineResult.surfaces`).
    """

    refine_iterations: int = 20
    relaxation: float = 0.5
    tolerance: Optional[float] = None
    smoothing_iterations: int = 2
    max_step: Optional[float] = None
    keep_history: bool = False

    def _to_core(self) -> dict:
        d = asdict(self)
        d["max_iterations"] = d.pop("refine_iterations")
        return d


def _triple(v, name: str) -> Triple:
    if isinstance(v, Real):
        v = (float(v),) * 3
    v = tuple(float(x) for x in v)
    if len(v) != 3:
        raise ValueError(f"{name} must be a scalar or have 3 values")
    return v  # type: ignore[return-value]


def _as_u8(arr, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {a.shape}")
    if a.dtype in (np.bool_, np.uint8, np.int8):
        return np.ascontiguousarray(a).view(np.uint8)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be boolean or numeric, got {a.dtype}")
    return (a != 0).view(np.uint8)


_IMAGE_DTYPES = (np.float32, np.float64, np.int16, np.uint16, np.uint8, np.int8, np.int32, np.uint32)


def _as_image(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"image must be a 3D array, got shape {a.shape}")
    if a.dtype == np.bool_:
        return a.view(np.uint8)
    if a.dtype.type not in _IMAGE_DTYPES:
        if a.dtype.kind not in "iuf":
            raise TypeError(f"image must be numeric, got {a.dtype}")
        return a.astype(np.float64)
    return a


def _make_options(options: Optional[Options], spacing, kwargs) -> Options:
    o = options if options is not None else Options()
    valid = {f.name for f in fields(Options)}
    bad = set(kwargs) - valid
    if bad:
        raise TypeError(f"unknown option(s): {', '.join(sorted(bad))}")
    if spacing is not None:
        kwargs = dict(kwargs, spacing=spacing)
    return replace(o, **kwargs)


@dataclass(frozen=True)
class CbmResult:
    """Fitted parameters at every measured surface node.

    points, normals : (N, 3)
        Surface nodes and inward unit sampling directions.
    faces : (F, 3)
        Triangles between the nodes (indices into ``points``).
    params : (N, 5)
        ``(m, t, sigma, rho_s, rho_b)`` per node (see :data:`PARAM_NAMES`).
    residual : (N,)
        Weighted RMS fit residual per node.
    vertex_ids : (N,)
        Index of each node in the full surface ``surface_vertices``.
    surface_vertices, surface_faces
        The full smoothed outer surface extracted from the mask.
    global_params
        The single model fitted to all profiles.
    psf
        PSF used or estimated (``sigma="psf"/"estimate_psf"``), else ``None``.
    info
        Diagnostics: iterations, costs, convergence, timings (seconds).
    profile_x, profiles
        Sample positions and sampled profiles (``keep_profiles=True``).
    """

    points: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    params: np.ndarray
    residual: np.ndarray
    vertex_ids: np.ndarray
    surface_vertices: np.ndarray
    surface_faces: np.ndarray
    global_params: Tuple[float, ...]
    psf: Optional[Tuple[float, float, float]]
    rho_c: float
    info: dict
    options: Options
    profile_x: Optional[np.ndarray] = None
    profiles: Optional[np.ndarray] = None

    @classmethod
    def _from_core(cls, d: dict, options: Options) -> "CbmResult":
        return cls(
            points=d["points"],
            normals=d["normals"],
            faces=d["faces"],
            params=d["params"],
            residual=d["residual"],
            vertex_ids=d["vertex_ids"],
            surface_vertices=d["surface_vertices"],
            surface_faces=d["surface_faces"],
            global_params=tuple(d["global_params"]),
            psf=None if d["psf"] is None else tuple(d["psf"]),
            rho_c=d["rho_c"],
            info=d["info"],
            options=options,
            profile_x=d.get("profile_x"),
            profiles=d.get("profiles"),
        )

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def center(self) -> np.ndarray:
        """Distance ``m`` from the node to the cortical mid-line (inward +)."""
        return self.params[:, 0]

    @property
    def thickness(self) -> np.ndarray:
        return self.params[:, 1]

    @property
    def sigma(self) -> np.ndarray:
        return self.params[:, 2]

    @property
    def rho_s(self) -> np.ndarray:
        return self.params[:, 3]

    @property
    def rho_b(self) -> np.ndarray:
        return self.params[:, 4]

    @property
    def periosteal_points(self) -> np.ndarray:
        """Fitted periosteal edge ``p + (m - t/2) n``."""
        return self.points + (self.center - 0.5 * self.thickness)[:, None] * self.normals

    @property
    def endosteal_points(self) -> np.ndarray:
        """Fitted endosteal edge ``p + (m + t/2) n``."""
        return self.points + (self.center + 0.5 * self.thickness)[:, None] * self.normals

    def model(self, i: int, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Modelled profile of node ``i`` at positions ``x`` (default: the samples)."""
        if x is None:
            if self.profile_x is None:
                raise ValueError("pass x or run with keep_profiles=True")
            x = self.profile_x
        return profile_model(x, self.params[i], self.rho_c)

    def displaced_surface(self, side: Literal["periosteal", "endosteal"]) -> np.ndarray:
        """Full-surface vertices with fitted nodes moved to the fitted edge."""
        sign = {"periosteal": -0.5, "endosteal": 0.5}[side]
        v = self.surface_vertices.copy()
        v[self.vertex_ids] = self.points + (self.center + sign * self.thickness)[:, None] * self.normals
        return v

    def masks(self, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean ``(periosteal, endosteal, cortical)`` masks enclosed by the
        fitted surfaces (cortical = periosteal and not endosteal)."""
        o = self.options
        p = voxelize(self.displaced_surface("periosteal"), self.surface_faces, shape, o.spacing, o.origin)
        e = voxelize(self.displaced_surface("endosteal"), self.surface_faces, shape, o.spacing, o.origin)
        return p, e, p & ~e

    def save_vtk(self, path) -> None:
        """Write the measured nodes with all parameters as VTK PolyData."""
        scalars = [(name, self.params[:, k]) for k, name in enumerate(PARAM_NAMES)]
        scalars.append(("residual", self.residual))
        write_vtk(path, self.points, self.faces, dict(scalars), {"normals": self.normals})

    def to_pyvista(self):
        """The measured nodes as a ``pyvista.PolyData`` with point data."""
        import pyvista as pv

        faces = np.hstack([np.full((len(self.faces), 1), 3), self.faces]).astype(np.int64).ravel()
        mesh = pv.PolyData(self.points, faces)
        for k, name in enumerate(PARAM_NAMES):
            mesh.point_data[name] = self.params[:, k]
        mesh.point_data["residual"] = self.residual
        mesh.point_data["normals"] = self.normals
        return mesh


@dataclass(frozen=True)
class RefineResult:
    """Result of :func:`refine_segmentation`.

    result
        The final fit on the refined surface.
    periosteal_mask, endosteal_mask, cortical_mask
        Boolean masks of the regions enclosed by the refined periosteal and
        the fitted endosteal surfaces, and the cortex between them.
    endosteal_vertices
        The fitted endosteal surface (faces: ``result.surface_faces``).
    history
        Per-iteration RMS/max displacement, mean thickness and residual.
    surfaces
        Surface vertices before and after every iteration (``keep_history``);
        all share ``result.surface_faces``.
    converged
        Whether the displacement fell below the tolerance.
    """

    result: CbmResult
    periosteal_mask: np.ndarray
    endosteal_mask: np.ndarray
    cortical_mask: np.ndarray
    endosteal_vertices: np.ndarray
    history: list
    surfaces: list
    converged: bool

    @property
    def periosteal_vertices(self) -> np.ndarray:
        return self.result.surface_vertices


def cortical_bone_mapping(
    image,
    mask,
    spacing: Union[float, Sequence[float], None] = None,
    *,
    sub_mask=None,
    options: Optional[Options] = None,
    **kwargs,
) -> CbmResult:
    """Measure cortical thickness by cortical bone mapping.

    Parameters
    ----------
    image : array_like, 3D
        Grey-scale (ideally density-calibrated) image. float32/64, int8/16/32
        and uint8/16/32 are used without copying.
    mask : array_like, 3D
        Binary segmentation of the bone (non-zero = inside the periosteal
        surface).
    spacing : float or 3 floats, optional
        Voxel size per array axis (overrides ``options.spacing``).
    sub_mask : array_like, 3D, optional
        Only surface vertices adjacent to a non-zero sub-mask voxel are measured.
    options : Options, optional
        Base options; any field can also be given as a keyword argument.

    Returns
    -------
    CbmResult
    """
    o = _make_options(options, spacing, kwargs)
    img = _as_image(image)
    m = _as_u8(mask, "mask")
    if img.shape != m.shape:
        raise ValueError(f"image {img.shape} and mask {m.shape} shapes differ")
    s = None if sub_mask is None else _as_u8(sub_mask, "sub_mask")
    d = _core.run(img, m, s, o._to_core())
    return CbmResult._from_core(d, o)


def refine_segmentation(
    image,
    mask,
    spacing: Union[float, Sequence[float], None] = None,
    *,
    sub_mask=None,
    options: Optional[Options] = None,
    refine: Optional[RefineOptions] = None,
    **kwargs,
) -> RefineResult:
    """Iteratively snap the segmentation's surface to the fitted periosteal
    surface, with under-relaxation, until it stabilises.

    Keyword arguments may be fields of :class:`Options` or of
    :class:`RefineOptions`.
    """
    rf = {f.name for f in fields(RefineOptions)}
    r = refine if refine is not None else RefineOptions()
    r = replace(r, **{k: v for k, v in kwargs.items() if k in rf})
    o = _make_options(options, spacing, {k: v for k, v in kwargs.items() if k not in rf})
    if not (0.0 < r.relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    img = _as_image(image)
    m = _as_u8(mask, "mask")
    if img.shape != m.shape:
        raise ValueError(f"image {img.shape} and mask {m.shape} shapes differ")
    s = None if sub_mask is None else _as_u8(sub_mask, "sub_mask")
    d = _core.refine(img, m, s, o._to_core(), r._to_core())
    return RefineResult(
        result=CbmResult._from_core(d["result"], o),
        periosteal_mask=d["periosteal_mask"],
        endosteal_mask=d["endosteal_mask"],
        cortical_mask=d["cortical_mask"],
        endosteal_vertices=d["endosteal_vertices"],
        history=d["history"],
        surfaces=d["surfaces"],
        converged=d["converged"],
    )


def profile_model(x, params, rho_c: float) -> np.ndarray:
    """Evaluate the profile model at ``x`` for parameters ``(m, t, sigma,
    rho_s, rho_b)`` (``params`` may be (5,) or (N, 5); result broadcasts)."""
    from math import erf, sqrt

    x = np.asarray(x, float)
    p = np.asarray(params, float)
    m, t, s, rs, rb = (p[..., k, None] if p.ndim > 1 else p[k] for k in range(5))
    e = np.vectorize(erf, otypes=[float])
    return (
        rs
        + 0.5 * (rho_c - rs) * (1 + e((x - m + t / 2) / (s * sqrt(2))))
        + 0.5 * (rb - rho_c) * (1 + e((x - m - t / 2) / (s * sqrt(2))))
    )


def fit_profiles(
    profiles,
    x,
    *,
    init=(0.0, 1.0, 0.1, 0.0, 0.0),
    rho_c: float = 1000.0,
    residual_boost: float = 10.0,
    lower=(-3.0, 0.0607, 0.01, -200.0, -200.0),
    upper=(6.0, 12.0, 2.0, 400.0, 400.0),
    fixed: Sequence[str] = (),
    max_iterations: int = 100,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    xtol: float = 1e-10,
) -> dict:
    """Fit the profile model independently to each row of ``profiles``.

    This is the local method without the surface pipeline, e.g. for
    profiles sampled elsewhere. ``init`` is (5,) or (N, 5); parameters named
    in ``fixed`` (from :data:`PARAM_NAMES`) keep their initial value.

    Returns a dict with ``params`` (N, 5), ``cost`` (N,), ``iterations`` (N,)
    and ``converged`` (N,).
    """
    prof = np.ascontiguousarray(np.atleast_2d(profiles), dtype=np.float32)
    x = [float(v) for v in np.asarray(x, float).ravel()]
    ini = np.ascontiguousarray(np.atleast_2d(np.asarray(init, float)))
    bad = set(fixed) - set(PARAM_NAMES)
    if bad:
        raise ValueError(f"unknown parameter(s) {sorted(bad)}; use {PARAM_NAMES}")
    free = [name not in fixed for name in PARAM_NAMES]
    p, c, it, ok = _core.fit_profiles(
        prof, x, ini, float(rho_c), float(residual_boost),
        tuple(map(float, lower)), tuple(map(float, upper)), free,
        int(max_iterations), float(ftol), float(gtol), float(xtol),
    )
    return {"params": p, "cost": c, "iterations": it, "converged": ok}


def voxelize(vertices, faces, shape, spacing=1.0, origin=0.0) -> np.ndarray:
    """Rasterise a closed triangle mesh (physical coordinates) into a boolean
    mask of ``shape`` (even-odd rule along axis 0)."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError("shape must have 3 entries")
    return _core.voxelize(v, f, shape, _triple(spacing, "spacing"), _triple(origin, "origin"))


def write_vtk(path, vertices, faces, scalars: Optional[dict] = None, vectors: Optional[dict] = None) -> None:
    """Write a triangle mesh with per-vertex arrays as legacy binary VTK."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    s = [(k, [float(x) for x in np.asarray(a, float).ravel()]) for k, a in (scalars or {}).items()]
    vec = [(k, np.ascontiguousarray(a, dtype=np.float64)) for k, a in (vectors or {}).items()]
    _core.write_vtk(str(path), v, f, s, vec)


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
