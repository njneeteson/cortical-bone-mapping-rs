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
    "BonePhantom",
    "JointPhantom",
    "BuieOptions",
    "buie_segmentation",
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
    image units. Defaults for the model follow Table 1 of Neeteson & Boyd (2025).

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
        Cortical density (not fitted): a number, ``"estimate"`` (one value for
        all nodes: the median cortical plateau level of nodes whose locally
        fitted cortex has samples more than 2 sigma from both edges) or ``"profile_max"`` (per node: the maximum of
        its smoothed profile, as the original Python implementation does when
        no density is given; a thin or blurred cortex never reaches the tissue
        density, so its thickness is overestimated).
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
    rho_c: Union[float, Literal["estimate", "profile_max"]] = 1000.0
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
    multigrid_spacings: Sequence[float] = (7.7696, 3.8848, 1.9424, 0.9712)
    multigrid_neighbours: int = 5
    seed: int = 0
    reg_lambda: float = 1.0
    reg_sigma: float = 1.9424
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
        if isinstance(self.rho_c, str):
            if self.rho_c not in ("estimate", "profile_max"):
                raise ValueError('rho_c must be a number, "estimate" or "profile_max"')
            d["rho_c_mode"], d["rho_c"] = self.rho_c, 1000.0
        else:
            d["rho_c_mode"], d["rho_c"] = "fixed", float(self.rho_c)
        if self.threads is not None and (not isinstance(self.threads, int) or self.threads < 1):
            raise ValueError("threads must be a positive integer or None")
        return d

    @classmethod
    def preset(cls, site: Literal["radius_tibia", "knee"], voxel: float = 0.0607, **overrides) -> "Options":
        """The parameters of Table 1 of Neeteson & Boyd (2025) for ``site``.

        Isotropic voxels of size ``voxel`` (mm), array axis 0 the scan axis,
        regularised method. Taubin smoothing 100 iterations (pass-band 0.1,
        knee 0.01); sampling 5 mm outside and 15 mm inside at the voxel size;
        profile smoothing sigma = 3 samples; normals in the transverse plane
        (radius & tibia) or along the scan axis (knee); multigrid spacings
        128...16 (knee 256...16) voxels; regularisation lambda = 1,
        sigma_reg = 1.942 mm, 500 (knee 1000) neighbours. Table 1's L-BFGS-B
        tolerances do not apply to this solver. ``overrides`` replace any
        field.
        """
        if site not in ("radius_tibia", "knee"):
            raise ValueError('site must be "radius_tibia" or "knee"')
        knee = site == "knee"
        coarsest = 256 if knee else 128
        spacings = []
        s = coarsest
        while s >= 16:
            spacings.append(s * voxel)
            s //= 2
        base = cls(
            spacing=(voxel,) * 3,
            taubin_iterations=100,
            taubin_passband=0.01 if knee else 0.1,
            outside=5.0,
            inside=15.0,
            resolution=voxel,
            profile_smoothing=3.0,
            normal_constraint="along_axis" if knee else "in_plane",
            normal_axis=0,
            method="regularized",
            multigrid_spacings=tuple(spacings),
            reg_neighbours=1000 if knee else 500,
        )
        return replace(base, **overrides)


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
    rho_c_node: Optional[np.ndarray] = None

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
            rho_c_node=d.get("rho_c_node"),
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
        rc = self.rho_c if self.rho_c_node is None else float(self.rho_c_node[i])
        return profile_model(x, self.params[i], rc)

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
      ``gap`` in every slice;
    * optionally cortical ``breaks``, each ``(angle, width, z_center, length)``:
      the full cortical depth replaced by soft tissue over a periosteal arc of
      length ``width`` for ``|z - z_center| < length / 2``. The true
      periosteal surface spans the break.

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
    femur (above, along array axis 0) across a narrow joint space.

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

    This is a scaled-down joint (a real tibial plateau is ~70 mm wide): it
    reproduces the local geometry of the contact region in a small volume.
    """

    dims: Tuple[int, int, int] = (100, 200, 200)
    spacing: Triple = (0.0607, 0.0607, 0.0607)
    radius: float = 5.0
    femur_scale: float = 1.05
    shape_harmonics: int = 5
    shape_amplitude: float = 0.08
    joint_level: float = 0.45
    joint_space: float = 0.4
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

    TIBIA = 1
    FEMUR = 2

    def _spec(self) -> dict:
        d = asdict(self)
        d["dims"] = tuple(int(x) for x in self.dims)
        d["spacing"] = _triple(self.spacing, "spacing")
        d["blur"] = _triple(self.blur, "blur")
        d["contact_offset"] = tuple(float(v) for v in self.contact_offset)
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
class BuieOptions:
    """Options of :func:`buie_segmentation`, with the filter parameters of
    Buie et al. (2007), Fig. 1. Kernels are in voxels per array axis
    ``(z, y, x)`` (the paper's VTK ``(3,3,1)`` is ``(1, 3, 3)`` here) and, as in
    the paper, are not rescaled with resolution.

    periosteal_threshold, endosteal_threshold
        User inputs 1 and 2, in image units. The paper used native scanner
        units (human XtremeCT: 3000 / 5000); 450 / 320 mg HA/cm^3 are commonly
        used for density-calibrated HR-pQCT (Burghardt et al. 2010).
    gaussian_sigma, gaussian_support
        Step 12: sigma = 3 voxels, support (radius factors) (3,3,1).
    smoothed_threshold
        Step 13 "threshold by lower = 100" on 0/127 masks.
    connectivity_2d
        Per-slice connectivity (the paper: usually identical, faster).
    largest_bone_only
        Pre-processing: keep only the largest bone (removes fibula/ulna).
    """

    periosteal_threshold: float = 450.0
    endosteal_threshold: float = 320.0
    median_kernel: Tuple[int, int, int] = (1, 3, 3)
    periosteal_kernel: Tuple[int, int, int] = (1, 15, 15)
    endosteal_kernel: Tuple[int, int, int] = (1, 10, 10)
    gaussian_sigma: float = 3.0
    gaussian_support: Triple = (1.0, 3.0, 3.0)
    smoothed_threshold: float = 100.0 / 127.0
    connectivity_2d: bool = False
    largest_bone_only: bool = True


def buie_segmentation(image, options: Optional[BuieOptions] = None, **kwargs):
    """Dual-threshold cortical/trabecular segmentation of Buie et al. (2007),
    Bone 41:505-515, doi:10.1016/j.bone.2007.07.007. Returns boolean
    ``(periosteal, cortical, trabecular)`` masks. See ``docs/method.md`` for
    how each step of the paper's Fig. 1 is implemented.
    """
    o = replace(options or BuieOptions(), **kwargs)
    d = asdict(o)
    d["median_kernel"] = tuple(int(v) for v in o.median_kernel)
    d["periosteal_kernel"] = tuple(int(v) for v in o.periosteal_kernel)
    d["endosteal_kernel"] = tuple(int(v) for v in o.endosteal_kernel)
    d["gaussian_support"] = tuple(float(v) for v in o.gaussian_support)
    return _core.buie(_as_image(image), d)
