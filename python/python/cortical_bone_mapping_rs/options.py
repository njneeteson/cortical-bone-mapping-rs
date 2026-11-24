"""Options of the fit and of refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Literal, Optional, Sequence, Tuple, Union


from ._util import Quint, Triple, _triple


#: Column order of :attr:`CbmResult.params`.
PARAM_NAMES = ("center", "thickness", "sigma", "rho_s", "rho_b")



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
    half_width_inside
        Limit each profile's inside part to half the object's width along its
        normal (where the line leaves the region enclosed by the surface),
        and at least one voxel: at thin tips and plates the profile stops at
        the middle instead of running out through the far side.
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
    global_params
        Global methods: the parameters (names from ``PARAM_NAMES``) that take
        one value for the whole surface; the others are per-node fields,
        regularised in the regularised method. Default ``("rho_s",
        "rho_b")`` (Neeteson & Boyd). ``()`` fits the densities per node too,
        needed where the trabecular density varies over the bone (dense
        epiphyses, fatty diaphyseal marrow in clinical CT).
    density_smoothing
        Weight of the smoothing of density fields relative to that of ``m``
        and ``t``.

    **Solver**

    max_iterations, ftol, gtol, xtol
        Levenberg-Marquardt stopping criteria (per node for the local method,
        per problem for the global methods).
    cg_max_iterations, cg_tolerance
        Conjugate-gradient inner solves of the global methods.
    cg_max_total
        CG iterations allowed over one global solve; a fit that exhausts it is
        flagged (``info["budget_exhausted"]``) and stops refinement.
    progress_every
        Print the state of the global (multigrid / regularised) solver to
        stderr every this many iterations (cost, gradient, CG work, time);
        0 = never.

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
        ``"volumetric"``: the volumetric model (extension; see
        ``docs/volumetric_model.md``): the profile model evaluated at every
        voxel of the band ``[-outside, inside]`` around the surface, with a
        global PSF, a curvature penalty on ``m, t`` and spatial smoothing of
        the density fields (``vol_*`` options below).
    init_from_global
        Initialise from one model fitted to all profiles.
    init_from_local
        (regularized) Start from the local fit instead.

    **Curvature penalty** (``method="regularized"`` only; extension)

    curvature
        ``None``, ``"absolute"`` (penalise the curvature of the fitted
        periosteal/endosteal surfaces above ``curvature_kappa``, 1/length;
        ``curvature_kappa=0`` prefers a flat surface; the default, 5 / mm, is
        the contour curvature threshold of Neeteson et al. 2023, from real
        HR-pQCT radius and tibia segmentations), ``"variation"`` (penalise
        how fast the curvature changes along the surface, ``|grad(k1 + k2)|^2``:
        smoothly curved bone is free, bumps and dents are not; mean curvature
        only) or ``"change"`` (penalise curvature added relative to the
        parallel surface of the input, so uniform offsets are free; over a
        refinement its reference moves with the surface).
    curvature_measure
        ``"mean"`` (``|k1 + k2| = 2|H|``) or ``"total"``
        (``sqrt(k1^2 + k2^2)``, which unlike ``H`` is not zero at a saddle);
        both equal the contour curvature ``1/R`` on a shaft. ``"absolute"``
        only.
    curvature_weight
        Weight ``mu``. The residual is
        ``sqrt(mu / N) * rho_c * curvature_length * max(0, curvature - kappa)``.
    curvature_endosteal_weight
        Weight of the endosteal surface (``None`` = the same as the
        periosteal one).
    curvature_length
        Length scale (``None`` = globally fitted thickness).
    curvature_scale, curvature_neighbours
        Gaussian point-cloud Laplacian at this scale over this many neighbours;
        ``curvature_scale=None`` uses the cotangent mesh Laplacian (mesh
        resolution; slower to solve).
    curvature_surfaces
        ``"both"``, ``"periosteal"`` or ``"endosteal"``.

    **Volumetric model** (``method="volumetric"``; extension)

    vol_scanner_axis
        Array axis of the scanner (table) axis, along which the PSF has
        ``sigma_z``.
    vol_sigma_xy
        Fix the in-plane blur (``None`` = fitted, one global value).
    vol_sigma_z
        ``"fit"`` (a second global value), ``"tied"`` (equal to the in-plane
        blur: isotropic PSF, or a shaft whose normals are all in-plane) or a
        number (fixed).
    vol_lambda
        Weight of the spatial smoothing of the ``rho_s``, ``rho_b`` fields
        (operator as ``reg_sigma``, ``reg_neighbours``, ``reg_cutoff``);
        0 = none.
    vol_global_densities
        One global ``rho_s`` and ``rho_b`` instead of fields.
    vol_curvature, vol_curvature_weight
        The curvature penalty on the fitted surfaces (the regulariser of
        ``m, t``): ``"absolute"``, ``"variation"``, ``"change"`` or ``None``, with its weight;
        ``curvature_kappa``, ``curvature_measure``, ``curvature_length``, ``curvature_scale``,
        ``curvature_neighbours`` and ``curvature_surfaces`` apply.
    vol_voxel_stride
        Use every n-th voxel along each axis.

    **Neighbourhoods**

    neighbour_distance
        How the regularisation operator, the Gaussian curvature Laplacian and
        the multigrid basis measure the distance between nodes:
        ``"euclidean"`` (straight line, as in Neeteson & Boyd 2025) or
        ``"surface"`` (shortest path along the mesh, which does not couple
        the two walls of a notch or the two cortices of a plate). ``None`` =
        the method's default (Euclidean for the 1D methods, surface distance
        for the volumetric model).

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
    half_width_inside: bool = False
    normal_constraint: Literal["none", "in_plane", "along_axis"] = "none"
    normal_axis: int = 0
    # model
    rho_c: Union[float, Literal["estimate", "profile_max"]] = 1000.0
    residual_boost: float = 10.0
    init: Quint = (0.0, 1.0, 0.1, 0.0, 0.0)
    lower: Quint = (-3.0, 0.0607, 0.01, -200.0, -200.0)
    upper: Quint = (6.0, 12.0, 2.0, 400.0, 400.0)
    sigma: Literal["free", "fixed", "psf", "estimate_psf"] = "free"
    global_params: Tuple[str, ...] = ("rho_s", "rho_b")
    density_smoothing: float = 1.0
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
    cg_max_total: int = 2000
    progress_every: int = 1
    # method
    method: Literal["local", "multigrid", "regularized", "volumetric"] = "local"
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
    curvature: Optional[Literal["absolute", "variation", "change"]] = None
    curvature_measure: Literal["mean", "total"] = "mean"
    curvature_weight: float = 1.0
    curvature_kappa: float = 5.0
    curvature_endosteal_weight: Optional[float] = None
    curvature_length: Optional[float] = None
    curvature_scale: Optional[float] = 0.25
    curvature_neighbours: int = 64
    curvature_surfaces: Literal["both", "periosteal", "endosteal"] = "both"
    # volumetric
    vol_scanner_axis: int = 0
    vol_sigma_xy: Optional[float] = None
    vol_sigma_z: Union[Literal["fit", "tied"], float] = "fit"
    vol_lambda: float = 1.0
    vol_global_densities: bool = False
    vol_curvature: Optional[Literal["absolute", "variation", "change"]] = "change"
    vol_curvature_weight: float = 0.3
    vol_voxel_stride: int = 1
    # neighbourhoods
    neighbour_distance: Optional[Literal["euclidean", "surface"]] = None
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
        bad = set(self.global_params) - set(PARAM_NAMES)
        if bad:
            raise ValueError(f"unknown global_params {sorted(bad)}; use names from {PARAM_NAMES}")
        d["global"] = [n in self.global_params for n in PARAM_NAMES]
        del d["global_params"]
        if isinstance(self.rho_c, str):
            if self.rho_c not in ("estimate", "profile_max"):
                raise ValueError('rho_c must be a number, "estimate" or "profile_max"')
            d["rho_c_mode"], d["rho_c"] = self.rho_c, 1000.0
        else:
            d["rho_c_mode"], d["rho_c"] = "fixed", float(self.rho_c)
        if self.threads is not None and (not isinstance(self.threads, int) or self.threads < 1):
            raise ValueError("threads must be a positive integer or None")
        if isinstance(self.vol_sigma_z, str):
            if self.vol_sigma_z not in ("fit", "tied"):
                raise ValueError('vol_sigma_z must be "fit", "tied" or a number')
            d["vol_sigma_z_mode"], d["vol_sigma_z"] = self.vol_sigma_z, 0.0
        else:
            d["vol_sigma_z_mode"], d["vol_sigma_z"] = "fixed", float(self.vol_sigma_z)
        if self.neighbour_distance not in (None, "euclidean", "surface"):
            raise ValueError('neighbour_distance must be None, "euclidean" or "surface"')
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
    verbose
        Print one line to stderr after the initial fit and after every
        iteration (displacement, mean thickness, fit iterations, elapsed time).
    label
        Prefix of the progress lines (``refine_bones`` uses the bone's label).
    edge_ratio
        Inexact inner fits: while the surface still moves, each global solve
        stops once its steps change the fitted edges (``m - t/2``) by less
        than ``edge_ratio`` times the last displacement (RMS); 0 solves
        fully. The initial fit and the verifying fit are always full solves:
        refinement has converged only when a full, converged solve moves the
        surface by less than ``tolerance``.
    stall_iterations
        Stop as stalled (``RefineResult.stalled``, with a diagnosis in
        ``message``) when for this many fits the displacement has not fallen
        below 95% of its best and no further region has converged.
    region_radius, region_tolerance
        Regional convergence test: the displacement RMS over the nodes
        within ``region_radius`` of each node (default 5 voxels) must be
        below ``region_tolerance`` everywhere (default 2 x ``tolerance``;
        ``float("inf")`` disables it). A narrow band still moving (a surface
        zipping into a concavity) hardly changes the RMS over the whole
        surface but stands out regionally.
    level_set
        Carry the surface as a level set (signed distance on the image grid)
        and extract a new mesh every iteration, instead of moving the same
        vertices: the node spacing stays even and the surface cannot fold,
        however far it moves. ``smoothing_iterations`` is then unused.
    freeze
        Regional convergence: a node is *still* once, for
        ``freeze_patience`` consecutive iterations, it moved less than
        ``freeze_displacement`` and its thickness changed by less than
        ``freeze_thickness`` (both default to the tolerance). It freezes
        when every node within ``freeze_radius`` (default 3 voxels) is
        still. Frozen nodes are no longer moved or refitted, except those
        within ``freeze_halo`` of a moving node (default: the reach of the
        regularisation, ``cutoff * sigma_reg``), which are refitted as fixed
        values: boundary conditions for the regularisation. A node whose fit
        fails (cannot locate an edge) is never still. A frozen node thaws
        when the moving nodes within ``freeze_radius`` have moved
        ``freeze_thaw`` in total (largest nearby move summed over iterations;
        default 1 voxel). ``rho_c`` and the
        densities are fixed after the initial fit, and the whole surface is
        fitted once more at the end.
    """

    refine_iterations: int = 20
    relaxation: float = 0.5
    tolerance: Optional[float] = None
    smoothing_iterations: int = 2
    max_step: Optional[float] = None
    keep_history: bool = False
    verbose: bool = True
    label: str = ""
    edge_ratio: float = 0.1
    stall_iterations: int = 5
    region_radius: Optional[float] = None
    region_tolerance: Optional[float] = None
    level_set: bool = False
    freeze: bool = False
    freeze_patience: int = 3
    freeze_displacement: Optional[float] = None
    freeze_thickness: Optional[float] = None
    freeze_radius: Optional[float] = None
    freeze_halo: Optional[float] = None
    freeze_thaw: Optional[float] = None

    def _to_core(self) -> dict:
        d = asdict(self)
        d["max_iterations"] = d.pop("refine_iterations")
        return d


def _make_options(options: Optional[Options], spacing, kwargs) -> Options:
    o = options if options is not None else Options()
    valid = {f.name for f in fields(Options)}
    bad = set(kwargs) - valid
    if bad:
        raise TypeError(f"unknown option(s): {', '.join(sorted(bad))}")
    if spacing is not None:
        kwargs = dict(kwargs, spacing=spacing)
    return replace(o, **kwargs)
