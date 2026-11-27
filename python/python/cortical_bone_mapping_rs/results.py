"""Results of a fit and of refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

from .mesh import voxelize, write_vtk
from .model import profile_model
from .options import PARAM_NAMES, Options


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
    volumetric
        The volumetric model's outputs (``method="volumetric"``), else
        ``None``: ``sigma_xy``, ``sigma_z`` (the fitted PSF),
        ``scanner_axis``, ``voxels`` (band size), ``noise_sd``,
        ``noise_correlation`` (per axis) and ``noise_factor`` (the variance
        factor of a long block of correlated voxels, for reference), and per
        node ``std_error_t`` (local standard error of ``t``, from the node's
        own voxels with their noise correlation), ``data_share``
        (N, 4: the data's share of the curvature of ``m, t, rho_s, rho_b``)
        ``misfit`` (RMS residual relative to the noise) and ``reach`` (N, 2:
        how far the node's band reaches outside and inside the surface).
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
    volumetric: Optional[dict] = None

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
            volumetric=d.get("volumetric"),
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
        the fitted endosteal surfaces, and the cortex between them (``None``
        if :attr:`RefineOptions.masks` is off).
    endosteal_vertices
        The fitted endosteal surface (faces: ``result.surface_faces``).
    history
        One dict per fit (``history[k]`` is the fit on ``surfaces[k]``): the
        RMS/max displacement it implies, mean thickness and residual, the
        number of fittable and still-moving (not frozen) nodes, untrusted
        fits, whether it was a full fit and whether its solve converged.
    surfaces
        The input surface and the surface after every move
        (``keep_history``).
    converged
        Whether a full, converged fit moved the surface by less than the
        tolerance.
    surfaces_faces
        The faces of each of ``surfaces`` (all the same unless
        ``level_set``, which extracts a new mesh every iteration).
    stalled
        Refinement stopped because the surface stopped making progress.
    message
        Why refinement stopped (with a diagnosis if it stalled).
    timings
        Wall time per stage, summed over all fits, as (name, seconds).
    scanner
        The scanner parameters held in each refinement pass: dicts with
        ``psf`` (volumetric model), ``sigma`` (a global sigma) and ``rho_c``
        (``None`` where not held).
    """

    result: CbmResult
    periosteal_mask: Optional[np.ndarray]
    endosteal_mask: Optional[np.ndarray]
    cortical_mask: Optional[np.ndarray]
    endosteal_vertices: np.ndarray
    history: list
    surfaces: list
    converged: bool
    surfaces_faces: list = field(default_factory=list)
    stalled: bool = False
    message: str = ""
    timings: list = field(default_factory=list)
    scanner: list = field(default_factory=list)

    @property
    def periosteal_vertices(self) -> np.ndarray:
        return self.result.surface_vertices
