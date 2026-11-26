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


from . import _core
from .logs import log_to
from .bones import map_bones, name_bones, refine_bones, separate_bones, shared_scanner
from .mapping import cortical_bone_mapping, mask_surface, refine_segmentation
from .mesh import mesh_curvature, mesh_section, voxelize, write_vtk
from .model import fit_profiles, profile_model
from .options import PARAM_NAMES, Options, RefineOptions
from .phantoms import SITES, BonePhantom, FemurPhantom, Phantom, site_phantom
from .results import CbmResult, RefineResult
from .segmentation import (
    BuieOptions,
    bone_hulls,
    buie_segmentation,
    grow_bones,
    hull_segmentation,
    label_bones,
    otsu_thresholds,
    soft_tissue_threshold,
)

__all__ = [
    "log_to",
    "Options",
    "RefineOptions",
    "CbmResult",
    "RefineResult",
    "Phantom",
    "BonePhantom",
    "FemurPhantom",
    "site_phantom",
    "SITES",
    "BuieOptions",
    "buie_segmentation",
    "hull_segmentation",
    "otsu_thresholds",
    "label_bones",
    "bone_hulls",
    "grow_bones",
    "soft_tissue_threshold",
    "name_bones",
    "separate_bones",
    "cortical_bone_mapping",
    "refine_segmentation",
    "mask_surface",
    "map_bones",
    "refine_bones",
    "shared_scanner",
    "voxelize",
    "write_vtk",
    "mesh_section",
    "mesh_curvature",
    "fit_profiles",
    "profile_model",
    "PARAM_NAMES",
    "__version__",
]

__version__: str = _core.__version__

