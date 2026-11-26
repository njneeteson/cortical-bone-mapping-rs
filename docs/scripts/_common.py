"""Shared by the documentation scripts: provenance of results, figure style,
and 3D rendering.

Provenance. Every results file records the code it was computed with: a hash
of the crate's and the Python package's sources, of the script itself and of
its inputs. Figures are drawn in the same run as the results, or redrawn
from saved results only if that hash still matches; nothing is reused across
code changes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pickle
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
IMAGES = DOCS / "images"
RESULTS = DOCS / "results"
SOURCES = [ROOT / "crates" / "cortical-bone-mapping" / "src", ROOT / "python" / "python", ROOT / "python" / "src"]

# ------------------------------------------------------------------ provenance


FIGURES_MARKER = "# " + "-" * 66 + " figures"


def source_hash(script: str, inputs: dict) -> str:
    """Hash of the library sources, the calling script's computing part
    (everything above its ``FIGURES_MARKER`` line; drawing code cannot change
    results) and its inputs. Line endings are normalised, so that a checkout
    with CRLF endings (git on Windows) hashes the same."""
    h = hashlib.sha256()
    for base in SOURCES:
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix in (".rs", ".py") and "__pycache__" not in f.parts:
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    text = Path(script).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    h.update(text.split(FIGURES_MARKER)[0].encode())
    h.update(json.dumps(inputs, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def provenance(script: str, inputs: dict) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    import cortical_bone_mapping_rs as cbm

    return dict(
        source_hash=source_hash(script, inputs),
        commit=git("rev-parse", "--short", "HEAD"),
        uncommitted_changes=bool(git("status", "--porcelain", "--", "crates", "python")),
        version=cbm.__version__,
        date=datetime.date.today().isoformat(),
        machine=f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
        inputs=inputs,
    )


def _store(name: str) -> Path:
    return Path(tempfile.gettempdir()) / f"cbm_docs_{name}.pkl"


def save_run(name: str, prov: dict, data) -> None:
    """Keep a run's results so that its figures can be redrawn (only while
    the code is unchanged; see :func:`load_run`)."""
    _store(name).write_bytes(pickle.dumps(dict(provenance=prov, data=data)))


def load_run(name: str, script: str, inputs: dict):
    """A saved run, if it was computed by the current code with the same
    inputs; otherwise ``None``."""
    f = _store(name)
    if not f.exists():
        return None
    saved = pickle.loads(f.read_bytes())
    if saved["provenance"]["source_hash"] != source_hash(script, inputs):
        print(f"saved {name} run is from other code or inputs: recomputing")
        return None
    return saved


def write_json(path: Path, prov: dict, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(provenance=prov, **data), indent=2, default=float))
    print("wrote", path)


# ------------------------------------------------------------------ figures

METHODS = ("local", "multigrid", "regularized", "volumetric")
LABELS = {"local": "local (no regularisation)", "multigrid": "multigrid", "regularized": "regularised",
          "volumetric": "volumetric"}
TRUTH, START, PERI, ENDO = "#33dd33", "#ffd400", "#ff3b30", "#33c8ff"
BONE_COLOURS = {"femur": "#4c9be8", "tibia": "#f08a24", "patella": "#3fbf5f", "fibula": "#e8434f"}
DPI = 110


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.titlesize": 14, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "image.interpolation": "nearest",
    })
    return plt


def save(fig, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / name)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print("wrote", IMAGES / name)


def section_segments(vertices, faces, axis, index, spacing, origin=(0.0, 0.0, 0.0)):
    """A mesh cut by an image plane, as line segments in (column, row) voxel
    coordinates of the slice, ready for ``ax.plot``."""
    import cortical_bone_mapping_rs as cbm

    s = cbm.mesh_section(vertices, faces, axis, index, spacing, origin)
    return s[..., ::-1]


def draw_segments(ax, seg, colour, lw=1.6, ls="-"):
    from matplotlib.collections import LineCollection

    if len(seg):
        ax.add_collection(LineCollection(seg, colors=colour, linewidths=lw, linestyles=ls))


# ------------------------------------------------------------------ 3D


def pv_mesh(vertices, faces, scalars=None, name="values"):
    """A PyVista mesh from physical (axis 0, 1, 2) vertices, shown with x =
    axis 2, y = axis 1, z = axis 0."""
    import pyvista as pv

    v = np.asarray(vertices, float)[:, ::-1]
    f = np.hstack([np.full((len(faces), 1), 3), np.asarray(faces)]).astype(np.int64).ravel()
    m = pv.PolyData(v, f)
    if scalars is not None:
        m.point_data[name] = np.asarray(scalars, float)
    return m


def pv_image(image, spacing, origin=(0.0, 0.0, 0.0)):
    """The image as a PyVista grid in the same (x = axis 2, ...) frame."""
    import pyvista as pv

    g = pv.ImageData(dimensions=image.shape[::-1], spacing=tuple(spacing)[::-1], origin=tuple(origin)[::-1])
    g.point_data["intensity"] = np.ascontiguousarray(image).ravel(order="C")
    return g


def plotter(size=(900, 800)):
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    return p


def screenshot(p) -> np.ndarray:
    img = p.screenshot(return_img=True)
    p.close()
    return img
