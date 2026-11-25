"""Mid-sections of the site phantoms (``cbm.site_phantom``), with the true
periosteal (red) and endosteal (yellow) surfaces.

    python docs/scripts/site_phantoms_figure.py [OUT.png]

Writes ``docs/images/site_phantoms.png``.
"""

import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cortical_bone_mapping_rs as cbm

TITLES = {
    "distal_radius": "distal radius (HR-pQCT), axial",
    "distal_tibia": "distal tibia (HR-pQCT), axial",
    "proximal_femur": "proximal femur (clinical CT), coronal",
}


def section(name):
    """The image, periosteal and endosteal masks of a mid-section, and its
    pixel aspect (row spacing / column spacing)."""
    ph = cbm.site_phantom(name, seed=1)
    img, peri, endo = ph.generate()
    sp = ph.spacing
    if name.startswith("distal"):
        k = img.shape[0] // 2
        return img[k], peri[k] > 0, endo[k] > 0, sp[1] / sp[2], (-100, 1100)
    k = img.shape[1] // 2
    return img[:, k], peri[:, k] > 0, endo[:, k] > 0, sp[0] / sp[2], (-150, 1500)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/images/site_phantoms.png"
    fig, axes = plt.subplots(1, len(cbm.SITES), figsize=(4.5 * len(cbm.SITES), 5))
    for ax, name in zip(axes, cbm.SITES):
        img, peri, endo, aspect, (lo, hi) = section(name)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=aspect)
        ax.contour(peri, [0.5], colors="red", linewidths=0.6, origin="lower")
        ax.contour(endo, [0.5], colors="gold", linewidths=0.6, origin="lower")
        ax.set_title(TITLES[name], fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)


if __name__ == "__main__":
    main()
