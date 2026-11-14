import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm

CYL_OPTS = dict(outside=1.5, inside=2.5, normal_constraint="in_plane", normal_axis=0)


@pytest.fixture(scope="session")
def cylinder():
    ph = cbm.Phantom(noise=30.0, seed=1)
    image, mask = ph.generate()
    return ph, image, mask


@pytest.fixture(scope="session")
def clean_cylinder():
    ph = cbm.Phantom(noise=0.0)
    image, mask = ph.generate()
    return ph, image, mask


def thickness_error(ph, res):
    e = res.thickness - ph.true_thickness(res.points)
    return float(e.mean()), float(np.sqrt((e**2).mean()))


def dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())
