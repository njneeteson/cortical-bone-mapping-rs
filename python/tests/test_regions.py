"""Cropped regions give the same voxels as full-size masks."""

import numpy as np

from cortical_bone_mapping_rs._regions import Region, bounding_box, changed, grow, others


def _blobs(seed):
    rng = np.random.default_rng(seed)
    shape = (12, 40, 36)
    z, y, x = np.indices(shape)
    out = []
    for _ in range(3):
        c = rng.uniform([0, 0, 0], shape)
        r = rng.uniform(3, 9)
        out.append((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r)
    return shape, out


def test_regions_match_full_masks():
    for seed in range(5):
        shape, masks = _blobs(seed)
        regions = {i + 1: Region.of(m) for i, m in enumerate(masks)}
        for i, m in enumerate(masks):
            r = regions[i + 1]
            assert np.array_equal(r.full(), m)
            assert r.count() == int(m.sum())
            box = grow(bounding_box(m) or (slice(0, 1),) * 3, [2, 3, 1], shape)
            assert np.array_equal(r.within(box), m[box])
            want = np.zeros(shape, bool)
            for j, n in enumerate(masks):
                if j != i:
                    want |= n
            assert np.array_equal(others(regions, i + 1, box), want[box].astype(np.uint8))
        assert changed(regions[1], regions[2]) == int((masks[0] ^ masks[1]).sum())


def test_empty_region():
    r = Region.of(np.zeros((4, 5, 6), bool))
    assert r.box is None and r.count() == 0
    assert not r.full().any()
    assert changed(r, Region.of(np.ones((4, 5, 6), bool))) == 120
