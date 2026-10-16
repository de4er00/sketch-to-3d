import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sketch3d', 'judge'))
import judge  # noqa: E402
import metrics  # noqa: E402


def square(h, w, r0, r1, c0, c1):
    m = np.zeros((h, w), bool)
    m[r0:r1, c0:c1] = True
    return m


def test_iou_identical_disjoint_partial():
    a = square(10, 10, 0, 10, 0, 5)
    assert metrics.iou(a, a) == 1.0
    assert metrics.iou(a, ~a) == 0.0
    b = square(10, 10, 0, 10, 0, 10)
    assert metrics.iou(a, b) == pytest.approx(0.5)
    c = square(10, 10, 0, 10, 3, 8)  # 2 columns shared out of 8 covered
    assert metrics.iou(a, c) == pytest.approx(2 / 8)


def test_iou_of_empty_masks_is_zero():
    z = np.zeros((4, 4), bool)
    assert metrics.iou(z, z) == 0.0


def test_mirror_iou():
    sym = square(20, 20, 2, 18, 5, 15)
    assert metrics.mirror_iou(sym, 10) == 1.0
    lop = square(20, 20, 2, 18, 5, 10)  # only the left half drawn
    assert metrics.mirror_iou(lop, 10) == 0.0


def test_area_mm2():
    assert metrics.area_mm2(250, 1000.0) == pytest.approx(250.0)
    assert metrics.area_mm2(10, 500.0) == pytest.approx(40.0)


def test_blobs_size_filter_and_order():
    diff = np.zeros((40, 40), bool)
    diff[1:4, 1:4] = True  # 9 px, below the threshold
    diff[10:20, 10:20] = True
    diff[25:30, 25:35] = True
    found = metrics.blobs(diff, min_area_px=20)
    assert [a for _, a in found] == [100, 50]
    assert found[0][0][15, 15] and not found[0][0][27, 27]


def test_blobs_are_eight_connected():
    diff = np.zeros((6, 6), bool)
    diff[1, 1] = diff[2, 2] = diff[3, 3] = True  # a diagonal: one blob, not three
    assert [a for _, a in metrics.blobs(diff, 1)] == [3]


def test_budget():
    assert metrics.within_budget(4470, 2000, 9000)
    assert not metrics.within_budget(9001, 2000, 9000)
    assert not metrics.within_budget(1500, 2000, 9000)


CUBE = [
    [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],   # bottom, normal -z
    [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],   # top, normal +z
    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],   # front, normal -y
    [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],   # back, normal +y
    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],   # left, normal -x
    [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],   # right, normal +x
]


def test_signed_volume_detects_inverted_shell():
    assert metrics.signed_volume(CUBE) == pytest.approx(1.0)
    flipped = [list(reversed(f)) for f in CUBE]
    assert metrics.signed_volume(flipped) == pytest.approx(-1.0)


def test_accounting_flags_a_part_that_is_neither_built_nor_painted():
    C = dict(garments={'vest': dict(kind='torso'), 'belt': dict(kind='torso', painted=True),
                       'scarf': dict(kind='accessory')},
             hair={'painted': []}, face={'features': [dict(kind='eye'), dict(kind='mouth')]})
    objs = {'Body': dict(cls='skin', painted=''), 'Vest': dict(cls='vest', painted=''),
            'Hair': dict(cls='hair', painted='')}
    acc, _ = judge.accounting(C, objs)
    status = {a['feature']: a['status'] for a in acc}
    assert status['part: vest'] == 'built'
    assert status['part: belt'] == 'painted'
    assert status['part: scarf'] == 'MISSING'
    assert status['hair cap'] == 'built'


def test_a_convex_box_has_no_inward_fold_and_an_inner_corner_has_one():
    # a cube edge: the neighbour's centre lies behind the face
    bottom, front = CUBE[0], CUBE[2]

    def normal_centre(f):
        p = np.asarray(f, float)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        return n / np.linalg.norm(n), p.mean(0)
    n1, c1 = normal_centre(bottom)
    n2, c2 = normal_centre(front)
    assert not metrics.folds_inward(n1, c1, n2, c2, 2.0)
    # an inner corner: a floor meeting a wall that rises from it
    floor_n, floor_c = (0, 0, 1), (0, -0.5, 0)
    wall_n, wall_c = (0, -1, 0), (0, 0, 0.5)
    assert metrics.folds_inward(floor_n, floor_c, wall_n, wall_c, 2.0)
    assert not metrics.folds_inward((0, 0, 1), (0, 0, 0), (0, 0, 1), (1, 0, 0), 2.0)
