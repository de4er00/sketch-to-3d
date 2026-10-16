import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sketch3d.read.protocol import colour_bands  # noqa: E402

LABEL = (255, 0, 255)


def sheet(art_rows):
    """A 200 x 60 sheet with one part in rows 20..180, art colour per row from art_rows(row)."""
    img = np.full((200, 60, 3), 255.0, np.float32)
    art = np.full((200, 60, 3), 255.0, np.float32)
    img[20:180, 10:50] = LABEL
    for r in range(20, 180):
        art[r, 10:50] = art_rows(r)
    return img, art


def bands(img, art):
    return colour_bands(img, art, [(0, 0, LABEL)], {0: 'sock'}, 1, dict(c0=0, c1=60), floor=200, ppm=1000.0, log=lambda *a: None)


def test_repeating_stripes_become_bands():
    img, art = sheet(lambda r: (20, 20, 20) if (r // 20) % 2 == 0 else (230, 230, 230))
    out = bands(img, art)
    assert 'sock' in out
    cols = [tuple(b[2]) for b in out['sock']]
    assert len(cols) >= 4
    assert cols[0] != cols[1] and cols[0] == cols[2]


def test_a_shaded_part_is_one_colour():
    # a gradient from light to slightly darker is shading, not stripes
    img, art = sheet(lambda r: (200 - (r - 20) // 4, 150, 120))
    assert bands(img, art) == {}


def test_one_dark_band_is_not_stripes():
    img, art = sheet(lambda r: (20, 20, 20) if 90 <= r < 110 else (230, 180, 160))
    assert bands(img, art) == {}


def test_two_figures_touching_at_a_hand_are_split():
    from sketch3d.read.protocol import split_touching
    m = np.zeros((300, 400), bool)
    m[20:280, 20:80] = True  # a standing figure
    m[120:130, 80:200] = True  # its arm, touching the next figure's hand
    m[20:280, 200:260] = True  # the next figure
    m[20:280, 300:360] = True  # a third, apart
    figs = []
    for a, b in ((20, 260), (300, 360)):
        mm = np.zeros_like(m)
        mm[:, a:b] = m[:, a:b]
        ys, xs = np.nonzero(mm)
        figs.append(dict(c0=int(xs.min()), c1=int(xs.max()) + 1, r0=int(ys.min()), r1=int(ys.max()), cx=float(xs.mean()), area=int(len(ys)), mask=mm))
    out = split_touching(figs, log=lambda *a: None)
    assert len(out) == 3
    assert sorted(f['c1'] - f['c0'] < 200 for f in out) == [True, True, True]


def test_marks_inside_a_part_are_found_and_overlaps_are_not():
    from sketch3d.read.protocol import detail_marks
    H, W = 200, 300
    img = np.full((H, W, 3), 255.0, np.float32)
    art = np.full((H, W, 3), 255.0, np.float32)
    img[40:160, 40:260] = LABEL  # one part: a dark visor
    art[40:160, 40:260] = (30, 30, 40)
    art[80:110, 80:110] = (90, 240, 250)  # an eye drawn on it
    art[70:150, 200:290] = (240, 240, 240)  # another part drawn over its edge
    panels = [dict(name='front', c0=0, c1=W)]
    lab, marks = detail_marks(img, art, [(0, 0, LABEL)], {0: 'visor'}, 1, panels, 1000.0,
                              {'visor': [30, 30, 40]}, set(), log=lambda *a: None)
    assert len(marks) == 1
    assert marks[0]['view'] == 'front' and marks[0]['rgb'] == [90, 240, 250]
    assert lab[95, 95] == 1 and lab[100, 230] == 0


def test_art_standing_on_a_ground_line_is_not_shrunk():
    from sketch3d.read.protocol import register_art
    H, W = 300, 600
    mask = np.zeros((H, W), bool)
    mask[40:271, 100:180] = True  # the part map's figure, down to row 270
    figs = [dict(c0=100, c1=180, r0=40, r1=270, mask=mask)]
    art = np.full((H, W, 3), 255.0, np.float32)
    art[40:271, 100:180] = (200, 120, 90)  # the same figure on the art
    art[266:271, 10:590] = (90, 90, 90)  # a ground line under it
    logs = []
    out = register_art(art, figs, (H, W, 3), log=logs.append)
    assert 'scale 1.000, shift +0 px, +0 px' in logs[-1]
    assert (out[150, 140] == (200, 120, 90)).all()
