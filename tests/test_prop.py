import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sketch3d.read.prop import front_side_of  # noqa: E402


def sheet(lock_right):
    """A box (1) with a lock plate (2) on its front, seen at one edge in the side view."""
    lab = np.zeros((200, 400), np.int16)
    lab[50:150, 20:180] = 1  # front view: the box
    lab[90:110, 90:110] = 2  # the plate drawn on it
    lab[50:150, 240:340] = 1  # side view: the box
    if lock_right:
        lab[90:110, 340:346] = 2
    else:
        lab[90:110, 234:240] = 2
    classes = [dict(id=1, name='box'), dict(id=2, name='lock_plate')]
    panels = [dict(name='front', c0=0, c1=200, r0=50, r1=149, cx=100.0, cy=100.0),
              dict(name='side', c0=220, c1=360, r0=50, r1=149, cx=290.0, cy=100.0)]
    return lab, classes, panels


def test_the_front_is_found_from_a_part_drawn_on_the_front_face():
    lab, classes, panels = sheet(lock_right=True)
    assert front_side_of(lab, classes, panels, log=lambda *a: None)[0] == -1
    lab, classes, panels = sheet(lock_right=False)
    assert front_side_of(lab, classes, panels, log=lambda *a: None)[0] == 1
