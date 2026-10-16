"""Numeric helpers of the judge, kept free of Blender for tests."""
import numpy as np


def iou(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def mirror_iou(mask, mid):
    mask = np.asarray(mask, bool)
    half = int(min(mid, mask.shape[1] - mid))
    left = mask[:, int(mid) - half:int(mid)]
    right = mask[:, int(mid):int(mid) + half][:, ::-1]
    return iou(left, right)


def area_mm2(area_px, ppm):
    return area_px / ppm / ppm * 1e6


def blobs(diff, min_area_px):
    """Return 8-connected components of at least min_area_px pixels as (mask, area_px), largest first."""
    from scipy import ndimage as ndi
    comp, n = ndi.label(np.asarray(diff, bool), structure=np.ones((3, 3), int))
    out = []
    for k in range(1, n + 1):
        m = comp == k
        area = int(m.sum())
        if area >= min_area_px:
            out.append((m, area))
    return sorted(out, key=lambda t: -t[1])


def within_budget(tris, lo, hi):
    return lo <= tris <= hi


def signed_volume(polys):
    """Signed volume of a closed mesh, positive when faces wind counter-clockwise from outside."""
    v = 0.0
    for poly in polys:
        p = np.asarray(poly, float)
        for i in range(1, len(p) - 1):
            v += float(np.dot(p[0], np.cross(p[i], p[i + 1]))) / 6.0
    return v


def folds_inward(n1, c1, n2, c2, min_deg):
    """True when two faces sharing an edge fold inward by more than min_deg."""
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    cosang = float(np.clip(np.dot(n1, n2) / max(1e-12, np.linalg.norm(n1) * np.linalg.norm(n2)), -1.0, 1.0))
    if np.degrees(np.arccos(cosang)) <= min_deg:
        return False
    return float(np.dot(np.asarray(c2, float) - np.asarray(c1, float), n1)) > 0
