"""Row runs, central runs and station rows of a figure, in sheet pixels."""
import numpy as np


def row_runs(line):
    """Half-open runs [(a, b)] of True in a 1-D bool array."""
    if not line.any():
        return []
    d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0]
    return list(zip(st.tolist(), en.tolist()))


def runs_table(mask, c0, c1, r0=0, r1=None, min_w=1):
    H = mask.shape[0]
    r1 = H if r1 is None else r1
    out = {}
    sub = mask[:, c0:c1]
    for r in range(r0, r1):
        rr = [(a + c0, b + c0) for a, b in row_runs(sub[r]) if b - a >= min_w]
        if rr:
            out[r] = rr
    return out


def central(runs, mid):
    """The run containing `mid`, else the nearest one."""
    if not runs:
        return None
    for a, b in runs:
        if a <= mid < b:
            return (a, b)
    return min(runs, key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - mid))


def smooth1(vals, k=3):
    ks = sorted(vals)
    arr = np.array([vals[r] for r in ks], float)
    if len(arr) < 2 * k + 1:
        return dict(vals)
    pad = np.pad(arr, k, mode='edge')
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.convolve(pad, ker, mode='valid')
    return {r: float(v) for r, v in zip(ks, sm)}


def argmin_in(vals, r0, r1):
    ks = [r for r in vals if r0 <= r <= r1]
    if not ks:
        return None
    return min(ks, key=lambda r: vals[r])


def argmax_in(vals, r0, r1):
    ks = [r for r in vals if r0 <= r <= r1]
    if not ks:
        return None
    return max(ks, key=lambda r: vals[r])


class Frame:
    """Sheet pixels <-> metres, +X the character's left, +Y its back, z up from the floor."""

    def __init__(self, ppm, floor_row):
        self.ppm, self.floor = float(ppm), float(floor_row)

    def z(self, row):
        return (self.floor - (row + 0.5)) / self.ppm

    def row(self, z):
        return int(round(self.floor - z * self.ppm - 0.5))

    def mm(self, px):
        return px / self.ppm * 1000.0

    def m(self, px):
        return px / self.ppm


def dp_rows(rows, curves, tol_px, min_gap=6, corner_deg=10.0):
    """Pick station rows: Douglas-Peucker over all curves, then split at corners sharper than corner_deg."""
    rr = [r for r in rows if all(r in c for c in curves)]
    if len(rr) < 3:
        return rr
    keep = {0, len(rr) - 1}

    def dev(lo, hi):
        worst, wi = 0.0, None
        for i in range(lo + 1, hi):
            r = rr[i]
            t = (i - lo) / float(hi - lo)
            e = max(abs(c[r] - (c[rr[lo]] + (c[rr[hi]] - c[rr[lo]]) * t)) for c in curves)
            if e > worst:
                worst, wi = e, i
        return worst, wi

    def rec(lo, hi):
        if hi - lo < min_gap:
            return
        worst, wi = dev(lo, hi)
        if worst > tol_px and wi is not None:
            keep.add(wi)
            rec(lo, wi)
            rec(wi, hi)

    rec(0, len(rr) - 1)
    for _ in range(40):
        ks = sorted(keep)
        worst, wi = 0.0, None
        for j in range(1, len(ks) - 1):
            a, b, c = rr[ks[j - 1]], rr[ks[j]], rr[ks[j + 1]]
            for cv in curves:
                v1 = np.array([cv[b] - cv[a], b - a], float)
                v2 = np.array([cv[c] - cv[b], c - b], float)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                ang = np.degrees(np.arccos(np.clip(v1.dot(v2) / (n1 * n2), -1, 1)))
                if ang > worst:
                    worst, wi = ang, j
        if wi is None or worst <= corner_deg:
            break
        j = wi
        left, right = ks[j] - ks[j - 1], ks[j + 1] - ks[j]
        if max(left, right) < 2 * min_gap:
            break
        if left >= right:
            keep.add((ks[j] + ks[j - 1]) // 2)
        else:
            keep.add((ks[j] + ks[j + 1]) // 2)
    return [rr[i] for i in sorted(keep)]
