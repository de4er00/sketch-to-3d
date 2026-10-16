"""Carry a boundary loop of the body along a path to grow a limb out of the trunk."""
import math
import bmesh
from mathutils import Vector


def rmf(path):
    """Rotation-minimizing frames by double reflection."""
    n = len(path)
    T = []
    for i in range(n):
        a = path[max(0, i - 1)]
        b = path[min(n - 1, i + 1)]
        t = b - a
        T.append(t.normalized() if t.length > 1e-9 else Vector((0, 0, -1)))
    up = Vector((0, 0, 1))
    if abs(T[0].dot(up)) > 0.9:
        up = Vector((0, 1, 0))
    U = [(up - T[0] * up.dot(T[0])).normalized()]
    for i in range(n - 1):
        v1 = path[i + 1] - path[i]
        c1 = v1.dot(v1)
        if c1 < 1e-18:
            U.append(U[-1])
            continue
        uL = U[-1] - v1 * (2.0 / c1) * v1.dot(U[-1])
        tL = T[i] - v1 * (2.0 / c1) * v1.dot(T[i])
        v2 = T[i + 1] - tL
        c2 = v2.dot(v2)
        u = uL - v2 * (2.0 / c2) * v2.dot(uL) if c2 > 1e-18 else uL
        U.append((u - T[i + 1] * u.dot(T[i + 1])).normalized())
    V = [T[i].cross(U[i]).normalized() for i in range(n)]
    return T, U, V


def catmull(pts, n):
    P = [Vector(q) for q in pts]
    ext = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out, segs = [], len(P) - 1
    for i in range(n):
        u = i / float(n - 1) * segs
        j = min(int(u), segs - 1)
        t = u - j
        p0, p1, p2, p3 = ext[j], ext[j + 1], ext[j + 2], ext[j + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    return out


def profile(keys, n):
    ts = [k[0] for k in keys]
    rs = [k[1] for k in keys]
    out = []
    for i in range(n):
        t = i / float(n - 1)
        for j in range(len(ts) - 1):
            if ts[j] <= t <= ts[j + 1]:
                u = (t - ts[j]) / max(1e-9, ts[j + 1] - ts[j])
                u = u * u * (3 - 2 * u)  # smoothstep
                out.append(rs[j] + (rs[j + 1] - rs[j]) * u)
                break
        else:
            out.append(rs[-1])
    return out


def extrude_loop(bm, loop, path, radii, round_by=3, flat=None):
    """Carry `loop` along `path`, rounding it to a circle over `round_by` rings."""
    na = len(loop)
    T, U, V = rmf(path)
    c0 = sum((v.co for v in loop), Vector()) / na
    loc = []
    for v in loop:
        d = v.co - c0
        loc.append((d.dot(U[0]), d.dot(V[0])))
    r0 = max(1e-6, sum(math.hypot(*p) for p in loc) / na)
    loc = [(u / r0, w / r0) for (u, w) in loc]
    # phase the circle to the opening so no ring twists
    ph = math.atan2(loc[0][1], loc[0][0])
    sgn = 1.0
    if na > 2:
        a1 = math.atan2(loc[1][1], loc[1][0]) - ph
        sgn = 1.0 if math.sin(a1) >= 0 else -1.0
    circ = [(math.cos(ph + sgn * 2 * math.pi * k / na),
             math.sin(ph + sgn * 2 * math.pi * k / na)) for k in range(na)]

    rings = [list(loop)]
    for i in range(1, len(path)):
        t = min(1.0, i / float(max(1, round_by)))
        t = t * t * (3 - 2 * t)
        f = flat[i] if flat else 1.0
        row = []
        for k in range(na):
            u = loc[k][0] * (1 - t) + circ[k][0] * t
            w = loc[k][1] * (1 - t) + circ[k][1] * t
            row.append(bm.verts.new(path[i] + U[i] * (u * radii[i])
                                    + V[i] * (w * radii[i] * f)))
        rings.append(row)
    for a, b in zip(rings, rings[1:]):
        for k in range(na):
            j = (k + 1) % na
            bm.faces.new([a[k], a[j], b[j], b[k]])
    return rings


def open_patch(bm, rings, rows, cols, nc, scrap=None):
    """Delete a rows x cols patch from a ring grid and return its boundary loop."""
    dead = []
    for i in rows:
        for k in cols:
            a, b = rings[i], rings[i + 1]
            j = (k + 1) % nc
            for f in bm.faces:
                if set(f.verts) == {a[k], a[j], b[j], b[k]}:
                    dead.append(f)
    bmesh.ops.delete(bm, geom=dead, context='FACES_ONLY')
    i0, i1 = rows[0], rows[-1] + 1
    k0, k1 = cols[0], cols[-1] + 1
    top = [rings[i0][k] for k in range(k0, k1 + 1)]
    right = [rings[i][k1 % nc] for i in range(i0 + 1, i1 + 1)]
    bot = [rings[i1][k] for k in range(k1 - 1, k0 - 1, -1)]
    left = [rings[i][k0] for i in range(i1 - 1, i0, -1)]
    loop = top + right + bot + left
    # FACES_ONLY leaves interior verts as wire, the caller deletes them once `rings` is done with
    keep = set(loop)
    if scrap is not None:
        for i in range(i0, i1 + 1):
            for k in range(k0, k1 + 1):
                v = rings[i][k % nc]
                if v not in keep:
                    scrap.append(v)
    return loop
