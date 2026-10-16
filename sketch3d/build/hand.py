"""Parametric low-poly hand (about 200 triangles), scaled to the character by the body stage.

Local frame: fingers along -Z, back of the hand +X, index edge +Y, origin at the wrist ring centre.
"""
import bmesh
import math
from mathutils import Vector, Matrix

HW = 0.258  # head width in m, the proportions below are fractions of it

P = dict(
    length      = 0.360,
    knuckle_t   = 0.47,
    web_t       = 0.495,
    thenar_t    = 0.22,
    midpalm_t   = 0.38,

    # (t, half-extent) tables in fractions of HW
    y_front = ((0.00, 0.0779), (0.09, 0.0907), (0.17, 0.1035),
               (0.38, 0.1060), (0.50, 0.1070), (1.00, 0.1101)),
    y_back  = ((0.00, 0.0779), (0.09, 0.0876), (0.17, 0.1004),
               (0.24, 0.1101), (0.32, 0.1198), (0.39, 0.1295),
               (0.47, 0.1360), (0.62, 0.1554), (0.70, 0.1620),
               (0.80, 0.1686), (1.00, 0.1686)),

    u_col   = (-1.000, -0.539, 0.064, 0.603, 1.000),

    x_half  = ((0.00, 0.0519), (0.22, 0.0504), (0.38, 0.0446),
               (0.50, 0.0395), (0.58, 0.0340), (1.00, 0.0248)),

    finger_tip_t    = (0.971, 1.030, 0.991, 0.862),
    # per finger: ft, gap to the neighbour (m), thickness scale, width scale
    finger_seg      = ((0.16, 0.0026, 0.97, 0.93),
                       (0.50, 0.0024, 0.90, 0.84),
                       (0.80, 0.0024, 0.80, 0.71),
                       (0.92, 0.0024, 0.66, 0.78)),
    finger_tip_dome = 0.20,
    finger_curl     = 15.0,  # deg
    finger_tip_curl = 11.0,

    thumb_root_t = (0.22, 0.38),
    thumb_len    = 0.182,
    thumb_yaw    = 44.0,  # deg
    thumb_pitch  = 27.0,
    thumb_seg    = ((0.20, 0.76), (0.62, 0.68), (1.00, 0.44)),

    dome         = 0.40,
    cup          = 0.42,
    thenar_bulge = 0.60,
    thenar_y     = 0.0140,
    knuckle_rise = 0.30,
    edge_round   = 0.66,
    cap_wrist    = True,  # False when the arm bridges onto the wrist ring
)


def lerp_tab(tab, t):
    t = max(tab[0][0], min(tab[-1][0], t))
    for i in range(len(tab) - 1):
        a, va = tab[i]
        b, vb = tab[i + 1]
        if a <= t <= b:
            f = 0.0 if b == a else (t - a) / (b - a)
            return va + (vb - va) * f
    return tab[-1][1]


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_hand(over=None):
    """Build one left hand in the local frame, return (bm, wrist_ring_verts)."""
    p = dict(P)
    p.update(over or {})
    L = p['length'] * HW
    U = p['u_col']
    N = 2 * len(U)
    kt = p['knuckle_t']
    zk = -L * kt

    bm = bmesh.new()
    lay = bm.verts.layers.int.new('sc_wrist')

    def section(t):
        yf = lerp_tab(p['y_front'], t) * HW
        yb = lerp_tab(p['y_back'], t) * HW
        xh = lerp_tab(p['x_half'], t) * HW
        z = -L * t
        out = []
        for sgn in (+1, -1):
            idx = range(len(U)) if sgn > 0 else reversed(range(len(U)))
            for j in idx:
                u = U[j]
                # u runs linearly over the whole asymmetric span, so fingers do not slide as the pinky flares
                y = yf - (u + 1.0) * 0.5 * (yf + yb)
                x = sgn * xh
                x *= 1.0 - (1.0 - p['edge_round']) * abs(u) ** 3
                shape = 1.0 - u * u
                if sgn > 0:
                    x += xh * p['dome'] * shape
                    d = abs(t - kt)
                    if d < 0.10:
                        x += xh * p['knuckle_rise'] * shape * (1.0 - d / 0.10)
                else:
                    palmness = 1.0 - smooth((t - kt) / (1.0 - kt))
                    x -= xh * p['cup'] * shape * palmness
                    tb = math.exp(-((t - p['thenar_t'] - 0.06) / 0.11) ** 2)
                    x -= xh * p['thenar_bulge'] * tb * max(0.0, -u) ** 0.7
                    y += p['thenar_y'] * HW * tb * max(0.0, -u) ** 1.4
                out.append(Vector((x, y, z)))
        return out

    def curl(vs, deg):
        R = Matrix.Rotation(math.radians(deg), 3, 'Y')
        piv = Vector((0.0, 0.0, zk))
        return [piv + R @ (v - piv) for v in vs]

    faces = []

    def quad(a, b, c, d):
        try:
            faces.append(bm.faces.new((a, b, c, d)))
        except ValueError:
            pass

    stations = [0.0, p['thenar_t'], p['midpalm_t'], kt, p['web_t']]
    rings = []
    for t in stations:
        vs = section(t)
        if t > kt:
            vs = curl(vs, p['finger_curl'] * (t - kt) / (1.0 - kt))
        rings.append([bm.verts.new(v) for v in vs])
    bm.verts.ensure_lookup_table()
    for v in rings[0]:
        v[lay] = 1

    for r0, r1 in zip(rings[:-1], rings[1:]):
        for i in range(N):
            j = (i + 1) % N
            quad(r0[i], r0[j], r1[j], r1[i])

    web = rings[-1]
    wt = p['web_t']
    for k in range(4):
        bi, bj = k, k + 1
        pi, pj = 9 - k, 8 - k
        prev = [web[bi], web[bj], web[pj], web[pi]]
        ttip = p['finger_tip_t'][k]
        for ft, gap, sh, wsc in p['finger_seg']:
            t = wt + (ttip - wt) * ft
            full = section(t)
            qv = [full[bi], full[bj], full[pj], full[pi]]
            cen = sum(qv, Vector()) / 4.0
            # interior edges inset by half the gap, thickness tapers to a round tip
            lo_in, hi_in = (k > 0), (k < 3)
            nq = []
            for v, is_hi in zip(qv, (False, True, True, False)):
                y = cen.y + (v.y - cen.y) * wsc
                dy = 0.0
                if is_hi and hi_in:
                    dy = +gap * 0.5
                elif (not is_hi) and lo_in:
                    dy = -gap * 0.5
                nq.append(Vector((cen.x + (v.x - cen.x) * sh, y + dy, v.z)))
            qv = nq
            ang = (p['finger_curl'] + p['finger_tip_curl'] * ft) * \
                  (t - kt) / (1.0 - kt)
            qv = curl(qv, ang)
            nv = [bm.verts.new(v) for v in qv]
            for i in range(4):
                j = (i + 1) % 4
                quad(prev[i], prev[j], nv[j], nv[i])
            prev = nv
        rim = sum((v.co for v in prev), Vector()) / 4.0
        for v in prev:
            v.co += (rim - v.co) * p['finger_tip_dome']
        tipf = section(ttip)
        tipq = [tipf[bi], tipf[bj], tipf[pj], tipf[pi]]
        tipc = sum(tipq, Vector()) / 4.0
        tipc = curl([tipc], (p['finger_curl'] + p['finger_tip_curl'])
                    * (ttip - kt) / (1.0 - kt))[0]
        apex = bm.verts.new(tipc)
        for i in range(4):
            j = (i + 1) % 4
            try:
                bm.faces.new([prev[i], prev[j], apex])
            except ValueError:
                pass

    rb, rc = rings[1], rings[2]
    root = [rb[N - 2], rb[N - 1], rc[N - 1], rc[N - 2]]
    dead = [f for f in bm.faces if set(f.verts) == set(root)]
    bmesh.ops.delete(bm, geom=dead, context='FACES_ONLY')
    cen = sum((v.co for v in root), Vector()) / 4.0
    yaw = math.radians(p['thumb_yaw'])
    pit = math.radians(p['thumb_pitch'])
    axis = Vector((-math.sin(pit),
                   math.cos(pit) * math.sin(yaw),
                   -math.cos(pit) * math.cos(yaw))).normalized()
    tl = p['thumb_len'] * HW
    prev = root
    for ft, sc in p['thumb_seg']:
        tgt = cen + axis * (tl * ft)
        nv = [bm.verts.new(tgt + (v.co - cen) * sc) for v in root]
        for i in range(4):
            j = (i + 1) % 4
            quad(prev[i], prev[j], nv[j], nv[i])
        prev = nv
    quad(prev[0], prev[1], prev[2], prev[3])

    if p['cap_wrist']:
        try:
            bm.faces.new(list(reversed(rings[0])))
        except ValueError:
            pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm, rings[0]
