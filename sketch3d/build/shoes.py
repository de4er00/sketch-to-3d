"""Build shoes: an upper around the foot stub and a sole slab per foot.

    blender -b --factory-startup --python sketch3d/build/shoes.py -- <char.json> <body.json> <in.blend> <out.blend>
"""
import bpy, bmesh, os, sys, json, math
from mathutils import Vector

D = os.path.dirname(os.path.abspath(__file__))
if D not in sys.path:
    sys.path.insert(0, D)
from shell import BodyRef, emit                                    # noqa: E402

COLS = 12
CLEAR = 0.0022
THICK = 0.0028
RIM_IN = 0.0030
RIM_DROP = 0.0100
MOUTH_EXTRA = 0.0030
SOLE_GROW = 0.0040
SOLE_FLARE = 0.0060
SOLE_FLARE_IN = 0.0020
TOE_SPRING = 0.0040
UPPER_FOOT = 0.0100  # the bottom ring's width is read this far above the sole top
# ring heights as fractions of ankle -> sole top
LEVELS = (0.00, 0.34, 0.58, 0.81, 1.00)
# superellipse exponent per ring: an ellipse at the mouth, blunt at the sole
EXPO = (2.05, 2.20, 2.35, 2.45, 2.55)


def st_at(rows, z, keys=('w_mm', 'd_mm', 'y_mid_mm', 'x_c_mm')):
    st = [s for s in rows if s.get('w_mm') is not None and s.get('d_mm') is not None]
    if not st:
        return None
    st = sorted(st, key=lambda s: s['z'])
    zs = [s['z'] for s in st]
    z = min(max(z, zs[0]), zs[-1])
    for i in range(len(zs) - 1):
        if zs[i] <= z <= zs[i + 1]:
            t = (z - zs[i]) / max(1e-9, zs[i + 1] - zs[i])
            return dict((k, (st[i][k] + (st[i + 1][k] - st[i][k]) * t)) for k in keys)
    return dict((k, st[-1][k]) for k in keys)


def drawn_section(C, z):
    """Drawn section at z averaged over both shoes, in the right shoe's frame (x <= 0)."""
    got = []
    for side in ('R', 'L'):
        s = st_at(C['stations'].get('shoe_' + side) or [], z)
        if s:
            got.append(s)
    if not got:
        return None
    n = float(len(got))
    w = sum(s['w_mm'] for s in got) / n / 1000.0
    d = sum(s['d_mm'] for s in got) / n / 1000.0
    ym = sum(s['y_mid_mm'] for s in got) / n / 1000.0
    xc = -sum(abs(s['x_c_mm']) for s in got) / n / 1000.0
    return dict(cx=xc, cy=ym, rxn=0.5 * w, rxp=0.5 * w, ryn=0.5 * d, ryp=0.5 * d)


class FootRef:
    """Body section under the shoe, interpolated between the ankle ring and the foot stub rings."""

    def __init__(self, body, M, side):
        rings = [M['leg'][side][-1], M['foot'][side][0], M['foot'][side][1]]
        self.polys = [[Vector((body.co[i].x, body.co[i].y)) for i in r] for r in rings]
        self.zs = [sum(body.co[i].z for i in r) / len(r) for r in rings]

    def at(self, z):
        if z >= self.zs[0]:
            return self.polys[0]
        if z <= self.zs[-1]:
            return self.polys[-1]
        for i in range(len(self.zs) - 1):
            if self.zs[i + 1] <= z <= self.zs[i]:
                t = (self.zs[i] - z) / max(1e-9, self.zs[i] - self.zs[i + 1])
                a, b = self.polys[i], self.polys[i + 1]
                return [a[k] + (b[k] - a[k]) * t for k in range(len(a))]
        return self.polys[-1]


def enclose(sec, pts, p, cl, iters=10):
    """Grow the four quadrant radii until `pts` fit inside with margin `cl`."""
    grown = 0.0

    def bump(k, v):
        nonlocal grown
        if v > sec[k]:
            grown = max(grown, v - sec[k])
            sec[k] = v
    bump('rxn', sec['cx'] - (min(q.x for q in pts) - cl))
    bump('rxp', (max(q.x for q in pts) + cl) - sec['cx'])
    bump('ryn', sec['cy'] - (min(q.y for q in pts) - cl))
    bump('ryp', (max(q.y for q in pts) + cl) - sec['cy'])
    for _ in range(iters):
        need = {}
        for q in pts:
            dx, dy = q.x - sec['cx'], q.y - sec['cy']
            kx = 'rxp' if dx >= 0 else 'rxn'
            ky = 'ryp' if dy >= 0 else 'ryn'
            a = (abs(dx) + cl) / sec[kx]
            b = (abs(dy) + cl) / sec[ky]
            f = a ** p + b ** p
            if f <= 1.0:
                continue
            s = min(f ** (1.0 / p), 1.08)
            need[kx] = max(need.get(kx, 1.0), s)
            need[ky] = max(need.get(ky, 1.0), s)
        if not need:
            break
        for k, s in need.items():
            bump(k, sec[k] * s)
    return grown


def ring_points(sec, p, z, taper=0.0, cols=COLS):
    """Superellipse loop from the toe (-y), counter-clockwise from above, `taper` narrows the heel."""
    out = []
    for k in range(cols):
        ph = 2.0 * math.pi * k / cols
        ux, uy = math.sin(ph), -math.cos(ph)
        sx = math.copysign(abs(ux) ** (2.0 / p), ux) if abs(ux) > 1e-12 else 0.0
        sy = math.copysign(abs(uy) ** (2.0 / p), uy) if abs(uy) > 1e-12 else 0.0
        r = sec['rxp'] if sx >= 0 else sec['rxn']
        if taper and sy > 0.0:
            r *= 1.0 - taper * sy * sy
        out.append(Vector((sec['cx'] + sx * r,
                           sec['cy'] + sy * (sec['ryp'] if sy >= 0 else sec['ryn']), z)))
    return out


def inside(poly, q):
    n, c = len(poly), False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a.y > q.y) != (b.y > q.y):
            if q.x < a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y):
                c = not c
    return c


def fit_taper(sec, p, z, pts, cands=(0.22, 0.17, 0.12, 0.07, 0.0)):
    for t in cands:
        ring = ring_points(sec, p, z, t)
        if all(inside(ring, q) for q in pts):
            return t
    return None


def grow(sec, m):
    s = dict(sec)
    for k in ('rxn', 'rxp', 'ryn', 'ryp'):
        s[k] = sec[k] + m
    return s


def _mk(bm, ring, flip):
    return [bm.verts.new(Vector((-v.x if flip else v.x, v.y, v.z))) for v in ring]


def _face(bm, vs, flip):
    try:
        bm.faces.new(vs[::-1] if flip else vs)
    except ValueError:
        pass


def make_upper(rings, rim, sleeve, side):
    flip = (side == 'L')
    bm = bmesh.new()
    rows = [_mk(bm, r, flip) for r in rings]
    RIM = _mk(bm, rim, flip)
    SLV = _mk(bm, sleeve, flip)
    n = COLS
    for i in range(len(rows) - 1):
        A, B = rows[i], rows[i + 1]
        for k in range(n):
            j = (k + 1) % n
            _face(bm, [A[k], B[k], B[j], A[j]], flip)
    for k in range(n):
        j = (k + 1) % n
        _face(bm, [rows[0][k], rows[0][j], RIM[j], RIM[k]], flip)
    for k in range(n):
        j = (k + 1) % n
        _face(bm, [RIM[k], RIM[j], SLV[j], SLV[k]], flip)
    return emit(SHOE_CLS[0].capitalize() + '_' + side, bm,
                dict(s3d_part='shoe', s3d_class=SHOE_CLS[0], s3d_kind='legs', s3d_side=side))


def _ladder(bm, ring, flip, up):
    n = len(ring)
    for i in range(n // 2 - 1):
        q = [ring[i], ring[i + 1], ring[n - 2 - i], ring[n - 1 - i]]
        _face(bm, q if up else q[::-1], flip)


def make_sole(top, bottom, side):
    flip = (side == 'L')
    bm = bmesh.new()
    T = _mk(bm, top, flip)
    B = _mk(bm, bottom, flip)
    n = COLS
    for k in range(n):
        j = (k + 1) % n
        _face(bm, [T[k], B[k], B[j], T[j]], flip)
    _ladder(bm, T, flip, True)
    _ladder(bm, B, flip, False)
    return emit(SOLE_CLS[0].capitalize() + '_' + side, bm,
                dict(s3d_part='shoe', s3d_class=SOLE_CLS[0], s3d_kind='legs', s3d_side=side))


SHOE_CLS = ['sneaker']
SOLE_CLS = ['sole']


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, out_blend = argv[:4]
    C = json.load(open(char_path))
    M = json.load(open(map_path))
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    body = BodyRef(bpy.data.objects['Body'])
    lm = C['landmarks']
    if C['roles'].get('shoe'):
        SHOE_CLS[0] = C['roles']['shoe'][0]
    if C['roles'].get('sole'):
        SOLE_CLS[0] = C['roles']['sole'][0]
    z_ank = 0.5 * (lm['ankle_R']['z'] + lm['ankle_L']['z'])
    z_sole = 0.5 * (lm['sole_top_R']['z'] + lm['sole_top_L']['z'])
    print('[shoes] ankle z %.4f  sole_top z %.4f  (upper %.0f mm tall, sole %.0f mm)'
          % (z_ank, z_sole, (z_ank - z_sole) * 1000, z_sole * 1000))

    foot = dict((s, FootRef(body, M, s)) for s in ('R', 'L'))

    def body_pts(z):
        """Body section at z for both feet in the right shoe's frame, so the shoes mirror."""
        out = list(foot['R'].at(z))
        out += [Vector((-q.x, q.y)) for q in foot['L'].at(z)]
        return out

    ank = body_pts(z_ank)
    cx = 0.5 * (min(q.x for q in ank) + max(q.x for q in ank))
    cy = 0.5 * (min(q.y for q in ank) + max(q.y for q in ank))
    rx = 0.5 * (max(q.x for q in ank) - min(q.x for q in ank))
    ry = 0.5 * (max(q.y for q in ank) - min(q.y for q in ank))
    R = max((q - Vector((cx, cy))).length for q in ank)
    off = R * (1.0 - math.cos(math.pi / COLS)) + CLEAR + THICK + MOUTH_EXTRA
    mouth = dict(cx=cx, cy=cy, rxn=rx + off, rxp=rx + off, ryn=ry + off, ryp=ry + off)
    print('[shoes] mouth: ankle %.0f x %.0f mm + %.1f mm -> %.0f x %.0f mm'
          % (2000 * rx, 2000 * ry, 1000 * off, 2000 * mouth['rxn'], 2000 * mouth['ryn']))

    secs, zs, taps = [], [], []
    for i, f in enumerate(LEVELS):
        z = z_ank + (z_sole - z_ank) * f
        p = EXPO[i]
        sec = dict(mouth) if i == 0 else drawn_section(C, z)
        if sec is None:
            continue
        if i == len(LEVELS) - 1:
            # the sole_top row measures the sole's flare: take the upper's width 10 mm higher
            up = drawn_section(C, z + UPPER_FOOT)
            if up:
                sec['cx'], sec['rxn'], sec['rxp'] = up['cx'], up['rxn'], up['rxp']
        rmax = max(sec['rxn'], sec['rxp'], sec['ryn'], sec['ryp'])
        cl = CLEAR + rmax * (1.0 - math.cos(math.pi / COLS))
        pts = body_pts(z)
        g = enclose(sec, pts, p, cl)
        t = fit_taper(sec, p, z, pts, (0.0,) if i == 0 else (0.22, 0.17, 0.12, 0.07, 0.0))
        if t is None:
            print('[shoes] ring %d: the section does NOT contain the body even untapered' % i)
            t = 0.0
        secs.append(sec)
        zs.append(z)
        taps.append(t)
        print('[shoes] ring %d z %.4f  x %6.1f..%6.1f (w %5.1f)  y %6.1f..%6.1f (d %5.1f)'
              '  grown %4.1f mm  heel taper %.0f%%'
              % (i, z, 1000 * (sec['cx'] - sec['rxn']), 1000 * (sec['cx'] + sec['rxp']),
                 1000 * (sec['rxn'] + sec['rxp']), 1000 * (sec['cy'] - sec['ryn']),
                 1000 * (sec['cy'] + sec['ryp']), 1000 * (sec['ryn'] + sec['ryp']), 1000 * g, 100 * t))
    rings = [ring_points(secs[i], EXPO[i], zs[i], taps[i]) for i in range(len(secs))]

    mc = Vector((mouth['cx'], mouth['cy'], 0.0))
    rim = []
    for v in rings[0]:
        d = mc - Vector((v.x, v.y, 0.0))
        rim.append(v + d.normalized() * RIM_IN)
    sleeve = [Vector((v.x, v.y, v.z - RIM_DROP)) for v in rim]

    sole_top = grow(secs[-1], SOLE_GROW)
    sole_bot = dict(sole_top)
    for k, m in (('rxn', SOLE_FLARE), ('rxp', SOLE_FLARE_IN),
                 ('ryn', SOLE_FLARE), ('ryp', SOLE_FLARE)):
        sole_bot[k] = sole_top[k] + m
    print('[shoes] sole  top   x %6.1f..%6.1f (w %5.1f)  y %6.1f..%6.1f (d %5.1f)'
          % (1000 * (sole_top['cx'] - sole_top['rxn']), 1000 * (sole_top['cx'] + sole_top['rxp']),
             1000 * (sole_top['rxn'] + sole_top['rxp']), 1000 * (sole_top['cy'] - sole_top['ryn']),
             1000 * (sole_top['cy'] + sole_top['ryp']), 1000 * (sole_top['ryn'] + sole_top['ryp'])))
    top = ring_points(sole_top, EXPO[-1], zs[-1], taps[-1])
    bot = ring_points(sole_bot, EXPO[-1], 0.0, taps[-1])
    for k in range(COLS):
        c = math.cos(2.0 * math.pi * k / COLS)
        if c > 0.0:
            bot[k].z += TOE_SPRING * c ** 3

    made = []
    for side in ('R', 'L'):
        made.append(make_upper(rings, rim, sleeve, side))
        made.append(make_sole(top, bot, side))

    bad = 0
    for ob in made:
        me = ob.data
        tris = sum(len(p.vertices) - 2 for p in me.polygons)
        use = {}
        for pg in me.polygons:
            vs = list(pg.vertices)
            for a, b in zip(vs, vs[1:] + vs[:1]):
                k = (min(a, b), max(a, b))
                use[k] = use.get(k, 0) + 1
        bnd = sum(1 for k, v in use.items() if v == 1)
        nonman = sum(1 for k, v in use.items() if v > 2)
        inside = [v.index for v in me.vertices if body.in_skin(v.co.copy())]
        bad += len(inside)
        print('[shoes] %-10s %3d tris  %2d boundary / %d non-manifold edges  '
              '%d verts inside Body %s'
              % (ob.name, tris, bnd, nonman, len(inside), inside[:8]))
    if bad:
        print('[shoes] WARNING: %d shoe vertices sit inside the body' % bad)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[shoes] %d objects -> %s' % (len(made), out_blend))


if __name__ == '__main__':
    main()
