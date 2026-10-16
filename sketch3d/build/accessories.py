"""Build accessories: headphones, neck bands and studs by recipe, everything else via attach.py.

    blender -b --factory-startup --python sketch3d/build/accessories.py -- <char.json> <body.json> <in.blend> <out.blend>
"""
import bpy, bmesh, os, sys, json, math, re
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from shell import BodyRef, emit  # noqa: E402

CUP_DIAM = 0.055
CUP_THICK = 0.025
CUP_GAP = 0.004
TUBE_R = 0.007
STUD_BASE = 0.008
STUD_H = 0.008
EMBLEM_T = 0.008
CHAR_PATH = ['']
MAP_PATH = ['']
STUD_CLEAR = 0.0020


def neck_station(C):
    for s in C['stations']['trunk']:
        if 'neck' in (s.get('tags') or []):
            return s
    raise RuntimeError('no trunk station tagged "neck"')


def add_ngon_fan(bm, verts, flip=False):
    c = sum((v.co for v in verts), Vector()) / len(verts)
    cv = bm.verts.new(c)
    n = len(verts)
    seq = list(reversed(verts)) if flip else verts
    for i in range(n):
        a, b = seq[i], seq[(i + 1) % n]
        try:
            bm.faces.new([cv, a, b])
        except ValueError:
            pass


def build_cup(bm, centre, sign, n=12):
    """Closed 12-gon cylinder along X, sign +1 for the character's left."""
    r, half_t = CUP_DIAM / 2.0, CUP_THICK / 2.0
    inner_x = centre.x - sign * half_t
    outer_x = centre.x + sign * half_t
    inner, outer = [], []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        y = centre.y + r * math.cos(a)
        z = centre.z + r * math.sin(a)
        inner.append(bm.verts.new(Vector((inner_x, y, z))))
        outer.append(bm.verts.new(Vector((outer_x, y, z))))
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([inner[i], inner[j], outer[j], outer[i]])
        except ValueError:
            pass
    # flip so both caps wind outward
    add_ngon_fan(bm, inner, flip=(sign > 0))
    add_ngon_fan(bm, outer, flip=(sign < 0))


def build_band(bm, y0, z_mid, r_major, tube_r=TUBE_R, minor=6, major_step_deg=30.0, n_seg=9):
    """3/4 torus around the neck, open at the front (-Y), angle 0 at +X."""
    phi0 = -45.0
    rings = []
    for k in range(n_seg + 1):
        phi = math.radians(phi0 + k * major_step_deg)
        cx, cy = r_major * math.cos(phi), y0 + r_major * math.sin(phi)
        rad = Vector((math.cos(phi), math.sin(phi), 0.0))
        ring = []
        for j in range(minor):
            b = 2.0 * math.pi * j / minor
            p = Vector((cx, cy, z_mid)) + rad * (tube_r * math.cos(b)) + Vector((0, 0, tube_r * math.sin(b)))
            ring.append(bm.verts.new(p))
        rings.append(ring)
    for k in range(n_seg):
        a, b = rings[k], rings[k + 1]
        for j in range(minor):
            j2 = (j + 1) % minor
            try:
                bm.faces.new([a[j], a[j2], b[j2], b[j]])
            except ValueError:
                pass
    add_ngon_fan(bm, rings[0], flip=True)
    add_ngon_fan(bm, rings[-1], flip=False)


def ray_in_skin(bvh, co):
    """Parity ray test against any BVH, approximate for an open shell."""
    d, p, n = Vector((0.4472, 0.5774, 0.6832)), co.copy(), 0
    for _ in range(24):
        hit = bvh.ray_cast(p, d, 100.0)
        if hit[0] is None:
            break
        n += 1
        p = hit[0] + d * 1e-5
    return n % 2 == 1


def cup_ring_points(x_off, y0, z_mid, n=12):
    r, half_t = CUP_DIAM / 2.0, CUP_THICK / 2.0
    pts = []
    for sign in (+1, -1):
        cx = sign * x_off
        for face_x in (cx - sign * half_t, cx + sign * half_t):
            for i in range(n):
                a = 2.0 * math.pi * i / n
                pts.append(Vector((face_x, y0 + r * math.cos(a), z_mid + r * math.sin(a))))
    return pts


def place_cups(body, jacket_bvh, y0, x_off0, z_spec, z_lo, z_hi):
    """Search x outward and z near the recipe for cups that clear the body and the jacket."""
    zs = sorted({round(z_lo + 0.001 * i, 4) for i in range(int((z_hi - z_lo) / 0.001) + 1)} | {z_spec},
                key=lambda z: abs(z - z_spec))
    x_off = x_off0
    for _ in range(60):
        for z_mid in zs:
            pts = cup_ring_points(x_off, y0, z_mid)
            if all((not body.in_skin(p)) and (not ray_in_skin(jacket_bvh, p)) for p in pts):
                return x_off, z_mid, True
        x_off += 0.005
    return x_off0, z_spec, False


def phones_from_drawing(C, cls='phones'):
    """Cup centres, cup size and the band's top-back point from the phones stations, in metres."""
    g = C['garments'][cls]
    cal = C['calibration']
    st = [q for q in g.get('stations', []) if q.get('w_mm')]
    z0, z1 = sorted(g['z'])
    # cups: the rows near the maximum outer width
    w_max = max(q['w_mm'] for q in st)
    cup_rows = [q for q in st if q['w_mm'] >= 0.85 * w_max]
    x_outer = 0.5 * max(q['w_mm'] for q in cup_rows) / 1000.0
    gap = g.get('open_front', {}).get('gap_mm', {})
    gaps = []
    for r, (ga, gb) in gap.items():
        z = (cal['floor_row'] - (int(r) + 0.5)) / cal['ppm']
        if min(q['z'] for q in cup_rows) - 0.01 <= z <= max(q['z'] for q in cup_rows) + 0.01:
            gaps.append(0.5 * (abs(ga) + abs(gb)) / 1000.0)
    x_inner = (sum(gaps) / len(gaps)) if gaps else x_outer - CUP_THICK
    z_cup_lo = min(q['z'] for q in cup_rows)
    z_cup_hi = max(q['z'] for q in cup_rows)
    diam = max(0.03, min(0.09, (z_cup_hi - z_cup_lo) + 0.012))
    z_mid = 0.5 * (z_cup_lo + z_cup_hi)
    thick = max(0.012, min(0.04, x_outer - x_inner))
    x_c = 0.5 * (x_outer + x_inner)
    ys = [q['y_mid_mm'] for q in cup_rows if q.get('y_mid_mm') is not None]
    y_cup = (sum(ys) / len(ys)) / 1000.0 if ys else 0.0
    top = max(st, key=lambda q: q['z'])
    y_back = (top['y_mid_mm'] or 0.0) / 1000.0 if top.get('y_mid_mm') is not None else y_cup + 0.05
    z_back = top['z']
    return dict(x_c=x_c, y_cup=y_cup, z_mid=z_mid, diam=diam, thick=thick, y_back=max(y_back, y_cup + 0.03), z_back=max(z_back, z_mid + 0.02))


def build_cup2(bm, centre, sign, diam, thick, n=12):
    r, half_t = diam / 2.0, thick / 2.0
    inner_x = centre.x - sign * half_t
    outer_x = centre.x + sign * half_t
    inner, outer = [], []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        y = centre.y + r * math.cos(a)
        z = centre.z + r * math.sin(a)
        inner.append(bm.verts.new(Vector((inner_x, y, z))))
        outer.append(bm.verts.new(Vector((outer_x, y, z))))
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([inner[i], inner[j], outer[j], outer[i]])
        except ValueError:
            pass
    add_ngon_fan(bm, inner, flip=(sign > 0))
    add_ngon_fan(bm, outer, flip=(sign < 0))


def build_band_arc(bm, pL, pB, pR, tube_r=TUBE_R, minor=6, n_seg=10):
    """Tube along the circular arc through pL, pB, pR: the headband behind the neck."""
    # circle through three points
    a_, b_, c_ = pL, pB, pR
    ab, ac = b_ - a_, c_ - a_
    nrm = ab.cross(ac)
    if nrm.length < 1e-9:
        nrm = Vector((0, 0, 1))
    nrm.normalize()
    t_ = ab.dot(ab) * ac.cross(nrm) + ac.dot(ac) * nrm.cross(ab)
    centre = a_ + t_ / (2.0 * ab.cross(ac).length ** 2) * ab.cross(ac).length ** 2 / max(1e-12, ab.cross(ac).length ** 2) if False else None
    u = ab.normalized()
    v = nrm.cross(u).normalized()
    A2 = Vector((0.0, 0.0)); B2 = Vector((ab.dot(u), ab.dot(v))); C2 = Vector((ac.dot(u), ac.dot(v)))
    d = 2 * (A2.x * (B2.y - C2.y) + B2.x * (C2.y - A2.y) + C2.x * (A2.y - B2.y))
    if abs(d) < 1e-12:
        cx, cy = 0.5 * C2.x, 0.5 * C2.y
    else:
        cx = ((A2.x ** 2 + A2.y ** 2) * (B2.y - C2.y) + (B2.x ** 2 + B2.y ** 2) * (C2.y - A2.y) + (C2.x ** 2 + C2.y ** 2) * (A2.y - B2.y)) / d
        cy = ((A2.x ** 2 + A2.y ** 2) * (C2.x - B2.x) + (B2.x ** 2 + B2.y ** 2) * (A2.x - C2.x) + (C2.x ** 2 + C2.y ** 2) * (B2.x - A2.x)) / d
    centre = a_ + u * cx + v * cy
    R = (a_ - centre).length
    e1 = (a_ - centre).normalized()
    e2 = nrm.cross(e1).normalized()

    def ang(p):
        q = p - centre
        return math.atan2(q.dot(e2), q.dot(e1))
    aL, aB, aR = ang(pL), ang(pB), ang(pR)
    def unwrap(t, ref):
        while t < ref - math.pi:
            t += 2 * math.pi
        while t > ref + math.pi:
            t -= 2 * math.pi
        return t
    aB = unwrap(aB, aL)
    aR = unwrap(aR, aB)
    rings = []
    for k in range(n_seg + 1):
        t = aL + (aR - aL) * k / n_seg
        cpt = centre + (e1 * math.cos(t) + e2 * math.sin(t)) * R
        rad = (e1 * math.cos(t) + e2 * math.sin(t))
        ring = []
        for j in range(minor):
            bb = 2.0 * math.pi * j / minor
            ring.append(bm.verts.new(cpt + rad * (tube_r * math.cos(bb)) + nrm * (tube_r * math.sin(bb))))
        rings.append(ring)
    for k in range(n_seg):
        a1, b1 = rings[k], rings[k + 1]
        for j in range(minor):
            j2 = (j + 1) % minor
            try:
                bm.faces.new([a1[j], a1[j2], b1[j2], b1[j]])
            except ValueError:
                pass
    add_ngon_fan(bm, rings[0], flip=True)
    add_ngon_fan(bm, rings[-1], flip=False)


def build_phones(C, body, jacket_bvh, out_props, cls='phones'):
    P = phones_from_drawing(C, cls)
    x_c, y_cup, z_mid, diam, thick = P['x_c'], P['y_cup'], P['z_mid'], P['diam'], P['thick']
    for _ in range(20):
        pts = [Vector((sx * (x_c - thick / 2), y_cup + diam / 2 * math.cos(a), z_mid + diam / 2 * math.sin(a)))
               for sx in (1, -1) for a in [2 * math.pi * i / 12 for i in range(12)]]
        if not any(body.in_skin(q) for q in pts):
            break
        x_c += 0.003
    def under_jacket(q):
        if jacket_bvh is None:
            return False
        hit = jacket_bvh.find_nearest(q)
        if hit[0] is None:
            return False
        d = q - hit[0]
        return d.length < 0.030 and d.dot(hit[1]) < 0.0
    # a cup still inside a tall collar stays where drawn
    for it in range(8):
        pts = [Vector((sx * (x_c + dx), y_cup + diam / 2 * math.cos(a), z_mid + diam / 2 * math.sin(a)))
               for sx in (1, -1) for dx in (-thick / 2, 0.0, thick / 2) for a in [2 * math.pi * i / 12 for i in range(12)]]
        n_under = sum(1 for q in pts if q.z < z_mid - 0.3 * diam and under_jacket(q))
        if n_under == 0:
            break
        z_mid += 0.003
    bm = bmesh.new()
    build_cup2(bm, Vector((x_c, y_cup, z_mid)), +1, diam, thick)
    build_cup2(bm, Vector((-x_c, y_cup, z_mid)), -1, diam, thick)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob = emit(cls.capitalize(), bm, out_props)
    bmb = bmesh.new()
    pL = Vector((x_c - thick / 2, y_cup + 0.004, z_mid + diam / 2 - TUBE_R))
    pR = Vector((-(x_c - thick / 2), y_cup + 0.004, z_mid + diam / 2 - TUBE_R))
    pB = Vector((0.0, P['y_back'], P['z_back']))
    for _ in range(20):
        if not body.in_skin(pB) and not body.in_skin(Vector((0.0, pB.y - TUBE_R - 0.002, pB.z))):
            break
        pB.y += 0.004
    build_band_arc(bmb, pL, pB, pR)
    bmesh.ops.recalc_face_normals(bmb, faces=bmb.faces[:])
    band = emit(cls.capitalize() + '_band_tmp', bmb, {})
    mod = ob.modifiers.new('Union', 'BOOLEAN')
    mod.operation = 'UNION'
    mod.object = band
    try:
        mod.solver = 'EXACT'
    except Exception:
        pass
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    n_before = len(ob.data.polygons)
    bpy.ops.object.modifier_apply(modifier='Union')
    if len(ob.data.polygons) < n_before:
        print('[accessories] Phones: union failed, joining band and cups')
        ob.select_set(True); band.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.join()
    else:
        bpy.data.objects.remove(band, do_unlink=True)
    bmu = bmesh.new()
    bmu.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bmu, verts=bmu.verts[:], dist=0.0003)
    big = [f for f in bmu.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bmu, faces=big)
    bmesh.ops.recalc_face_normals(bmu, faces=bmu.faces[:])
    bmu.to_mesh(ob.data)
    bmu.free()
    for p_ in ob.data.polygons:
        p_.use_smooth = False
    tris = sum(len(p_.vertices) - 2 for p_ in ob.data.polygons)
    print('[accessories] Phones from the drawing: cups at x +-%.1f mm, y %.1f, z %.3f, diam %.0f mm, thick %.0f mm; band back at (0, %.3f, %.3f)' % (
        x_c * 1000, y_cup * 1000, z_mid, diam * 1000, thick * 1000, pB.y, pB.z))
    return ob, tris, dict(x_off=x_c, z_mid=z_mid, y0=y_cup, diam=diam, neck_half=0.0, x_off0=x_c, z_spec=z_mid, clean=True)


def build_neckband(C, body, cls, props, others=()):
    """A choker or collar: the neck rings in the drawn z range, pushed out to the drawn section."""
    from shell import shell, ring_z
    M = json.load(open(MAP_PATH[0]))
    g = C['garments'][cls]
    z0, z1 = sorted(g['z'])
    trunk = M['trunk']
    rings = [r for r in trunk if z0 - 0.004 <= ring_z(body, r) <= z1 + 0.004]
    if len(rings) < 2:
        zs = sorted(trunk, key=lambda r: abs(ring_z(body, r) - 0.5 * (z0 + z1)))
        rings = sorted(zs[:2], key=lambda r: -ring_z(body, r))
    st = g.get('stations', [])

    def tgt(z, ring):
        q = None
        for s_ in st:
            if abs(s_['z'] - z) < 0.012 and s_.get('w_mm') and s_.get('d_mm'):
                q = s_
                break
        xs = [body.co[i].x for i in ring]; ys = [body.co[i].y for i in ring]
        w_min = (max(xs) - min(xs)) + 0.012; d_min = (max(ys) - min(ys)) + 0.012
        # stand off what is already around the neck within 90 mm of its axis
        cx_, cy_ = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))
        for ob_ in bpy.data.objects:
            if ob_.type != 'MESH' or ob_.get('s3d_part') in ('body', 'hair') or ob_.get('s3d_class') == cls:
                continue
            band = [v.co for v in ob_.data.vertices if abs(v.co.z - z) < 0.008 and math.hypot(v.co.x - cx_, v.co.y - cy_) < 0.09]
            if len(band) >= 6:
                w_min = max(w_min, (max(q.x for q in band) - min(q.x for q in band)) + 0.010)
                d_min = max(d_min, (max(q.y for q in band) - min(q.y for q in band)) + 0.010)
        if q is None:
            return dict(w=w_min, d=d_min, x_c=0.5 * (max(xs) + min(xs)), y_mid=0.5 * (max(ys) + min(ys)))
        return dict(w=max(w_min, q['w_mm'] / 1000.0), d=max(d_min, q['d_mm'] / 1000.0),
                    x_c=(q.get('x_c_mm') or 0.0) / 1000.0, y_mid=(q.get('y_mid_mm') or 0.0) / 1000.0)
    ob = shell(body, [rings], cls.capitalize(), target=tgt, clearance=0.002, thick=0.003, props=dict(props, s3d_kind='band'))
    print('[accessories] %s: neck band over %d rings, z %.3f..%.3f' % (ob.name, len(rings), z0, z1))
    return ob


def flood_components(m):
    """8-connected pixel sets of a bool mask (no scipy in Blender's Python)."""
    ys, xs = np.nonzero(m)
    todo = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while todo:
        seed = todo.pop()
        stack, pts = [seed], []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
        comps.append(pts)
    return comps


def prism(poly3, axis, a0, a1):
    """Sweep a planar polygon along axis 0 (x) or 1 (y) from a0 to a1, ear-clipping the caps."""
    bm = bmesh.new()
    A, B = [], []
    for q in poly3:
        qa, qb = q.copy(), q.copy()
        qa[axis] = a0; qb[axis] = a1
        A.append(bm.verts.new(qa)); B.append(bm.verts.new(qb))
    try:
        bm.faces.new(A)
        bm.faces.new(B[::-1])
    except ValueError:
        pass
    n = len(A)
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([A[i], A[j], B[j], B[i]])
        except ValueError:
            pass
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def loft_hull(name, f_poly, s_poly, props, log=print):
    """Accessory volume from the front outline, each vertex taking its depth from the side outline."""
    zs_s = [q.z for q in s_poly]
    z_lo, z_hi = min(zs_s), max(zs_s)
    n_s = len(s_poly)

    def depth_at(z):
        z = min(max(z, z_lo + 1e-6), z_hi - 1e-6)
        ys = []
        for i in range(n_s):
            a, b = s_poly[i], s_poly[(i + 1) % n_s]
            if (a.z <= z < b.z) or (b.z <= z < a.z):
                t = (z - a.z) / (b.z - a.z)
                ys.append(a.y + (b.y - a.y) * t)
        if len(ys) < 2:
            ys = [min(q.y for q in s_poly), max(q.y for q in s_poly)]
        return min(ys), max(ys)
    bm = bmesh.new()
    F, B = [], []
    for q in f_poly:
        yf, yb = depth_at(q.z)
        if yb - yf < 0.004:
            c = 0.5 * (yf + yb)
            yf, yb = c - 0.002, c + 0.002
        F.append(bm.verts.new(Vector((q.x, yf, q.z))))
        B.append(bm.verts.new(Vector((q.x, yb, q.z))))
    try:
        bm.faces.new(F)
        bm.faces.new(B[::-1])
    except ValueError:
        pass
    n = len(F)
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([F[i], F[j], B[j], B[i]])
        except ValueError:
            pass
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big, ngon_method='BEAUTY')
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) == 4], quad_method='SHORT_EDGE')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    clean_hull(bm)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for k_, v_ in props.items():
        ob[k_] = v_
    for pg in me.polygons:
        pg.use_smooth = False
    return ob, True


def visual_hull(name, bm_a, bm_b, props, log=print):
    """Intersect the front and side prisms, falling back to the front prism."""
    oa = emit(name, bm_a, props)
    ob = emit(name + '_side', bm_b, {})
    ok = False
    for solver in ('EXACT', 'FLOAT'):
        backup = oa.data.copy()
        mod = oa.modifiers.new('i', 'BOOLEAN')
        mod.operation = 'INTERSECT'
        mod.solver = solver
        if solver == 'EXACT':
            mod.use_hole_tolerant = True
        mod.object = ob
        bpy.context.view_layer.objects.active = oa
        bpy.ops.object.modifier_apply(modifier=mod.name)
        if len(oa.data.polygons) >= 4:
            bpy.data.meshes.remove(backup)
            ok = True
            break
        bad = oa.data
        oa.data = backup
        bpy.data.meshes.remove(bad)
        log('[accessories] %s: visual hull with %s failed' % (name, solver))
    bpy.data.objects.remove(ob, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(oa.data)
    clean_hull(bm)
    try:
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(6.0), use_dissolve_boundaries=False, verts=bm.verts[:], edges=bm.edges[:])
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], ngon_method='EAR_CLIP')
    except Exception:
        pass
    bm.to_mesh(oa.data)
    bm.free()
    for pg in oa.data.polygons:
        pg.use_smooth = False
    return oa, ok


def clean_hull(bm, rounds=5):
    """Clean a Boolean's output: merge doubles and coplanar faces, collapse needles, keep it manifold."""
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0025)
    bmesh.ops.dissolve_degenerate(bm, dist=0.0005, edges=bm.edges[:])
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(8.0), verts=bm.verts[:], edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3], quad_method='SHORT_EDGE', ngon_method='BEAUTY')

    def needles_of(b):
        out = []
        for f in b.faces:
            if len(f.verts) != 3 or f.calc_area() * 1e6 < 6.0:
                continue
            vs = f.verts
            worst, k_sharp = 180.0, 0
            for i in range(3):
                u, w = vs[i - 1].co - vs[i].co, vs[(i + 1) % 3].co - vs[i].co
                if u.length < 1e-9 or w.length < 1e-9:
                    continue
                ang = math.degrees(u.angle(w))
                if ang < worst:
                    worst, k_sharp = ang, i
            if worst < 4.0:
                out.append((f, vs[k_sharp]))
        return out

    def rebuild(b, src):
        me_tmp = bpy.data.meshes.new('_hull_tmp')
        src.to_mesh(me_tmp)
        b.clear()
        b.from_mesh(me_tmp)
        bpy.data.meshes.remove(me_tmp)
    for _ in range(rounds):
        trial = bm.copy()
        needles = needles_of(trial)
        if not needles:
            trial.free()
            break
        done = set()
        for f, sharp in needles:
            if not f.is_valid or any(v in done for v in f.verts):
                continue
            e0 = min(f.edges, key=lambda e: e.calc_length())
            if e0.calc_length() < 0.008:
                done.update(e0.verts)
                try:
                    bmesh.ops.collapse(trial, edges=[e0], uvs=False)
                except Exception:
                    pass
            else:
                done.add(sharp)
                try:
                    bmesh.ops.dissolve_verts(trial, verts=[sharp])
                except Exception:
                    pass
        bmesh.ops.dissolve_degenerate(trial, dist=0.0005, edges=trial.edges[:])
        bmesh.ops.triangulate(trial, faces=[f for f in trial.faces if len(f.verts) > 3], quad_method='SHORT_EDGE', ngon_method='BEAUTY')
        if any(not e.is_manifold for e in trial.edges) or len(needles_of(trial)) >= len(needles):
            trial.free()
            break
        rebuild(bm, trial)
        trial.free()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])




def trace_outline(pts2, tol, max_pts=4000):
    """Moore-neighbour trace of a pixel set's outer boundary, simplified to `tol` px."""
    if len(pts2) < 8:
        return []
    xs = pts2[:, 0].astype(int); ys = pts2[:, 1].astype(int)
    x0, y0 = int(xs.min()) - 1, int(ys.min()) - 1
    m = np.zeros((int(ys.max()) - y0 + 2, int(xs.max()) - x0 + 2), bool)
    m[ys - y0, xs - x0] = True
    H, W = m.shape
    rs, cs = np.nonzero(m)
    k0 = np.lexsort((cs, rs))[0]
    start = (int(rs[k0]), int(cs[k0]))
    # 8 neighbours clockwise from west
    nb = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

    def at(q):
        return 0 <= q[0] < H and 0 <= q[1] < W and m[q[0], q[1]]
    contour = [start]
    b = start
    back = (start[0], start[1] - 1)
    first_move = None
    for _ in range(max_pts):
        di = [i for i, d in enumerate(nb) if (b[0] + d[0], b[1] + d[1]) == back]
        i0 = di[0] if di else 0
        found = None
        for j in range(1, 9):
            i = (i0 + j) % 8
            q = (b[0] + nb[i][0], b[1] + nb[i][1])
            if at(q):
                found = (q, (b[0] + nb[(i - 1) % 8][0], b[1] + nb[(i - 1) % 8][1]))
                break
        if found is None:
            break
        q, back = found
        if q == start and first_move is not None and (b, q) == first_move:
            break
        if first_move is None:
            first_move = (b, q)
        b = q
        if b == start and len(contour) > 2:
            break
        contour.append(b)
    if len(contour) < 3:
        return []
    ring = [(float(c + x0), float(r + y0)) for r, c in contour]

    def dp(seg):
        if len(seg) < 3:
            return seg
        (ax, ay), (bx, by) = seg[0], seg[-1]
        L = math.hypot(bx - ax, by - ay)
        best, bi = -1.0, -1
        for i in range(1, len(seg) - 1):
            x, y = seg[i]
            dd = abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / L if L > 1e-9 else math.hypot(x - ax, y - ay)
            if dd > best:
                best, bi = dd, i
        if best <= tol:
            return [seg[0], seg[-1]]
        return dp(seg[:bi + 1])[:-1] + dp(seg[bi:])
    c = pts2.mean(0)
    k = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - c[0], ring[i][1] - c[1]))
    ring = ring[k:] + ring[:k]
    out = dp(ring + [ring[0]])[:-1]
    clean = []
    for q in out:
        if not clean or math.hypot(q[0] - clean[-1][0], q[1] - clean[-1][1]) > 0.5:
            clean.append(q)
    return clean if len(clean) >= 3 else []


def star_outline(pts2, n_rays, tol):
    """Star polygon about the centroid: the farthest pixel per ray, simplified to `tol` px."""
    if len(pts2) < 3:
        return []
    c = pts2.mean(0)
    d = pts2 - c
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.hypot(d[:, 0], d[:, 1])
    ring = []
    step = 2 * math.pi / n_rays
    for i in range(n_rays):
        a0 = -math.pi + i * step
        sel = (ang >= a0) & (ang < a0 + step)
        if not sel.any():
            continue
        j = np.argmax(np.where(sel, rad, -1))
        ring.append((float(pts2[j, 0]), float(pts2[j, 1])))
    if len(ring) < 3:
        return ring

    def dp(seg):
        if len(seg) < 3:
            return seg
        (x0, y0), (x1, y1) = seg[0], seg[-1]
        L = math.hypot(x1 - x0, y1 - y0)
        best, bi = -1.0, -1
        for i in range(1, len(seg) - 1):
            x, y = seg[i]
            dd = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / L if L > 1e-9 else math.hypot(x - x0, y - y0)
            if dd > best:
                best, bi = dd, i
        if best <= tol:
            return [seg[0], seg[-1]]
        return dp(seg[:bi + 1])[:-1] + dp(seg[bi:])
    k0 = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - c[0], ring[i][1] - c[1]))
    ring = ring[k0:] + ring[:k0]
    out = dp(ring + [ring[0]])[:-1]
    return out if len(out) >= 3 else ring


def simplify_ring(pts, tol):
    if len(pts) < 4:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        vx, vy = c[0] - a[0], c[1] - a[1]
        L = math.hypot(vx, vy)
        d = abs(vx * (b[1] - a[1]) - vy * (b[0] - a[0])) / L if L > 1e-9 else 0.0
        if d < tol:
            continue
        if out and math.hypot(b[0] - out[-1][0], b[1] - out[-1][1]) < tol:
            continue
        out.append(b)
    return out if len(out) >= 3 else pts


def convex_hull(P):
    """Convex hull by Andrew's monotone chain, counter-clockwise."""
    pts = sorted(set((float(x), float(y)) for x, y in P))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def gather_stud_islands(C, hosts=('jacket', 'coat', 'vest')):
    ppm = C['calibration']['ppm']
    floor_row = C['calibration']['floor_row']
    panels = dict((p['name'], p) for p in C['calibration']['panels'])
    armpit_row = C['landmarks']['armpit']['row']
    out = []
    for it in C.get('islands', []):
        if it.get('host') not in hosts or it.get('cls') in hosts or it.get('cls') in ('hair', 'skin'):
            continue
        if it.get('panel') not in ('front', 'back'):
            continue
        r, c = it['centre']
        if r >= armpit_row:
            continue
        z = (floor_row - r) / ppm
        if it['panel'] == 'front':
            x = (c - panels['front']['midline']) / ppm
        else:
            x = (panels['back']['midline'] - c) / ppm
        out.append(dict(panel=it['panel'], x=x, z=z, row=r))
    return out


def build_stud(bm, hit, normal, clear=STUD_CLEAR):
    n = normal.normalized()
    tmp = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    u = tmp.cross(n).normalized()
    v = n.cross(u).normalized()
    base_c = hit + n * clear
    h = STUD_BASE / 2.0
    corners = [base_c + u * h + v * h, base_c - u * h + v * h,
               base_c - u * h - v * h, base_c + u * h - v * h]
    bverts = [bm.verts.new(p) for p in corners]
    apex = bm.verts.new(base_c + n * STUD_H)
    try:
        bm.faces.new([bverts[3], bverts[2], bverts[1], bverts[0]])  # base, wound to face -n
    except ValueError:
        pass
    for i in range(4):
        a, b = bverts[i], bverts[(i + 1) % 4]
        try:
            bm.faces.new([a, b, apex])
        except ValueError:
            pass
    return bverts + [apex]


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, out_blend = argv[:4]
    CHAR_PATH[0], MAP_PATH[0] = char_path, map_path
    C = json.load(open(char_path))
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    body = BodyRef(bpy.data.objects['Body'])
    G = C['garments']
    # the host is the outermost torso garment
    torso_names = [g for g, e in G.items() if e['kind'] == 'torso' and g.capitalize() in bpy.data.objects]
    host_name = None
    if torso_names:
        def mean_w(g):
            ws = [q['w_mm'] for q in G[g].get('stations', []) if q.get('w_mm')]
            return sum(ws) / len(ws) if ws else 0.0
        host_name = max(torso_names, key=mean_w)
    jacket_ob = bpy.data.objects[host_name.capitalize()] if host_name else None
    jacket_bvh = None
    if jacket_ob is not None:
        jbm = bmesh.new()
        jbm.from_mesh(jacket_ob.data)
        jbm.transform(jacket_ob.matrix_world)
        jacket_bvh = BVHTree.FromBMesh(jbm)
        jbm.free()

    made = []

    def kind_of(name):
        n = name.lower()
        if re.search(r'phone|headphone|headset', n):
            return 'headphones'
        if re.search(r'choker|neckband|necklace|collar', n):
            return 'neckband'
        if re.search(r'bow|ribbon|badge|patch|pin|emblem|button', n):
            return 'emblem'
        return 'emblem'
    for aname, g in G.items():
        if g['kind'] != 'accessory':
            continue
        akind = kind_of(aname)
        props = dict(s3d_part='accessory', s3d_class=aname)
        if akind == 'headphones':
            phones, phones_tris, info = build_phones(C, body, jacket_bvh, props, cls=aname)
            made.append((phones, phones_tris))
            n_in_body = sum(1 for v in phones.data.vertices if body.in_skin(v.co.copy()))
            if n_in_body:
                print('[accessories] WARNING %s: %d/%d verts inside Body' % (phones.name, n_in_body, len(phones.data.vertices)))
        elif akind == 'neckband':
            ob = build_neckband(C, body, aname, props, others=[m for m, _ in made if m.get('s3d_kind') == 'band'])
            if ob is not None:
                made.append((ob, sum(len(f.vertices) - 2 for f in ob.data.polygons)))
        else:
            import attach
            from carve import Sheet
            sheet_ = Sheet(char_path)
            for ob in attach.build_class(C, sheet_, aname, props, log=print):
                made.append((ob, sum(len(f.vertices) - 2 for f in ob.data.polygons)))

    if jacket_bvh is None:
        total_tris = sum(t for _, t in made)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        print('[accessories] %d objects, %d tris (no torso host: no studs) -> %s' % (len(made), total_tris, out_blend))
        return
    islands = gather_stud_islands(C, hosts=(host_name,))
    print('[accessories] stud islands (light islands in %s, above the armpit row): %d' % (host_name, len(islands)))
    sbm = bmesh.new()
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.get('s3d_part') in ('body', 'hair') or o.get('s3d_class') == 'stud':
            continue
        t_ = bmesh.new(); t_.from_mesh(o.data); t_.transform(o.matrix_world)
        for f in t_.faces:
            try:
                sbm.faces.new([sbm.verts.new(v.co) for v in f.verts])
            except ValueError:
                pass
        t_.free()
    if len(sbm.faces):
        jacket_bvh = BVHTree.FromBMesh(sbm)
    sbm.free()
    n_stud_in_body = 0
    for i, isl in enumerate(islands):
        side = 'L' if isl['x'] > 0 else 'R'
        # cast a ray from the drawn spot, stepping down a few mm on a miss
        front = isl['panel'] == 'front'
        direction = Vector((0.0, 1.0 if front else -1.0, 0.0))
        cast, method = None, None

        def first_outer_hit(origin, direction):
            """First hit of the host along the ray, only on the viewer's side of the body."""
            c = jacket_bvh.ray_cast(origin, direction, 2.0)
            if c[0] is None:
                return None
            near = body.bvh.find_nearest(c[0])
            if near[0] is not None and (c[0] - near[0]).dot(direction) > 0.004:
                return None
            return c
        for dz in (0.0, -0.003, -0.006, -0.010, -0.015, -0.020):
            origin = Vector((isl['x'], -1.0 if front else 1.0, isl['z'] + dz))
            c = first_outer_hit(origin, direction)
            if c is not None:
                cast = c
                method = 'ray' if dz == 0.0 else 'ray (z%+.0fmm)' % (dz * 1000)
                break
        if cast is None:
            print('[accessories] stud %d (%s panel, row %d): no outer surface of the host under its drawn spot - skipped'
                  % (i, isl['panel'], isl['row']))
            continue
        hit, normal = cast[0], cast[1]
        # the open jacket's normals are unreliable: orient by the panel's view direction
        guess_out = Vector((0.0, -1.0, 0.0)) if front else Vector((0.0, 1.0, 0.0))
        if normal is None or normal.length < 1e-6:
            normal = guess_out.copy()
        if normal.dot(guess_out) < 0.0:
            normal = -normal
        # a nearest-point fallback may land on a fold: more clearance
        clear = STUD_CLEAR if method.startswith('ray') else 0.006
        bm = bmesh.new()
        build_stud(bm, hit, normal, clear)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        tris = sum(len(f.verts) - 2 for f in bm.faces)
        name = 'Stud_%d' % i
        ob = emit(name, bm, dict(s3d_part='accessory', s3d_class='stud', s3d_side=side))
        made.append((ob, tris))
        for v in ob.data.vertices:
            if body.in_skin(v.co.copy()):
                n_stud_in_body += 1
        print('[accessories] %-8s %s panel row %-4d x=%7.1fmm z=%.4f -> hit=(%.4f,%.4f,%.4f) via %s  %d tris'
              % (name, isl['panel'], isl['row'], isl['x'] * 1000, isl['z'], hit.x, hit.y, hit.z, method, tris))
    if n_stud_in_body:
        print('[accessories] WARNING %d stud vertices land inside Body' % n_stud_in_body)

    total_tris = sum(t for _, t in made)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[accessories] %d objects, %d tris -> %s' % (len(made), total_tris, out_blend))


if __name__ == '__main__':
    main()
