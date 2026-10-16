"""Build the hair: a cap over the skull, plates (crests) and spikes.

    blender -b --factory-startup --python sketch3d/build/hair.py -- <char.json> <body.json> <in.blend> <out.blend>
"""
import bpy, bmesh, os, sys, json, math
from mathutils import Vector
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from shell import BodyRef, emit   # noqa: E402
from limb import rmf   # noqa: E402
from carve import Sheet, pin_edge_to_skin, fix_selfx   # noqa: E402

THICK = 0.0030
CLEAR = 0.0020
OFF_MIN = 0.0070
OFF_MAX = 0.0240
MASS_STEP = 0.010
MASS_NC = 20
MASS_P = 1.00
SPIKE_S = (0.10, 0.27, 0.46, 0.64, 0.80, 0.92)  # 0 = base, 1 = tip
SPIKE_ROOT = 0.035
ROOT_FLARE = 1.45
Z_FRONT_W = 0.7
PAIR_MAX = 0.100


def table_at(rows, z, tol=0.006):
    rs = sorted([r for r in rows if r.get('u0') is not None and r.get('u1') is not None], key=lambda r: r['z'])
    if len(rs) < 2:
        return None
    zs = [r['z'] for r in rs]
    if z < zs[0] - tol or z > zs[-1] + tol:
        return None
    z = min(max(z, zs[0]), zs[-1])
    for i in range(len(zs) - 1):
        if zs[i] <= z <= zs[i + 1]:
            t = (z - zs[i]) / max(1e-9, zs[i + 1] - zs[i])
            a0, a1 = sorted((rs[i]['u0'], rs[i]['u1']))
            b0, b1 = sorted((rs[i + 1]['u0'], rs[i + 1]['u1']))
            return (a0 + (b0 - a0) * t, a1 + (b1 - a1) * t)
    return tuple(sorted((rs[-1]['u0'], rs[-1]['u1'])))


def smooth_table(rows, half=0.015):
    rs = [r for r in rows if r.get('u0') is not None and r.get('u1') is not None]
    out = []
    for r in rs:
        nb = [q for q in rs if abs(q['z'] - r['z']) <= half]
        out.append(dict(z=r['z'], u0=sum(min(q['u0'], q['u1']) for q in nb) / len(nb), u1=sum(max(q['u0'], q['u1']) for q in nb) / len(nb)))
    return out


def skull_extent(body, rings, z):
    best = None
    for r in rings:
        if len(r) < 3:
            continue
        zc = sum(body.co[i].z for i in r) / len(r)
        if best is None or abs(zc - z) < abs(best[0] - z):
            best = (zc, r)
    if best is None:
        return None
    r = best[1]
    xs = [body.co[i].x for i in r]
    ys = [body.co[i].y for i in r]
    return (min(xs), max(xs), min(ys), max(ys))


def find_pole(body, ring0):
    """The dome's apex vertex: it caps ring 0 but belongs to no ring."""
    z0 = max(body.co[i].z for i in ring0)
    s = set(ring0)
    cand = {}
    for vs in body.polys:
        n = sum(1 for i in vs if i in s)
        if n < 2:
            continue
        for i in vs:
            if i not in s and body.co[i].z > z0 + 1e-5:
                cand[i] = body.co[i].z
    if not cand:
        return None
    return max(cand, key=cand.get)


def signed_volume(bm):
    vol = 0.0
    for f in bm.faces:
        vs = [v.co for v in f.verts]
        for i in range(1, len(vs) - 1):
            vol += vs[0].dot(vs[i].cross(vs[i + 1]))
    return vol


def close_and_emit(name, bm, props, log):
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if signed_volume(bm) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    nf = len(bm.faces)
    ob = emit(name, bm, props)
    return ob, nf


def hair_body(H, sk, body, M, z_hair, props, log=print, sheet=None, C_roles=None):
    """Build the hair body as rings 10 mm apart from the nape to the crown, cut in front of the hairline."""
    cap_f = smooth_table(H.get('cap', {}).get('front', []))
    cap_s = smooth_table(H.get('cap', {}).get('side', []))
    cap_b = smooth_table(H.get('cap', {}).get('back', []))
    if not cap_s or not (cap_f or cap_b):
        return None
    zf = [r['z'] for r in (cap_f or cap_b)]
    zs = [r['z'] for r in cap_s]
    z_top = max(max(zf), max(zs))
    z_lo = min(zs)
    back_rows = H.get('mask_rows', {}).get('back', [])
    nape_rows = [r['z'] for r in back_rows if any(a <= 0.03 and b >= -0.03 for a, b in r['runs'])]
    if nape_rows:
        z_lo = min(z_lo, min(nape_rows))
    z_lo = max(z_lo, sk['chin_z'] - 0.03)
    if z_top - z_lo < 0.03:
        return None
    head_rings = M['head']
    z_skull_top = max(body.co[i].z for i in head_rings[0])
    z_centre = 0.5 * (sk['chin_z'] + z_skull_top)
    # a mass more than two stand-offs above the skull is a crest of spikes
    if z_top > z_skull_top + 2.0 * OFF_MAX:
        log('[hair] cap top %.3f capped at the skull top + %.0f mm = %.3f (the drawn mass above is the crest)' % (z_top, 2000 * OFF_MAX, z_skull_top + 2.0 * OFF_MAX))
        z_top = z_skull_top + 2.0 * OFF_MAX
    n = max(3, int(round((z_top - z_lo) / MASS_STEP)))
    zlist = [z_lo + (z_top - z_lo) * i / n for i in range(n + 1)]
    zlist += [z_top + 0.005, z_top + 0.009]
    scales = [1.0] * (n + 1) + [0.80, 0.50]
    z_s_top, z_f_top = max(zs), max(zf)
    bm = bmesh.new()
    rings = []
    C_roles = C_roles or {}
    last_fx = last_sy = None
    for z, sc in zip(zlist, scales):
        zq = min(z, z_top)
        fx = table_at(cap_f, zq) or table_at(cap_b, zq)
        sy = table_at(cap_s, zq)
        se = skull_extent(body, head_rings, zq) if zq <= z_skull_top + 0.01 else None
        if fx is None:
            if se is not None:
                fx = (se[0] - OFF_MIN, se[1] + OFF_MIN)
            elif last_fx is not None:
                fx = last_fx
            else:
                continue
        if sy is None:
            if last_sy is not None and last_fx is not None:
                k = (fx[1] - fx[0]) / max(1e-6, last_fx[1] - last_fx[0])
                cy = 0.5 * (last_sy[0] + last_sy[1]); hd = 0.5 * (last_sy[1] - last_sy[0]) * min(1.0, k)
                sy = (cy - hd, cy + hd)
            elif se is not None:
                sy = (se[2] - OFF_MIN, se[3] + OFF_MIN)
            else:
                continue
        if se is not None:
            fx = (min(fx[0], se[0] - OFF_MIN), max(fx[1], se[1] + OFF_MIN))
            sy = (sy[0], max(sy[1], se[3] + OFF_MIN))
            # max stand-off: OFF_MAX below the skull centre, 2.5 x above, blended over 100 mm
            t_ = min(1.0, max(0.0, (zq - (z_centre - 0.05)) / 0.10))
            lim = 1.5 * OFF_MAX + (2.5 * OFF_MAX - 1.5 * OFF_MAX) * (t_ * t_ * (3 - 2 * t_))
            fx = (max(fx[0], se[0] - lim), min(fx[1], se[1] + lim))
            sy = (max(sy[0], se[2] - lim), min(sy[1], se[3] + lim))
        elif zq > z_top - 1e-6:
            # the dome rings close over the last drawn section
            pass
        if zq <= z_f_top:
            last_fx = fx
        if zq <= z_s_top:
            last_sy = sy
        cx, cy = 0.5 * (fx[0] + fx[1]), 0.5 * (sy[0] + sy[1])
        hw, hd = 0.5 * (fx[1] - fx[0]) * sc, 0.5 * (sy[1] - sy[0]) * sc
        # the ring is the drawn hull or the skull + OFF_MIN, whichever is further out
        c3 = Vector((cx, cy, z))
        ring = []
        for k in range(MASS_NC):
            th = 2 * math.pi * k / MASS_NC
            sx = math.copysign(abs(math.sin(th)) ** MASS_P, math.sin(th))
            cs = math.copysign(abs(math.cos(th)) ** MASS_P, math.cos(th))
            q = Vector((cx + hw * sx, cy - hd * cs, z))
            dv = q - c3
            r_d = dv.length
            if r_d < 1e-6:
                ring.append(q)
                continue
            dv /= r_d
            r = r_d
            if se is not None:
                hit = body.bvh.ray_cast(c3, dv, 0.5)
                if hit[0] is not None:
                    r = max(r, (hit[0] - c3).length + OFF_MIN)
            q3 = c3 + dv * r
            co = body.lift_out(q3.copy(), OFF_MIN)
            ring.append(co if co is not None else q3)
        rings.append(ring)
    if len(rings) < 3:
        bm.free()
        return None
    bm.free()
    # smooth the stand-off as a lower envelope: a bulge (a bow, a tail root) is not scalp
    n_r, n_c = len(rings), len(rings[0])
    n_dome = 2
    for k in range(n_c):
        hits, offs = [], []
        for i in range(n_r):
            h = body.bvh.find_nearest(rings[i][k])
            if h[0] is None or i >= n_r - n_dome:
                hits.append(None); offs.append(None)
                continue
            d = rings[i][k] - h[0]
            if abs(d.z) > 0.7 * d.length:
                hits.append(None); offs.append(None)
                continue
            hits.append(h[0])
            offs.append(d.length)
        valid = [i for i in range(n_r) if offs[i] is not None]
        if len(valid) < 3:
            continue
        # median of three, then going down the stand-off grows by at most 0.6 ring step per ring
        sm = {}
        for i in valid:
            win = sorted(offs[j] for j in valid if abs(j - i) <= 1)
            sm[i] = win[len(win) // 2]
        order = sorted(valid, key=lambda i: -rings[i][k].z)
        for a_, b_ in zip(order, order[1:]):
            dz = abs(rings[a_][k].z - rings[b_][k].z)
            if sm[b_] > sm[a_] + 0.6 * dz:
                sm[b_] = sm[a_] + 0.6 * dz
        for i in valid:
            off = min(max(sm[i], OFF_MIN), 2.5 * OFF_MAX)
            d = rings[i][k] - hits[i]
            if d.length > 1e-6 and abs(off - offs[i]) > 0.0005:
                rings[i][k] = hits[i] + d.normalized() * off
    if sheet is None:
        return None
    return cap_from_grid(rings, sheet, C_roles, body, props, log)


def cap_from_grid(rings, sheet, roles, body, props, log=print):
    """Build the cap with a hairline: each column ends where the drawing stops showing hair."""
    NC = len(rings[0])
    T = rings[::-1]
    n_r = len(T)
    classes = list(roles.get('hair', ['hair'])) + list(roles.get('accessory', []))

    def inside(q):
        c = sum(T[min(n_r - 1, 3)], Vector()) / NC
        dv = q - Vector((c.x, c.y, q.z))
        if abs(dv.y) >= abs(dv.x):
            view = 'front' if dv.y < 0 else 'back'
        else:
            view = 'side'
        if view not in sheet.panels:
            view = 'front'
        m = sheet.mask(view, classes, tol_px=3, drawn_only=True, close_px=int(round(0.004 * sheet.ppm)))
        r, cc = sheet.project(view, q)
        return 0 <= r < m.shape[0] and 0 <= cc < m.shape[1] and bool(m[r, cc])
    z_edge = []
    for k in range(NC):
        last, gap = 0, 0
        for i in range(n_r):
            if inside(T[i][k]):
                last, gap = i, 0
            else:
                gap += 1
                if gap > 2:
                    break
        if last == 0 and not inside(T[0][k]):
            last = min(2, n_r - 1)
        if last < n_r - 1:
            a, b = T[last][k].copy(), T[last + 1][k].copy()
            for _ in range(6):
                mid = (a + b) * 0.5
                if inside(mid):
                    a = mid
                else:
                    b = mid
            z_edge.append(a.z)
        else:
            z_edge.append(T[-1][k].z)
    # smooth the hairline, a short run 40 mm above both flanks is an ear or a hand
    z_edge = list(z_edge)
    for k in range(NC):
        for run in (1, 2):
            ks = [(k + i) % NC for i in range(run)]
            a_, b_ = z_edge[(k - 1) % NC], z_edge[(k + run) % NC]
            if all(z_edge[q] > max(a_, b_) + 0.04 for q in ks):
                for i, q in enumerate(ks):
                    z_edge[q] = a_ + (b_ - a_) * (i + 1) / (run + 1)
                break
    zm = [sorted([z_edge[k - 1], z_edge[k], z_edge[(k + 1) % NC]])[1] for k in range(NC)]
    zm = [sum(zm[(k + d) % NC] for d in (-2, -1, 0, 1, 2)) / 5.0 for k in range(NC)]
    for _ in range(2):
        for k in range(NC):
            lo = max(zm[k - 1], zm[(k + 1) % NC]) - 0.05
            if zm[k] < lo:
                zm[k] = lo
    m_k, e_k = [], []
    for k in range(NC):
        last = 0
        for i in range(n_r):
            if T[i][k].z >= zm[k] - 1e-6:
                last = i
            else:
                break
        last = max(last, min(2, n_r - 1))
        m_k.append(last)
        if last < n_r - 1 and T[last][k].z - zm[k] > 0.006:
            a, b = T[last][k], T[last + 1][k]
            t = (a.z - zm[k]) / max(1e-6, a.z - b.z)
            e_k.append(a + (b - a) * min(1.0, max(0.0, t)))
        else:
            e_k.append(None)
    bm = bmesh.new()
    V = [[None] * NC for _ in range(n_r)]
    E = [None] * NC
    TAPER = 0.030

    ring_c = [sum(r, Vector()) / NC for r in T]

    def skin_radial(q, i):
        """Skin under q along the ray from its ring centre, not the nearest point."""
        c = ring_c[min(i, n_r - 1)]
        axis = Vector((c.x, c.y, q.z))
        d = q - axis
        if d.length < 1e-6:
            return None
        hit = body.bvh.ray_cast(axis, d.normalized(), 0.3)
        return hit[0] if hit[0] is not None else None

    def tapered(q, k, i):
        dz = q.z - zm[k]
        if dz >= TAPER:
            return q
        hit = skin_radial(q, i)
        if hit is None or (hit - q).length > 0.025:
            return q
        f = max(0.25, dz / TAPER)
        return hit + (q - hit) * f
    for k in range(NC):
        for i in range(m_k[k] + 1):
            V[i][k] = bm.verts.new(tapered(T[i][k], k, i))
        # an edge vertex within 6 mm of the ring vertex is that vertex, else a needle
        if e_k[k] is not None and (e_k[k] - T[m_k[k]][k]).length > 0.006:
            q = e_k[k]
            hit = skin_radial(q, m_k[k])
            if hit is not None and (hit - q).length < 0.025:
                q = hit + (q - hit).normalized() * 0.0025
            E[k] = bm.verts.new(q)
        else:
            E[k] = V[m_k[k]][k]

    def face(vs):
        vs2 = [v for j, v in enumerate(vs) if v is not vs[j - 1]]
        if len(vs2) >= 3:
            try:
                bm.faces.new(vs2)
            except ValueError:
                pass
    for k in range(NC):
        j = (k + 1) % NC
        m = min(m_k[k], m_k[j])
        big = abs(m_k[k] - m_k[j]) >= 2
        for i in range(m):
            if big and i == m - 1:
                continue
            face([V[i][k], V[i + 1][k], V[i + 1][j], V[i][j]])
        if big:
            lo_, hi_ = (j, k) if m_k[k] > m_k[j] else (k, j)
            mid_v = {}
            for i in range(m, max(m_k[k], m_k[j]) + 1):
                mid_v[i] = bm.verts.new(0.5 * (V[i][k].co if V[i][k] is not None else tapered(T[i][k], k, i)) + 0.5 * (V[i][j].co if V[i][j] is not None else tapered(T[i][j], j, i)))
            if m >= 1:
                face([V[m - 1][k], V[m][k], mid_v[m], V[m][j], V[m - 1][j]])
            if E[lo_] is not V[m][lo_]:
                if lo_ == j:
                    face([mid_v[m], E[j], V[m][j]])
                else:
                    face([V[m][k], E[k], mid_v[m]])
            for i in range(m, max(m_k[k], m_k[j])):
                if hi_ == k:
                    face([V[i][k], V[i + 1][k], mid_v[i + 1], mid_v[i]])
                else:
                    face([mid_v[i], mid_v[i + 1], V[i + 1][j], V[i][j]])
            mh = max(m_k[k], m_k[j])
            if E[hi_] is not V[mh][hi_]:
                if hi_ == k:
                    face([V[mh][k], E[k], mid_v[mh]])
                else:
                    face([mid_v[mh], E[j], V[mh][j]])
        elif m_k[k] > m:
            if E[j] is not V[m][j]:
                face([V[m][k], E[j], V[m][j]])
            for i in range(m, m_k[k]):
                face([V[i][k], V[i + 1][k], E[j]])
            if E[k] is not V[m_k[k]][k]:
                face([V[m_k[k]][k], E[k], E[j]])
        elif m_k[j] > m:
            if E[k] is not V[m][k]:
                face([V[m][k], E[k], V[m][j]])
            for i in range(m, m_k[j]):
                face([E[k], V[i + 1][j], V[i][j]])
            if E[j] is not V[m_k[j]][j]:
                face([E[k], E[j], V[m_k[j]][j]])
        else:
            face([V[m][k], E[k], E[j], V[m][j]])
    top = V[0]
    c_top = sum((v.co for v in top), Vector()) / NC
    pole = bm.verts.new(Vector((c_top.x, c_top.y, c_top.z + 0.006)))
    for k in range(NC):
        face([top[k], pole, top[(k + 1) % NC]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    c_all = sum((v.co for v in bm.verts), Vector()) / max(1, len(bm.verts))
    out_n = sum(1 for f in bm.faces if f.normal.dot(f.calc_center_median() - c_all) > 0)
    if out_n < len(bm.faces) // 2:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    me = bpy.data.meshes.new('Hair')
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('Hair', me)
    bpy.context.collection.objects.link(ob)
    for pg in me.polygons:
        pg.use_smooth = False
    for k_, v_ in dict(props, s3d_kind='mass').items():
        ob[k_] = v_
    n_edge = sum(1 for e in E if e is not None)
    log('[hair] cap: %d rings x %d columns, hairline designed per column (%d column ends on the drawn edge), %d faces' % (n_r, NC, n_edge, len(me.polygons)))
    return ob


def union_into(target, parts, log=print):
    """Boolean-union `parts` into `target`, clean up and return the piece count."""
    kept = []
    # separate pieces go in after the Booleans: the solver mangles pieces it does not touch
    separate = [ob for ob in parts if ob.get('s3d_separate')]
    for ob in parts:
        ok = False
        if ob.get('s3d_separate'):
            continue
        for solver in ('EXACT', 'FLOAT'):
            backup = target.data.copy()
            n0 = len(target.data.polygons)
            m = target.modifiers.new('u', 'BOOLEAN')
            m.operation = 'UNION'
            m.solver = solver
            if solver == 'EXACT':
                m.use_hole_tolerant = True
            m.object = ob
            bpy.context.view_layer.objects.active = target
            bpy.ops.object.modifier_apply(modifier=m.name)
            n1 = len(target.data.polygons)
            if n1 < 0.9 * n0:
                bad = target.data
                target.data = backup
                bpy.data.meshes.remove(bad)
                log('[hair] union with %s (%s) failed: %d -> %d faces' % (ob.name, solver, n0, n1))
                continue
            bpy.data.meshes.remove(backup)
            ok = True
            break
        if ok:
            bpy.data.objects.remove(ob, do_unlink=True)
        else:
            kept.append(ob)
            log('[hair] union with %s FAILED with every solver: kept as its own object' % ob.name)
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0004)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(2.0), verts=bm.verts[:], edges=bm.edges[:])
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for ob in separate:
        sb = bmesh.new()
        sb.from_mesh(ob.data)
        sb.transform(ob.matrix_world)
        vmap = {v: bm.verts.new(v.co) for v in sb.verts}
        for f in sb.faces:
            try:
                bm.faces.new([vmap[v] for v in f.verts])
            except ValueError:
                pass
        sb.free()
        log('[hair] %s joined as its own piece (tied off the head)' % ob.name)
        bpy.data.objects.remove(ob, do_unlink=True)
    seen, pieces = set(), 0
    for v in bm.verts:
        if v in seen:
            continue
        pieces += 1
        stack = [v]
        seen.add(v)
        while stack:
            q = stack.pop()
            for e in q.link_edges:
                w = e.other_vert(q)
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    bm.to_mesh(target.data)
    bm.free()
    target.data.update()
    log('[hair] union: %d faces, %d connected piece(s), %d spike(s) left separate' % (len(target.data.polygons), pieces, len(kept)))
    return pieces, kept


def slice_poly(poly, base, tip, s):
    """(centre offset, half-width) of the polygon across the base-tip axis at s, or None."""
    P = [Vector((u, z)) for u, z in poly]
    if len(P) < 3:
        return None
    b = Vector((base['u'], base['z'])); t = Vector((tip['u'], tip['z']))
    ax = t - b
    L = ax.length
    if L < 1e-6:
        return None
    d = ax / L
    perp = Vector((-d.y, d.x))
    p0 = b + d * (L * s)
    ts = []
    n = len(P)
    for i in range(n):
        a_, c_ = P[i], P[(i + 1) % n]
        fa, fc = (a_ - p0).dot(d), (c_ - p0).dot(d)
        if fa == fc:
            continue
        if (fa <= 0.0 <= fc) or (fc <= 0.0 <= fa):
            tt = fa / (fa - fc)
            q = a_ + (c_ - a_) * tt
            ts.append((q - p0).dot(perp))
    if len(ts) < 2:
        return None
    lo, hi = min(ts), max(ts)
    return (0.5 * (lo + hi), 0.5 * (hi - lo))


def is_lock(f_sp, skull_hw):
    """A front spur hanging over the head's width is a lock (a bang, a sideburn)."""
    if f_sp is None:
        return False
    dz = f_sp['tip']['z'] - f_sp['base']['z']
    du = f_sp['tip']['u'] - f_sp['base']['u']
    if not (dz < -0.01 and abs(dz) > 0.5 * abs(du)):
        return False
    return abs(f_sp['tip']['u']) < skull_hw + 0.01


def spike(name, f_sp, s_sp, x_fix, y_fix, props, hair_ref=None, lock=False, log=print, skull=None):
    """Build one spike from a front and/or side spur."""
    def extend_to_top(sp):
        if sp is None or not sp.get('poly'):
            return sp
        b = Vector((sp['base']['u'], sp['base']['z'])); t = Vector((sp['tip']['u'], sp['tip']['z']))
        ax = t - b
        if ax.length < 0.12 or (b.y - t.y) < 0.7 * ax.length:
            return sp
        d2 = ax / ax.length
        s_min = min((Vector((u, z)) - b).dot(d2) for u, z in sp['poly'])
        if s_min > -0.015:
            return sp
        nb = b + d2 * (s_min + 0.006)
        out = dict(sp)
        out['base'] = dict(u=nb.x, z=nb.y)
        out['length_m'] = (t - nb).length
        return out
    f_sp, s_sp = extend_to_top(f_sp), extend_to_top(s_sp)
    if f_sp is not None and s_sp is not None:
        # the front view weighs more where the views disagree on height
        base3 = Vector((f_sp['base']['u'], s_sp['base']['u'], Z_FRONT_W * f_sp['base']['z'] + (1 - Z_FRONT_W) * s_sp['base']['z']))
        tip3 = Vector((f_sp['tip']['u'], s_sp['tip']['u'], Z_FRONT_W * f_sp['tip']['z'] + (1 - Z_FRONT_W) * s_sp['tip']['z']))
        src = 'front+side'
    elif f_sp is not None:
        base3 = Vector((f_sp['base']['u'], y_fix, f_sp['base']['z']))
        tip3 = Vector((f_sp['tip']['u'], y_fix, f_sp['tip']['z']))
        src = 'front only, y %.3f' % y_fix
    else:
        base3 = Vector((x_fix, s_sp['base']['u'], s_sp['base']['z']))
        tip3 = Vector((x_fix, s_sp['tip']['u'], s_sp['tip']['z']))
        src = 'side only, x %.3f' % x_fix
    if lock and (hair_ref is not None or skull is not None):
        surf = hair_ref if hair_ref is not None else skull
        for pt in (base3, tip3):
            hit = surf.bvh.find_nearest(pt)
            if hit[0] is None:
                continue
            off = (pt - hit[0]).length
            if off > 0.008 or surf.in_skin(pt):
                pt.xyz = hit[0] + hit[1].normalized() * 0.003
    ax = tip3 - base3
    L3 = ax.length
    if L3 < 0.03:
        return None
    d = ax / L3
    e1 = d.cross(Vector((0.0, 1.0, 0.0)))
    if e1.length < 1e-3:
        e1 = Vector((1.0, 0.0, 0.0))
    e1.normalize()
    e2 = d.cross(e1).normalized()
    n_s = Vector((0.0, -d.z, d.y))
    if n_s.length < 1e-3:
        n_s = e2.copy()
    n_s.normalize()
    k1, k2 = e1.dot(n_s), e2.dot(n_s)
    bm = bmesh.new()
    rings = []
    secs = []
    a_prev = None
    b_prev = None
    L_est = max((f_sp or s_sp)['length_m'], (s_sp or f_sp)['length_m'])
    n_st = max(len(SPIKE_S), int(L_est / 0.035) + 1)
    stations = list(SPIKE_S) if n_st <= len(SPIKE_S) else [0.06 + (0.94 - 0.06) * i / (n_st - 1) for i in range(n_st)]
    hanging = (base3.z - tip3.z > 0.7 * L3) and L3 > 0.12
    if hanging:
        w_med = []
        for s_ in (0.2, 0.4, 0.6):
            fa_ = slice_poly(f_sp['poly'], f_sp['base'], f_sp['tip'], s_) if f_sp is not None else None
            if fa_ is not None:
                w_med.append(2.0 * fa_[1])
        if w_med and sorted(w_med)[len(w_med) // 2] < 0.02:
            hanging = False
    a_floor = 0.0
    if hanging:
        # the bow hides the tail's top: no section under 0.6 x the median
        stations = [0.012 + (0.94 - 0.012) * i / (n_st - 1) for i in range(n_st)]
        widths = []
        for s_ in stations:
            fa_ = slice_poly(f_sp['poly'], f_sp['base'], f_sp['tip'], s_) if f_sp is not None else None
            if fa_ is not None:
                widths.append(fa_[1])
        if widths:
            widths.sort()
            a_floor = 0.6 * widths[len(widths) // 2]
    for s in stations:
        fa = slice_poly(f_sp['poly'], f_sp['base'], f_sp['tip'], s) if f_sp is not None else None
        sa = slice_poly(s_sp['poly'], s_sp['base'], s_sp['tip'], s) if s_sp is not None else None
        if fa is None and sa is None:
            continue
        if fa is not None:
            cf, a = fa
        else:
            cf, a = 0.0, sa[1]
        if hanging and s < 0.3:
            a = max(a, a_floor)
        if sa is not None:
            cs, ws = sa
        else:
            cs, ws = 0.0, 0.6 * a
        if a < 0.004 and rings:
            continue
        a = max(a, 0.004)
        if a_prev is not None and a > a_prev * 1.05 and not hanging:
            a = a_prev * 1.05
        a_prev = a
        if abs(k2) > 0.25:
            b = (ws - 0.55 * a * abs(k1)) / abs(k2)
            b = min(max(b, 0.003, 0.25 * a), (1.0 if hanging else 0.6) * a)
            lam = (cs - cf * k1) / k2
            lam_max = 10.0 * a if hanging else a
            lam = min(max(lam, -lam_max), lam_max)
        else:
            b = 0.5 * a
            lam = 0.0
        # fade to the chord over the last 15 % so the tip cone does not fold back
        fade = min(1.0, (1.0 - s) / 0.15)
        if not hanging:
            cf = min(max(cf, -a), a)
        p = base3 + d * (L3 * s) + e1 * (cf * fade) + e2 * (lam * fade)
        if lock:
            b = min(b, 0.006)
        if b_prev is not None:
            b = min(max(b, 0.75 * b_prev), 1.3 * b_prev)
        b_prev = b
        # rings at least 0.6 of their width apart, or the bands shear
        if secs:
            c_prev = secs[-1][0]
            adv = (p - c_prev).dot(d)
            if adv < 0.6 * a:
                continue
        secs.append((p, a, b))
        rings.append(None)
    if len(secs) < 2:
        bm.free()
        return None
    # rotation-minimizing frames, dropping sections closer than the mitre reach r tan(theta/2)
    while True:
        path = [q[0] for q in secs]
        T_, U_, V_ = rmf(path)
        drop = None
        for i in range(1, len(secs)):
            th = math.acos(max(-1.0, min(1.0, T_[i].dot(T_[i - 1]))))
            reach = max(secs[i][1], secs[i][2], secs[i - 1][1], secs[i - 1][2]) * math.tan(min(math.radians(75), 0.5 * th)) + 0.002
            if (secs[i][0] - secs[i - 1][0]).length < reach and len(secs) > 2:
                drop = i if i < len(secs) - 1 else i - 1
                break
        if drop is None:
            break
        secs.pop(drop)
    path = [q[0] for q in secs]
    T_, U_, V_ = rmf(path)
    e1p = e1 - T_[0] * e1.dot(T_[0])
    e1p = e1p.normalized() if e1p.length > 1e-6 else U_[0]
    ca_, sa_ = e1p.dot(U_[0]), e1p.dot(V_[0])
    rings = []
    for (p, a, b), t_, u_, v_ in zip(secs, T_, U_, V_):
        f1 = (u_ * ca_ + v_ * sa_).normalized()
        f2 = t_.cross(f1).normalized()
        if f2.dot(e2) < 0 and abs(t_.dot(d)) > 0.5:
            f2 = -f2
        ring = [p + f1 * a, p + f1 * (0.35 * a) + f2 * b, p - f1 * (0.35 * a) + f2 * b,
                p - f1 * a, p - f1 * (0.35 * a) - f2 * b, p + f1 * (0.35 * a) - f2 * b]
        rings.append([bm.verts.new(q) for q in ring])
    if len(rings) < 2:
        bm.free()
        return None
    c0 = sum((v.co for v in rings[0]), Vector()) / len(rings[0])
    separate = False
    gather = None
    root_depth = 0.008 if hanging else min(SPIKE_ROOT, 0.4 * L3)
    root_flare = 1.0 if hanging else ROOT_FLARE
    if lock:
        root_depth = 0.004
        root_flare = 1.0
    shift = -d * root_depth
    if hair_ref is not None:
        found = False
        for dd in ([root_depth] if (hanging or lock) else [root_depth + 0.01 * i for i in range(0, 9)]):
            if hair_ref.in_skin(c0 - d * dd):
                shift = -d * dd
                found = True
                break
        if not found and not hanging and not lock:
            hit = hair_ref.bvh.find_nearest(c0)
            if hit[0] is not None:
                inward = hit[0] - hit[1] * 0.012
                if not hair_ref.in_skin(inward):
                    inward = hit[0] + hit[1] * 0.012
                shift = inward - c0
                # the root must lie behind the first section, else the root ring passes through it
                along = shift.dot(d)
                if along > -SPIKE_ROOT:
                    shift = shift - d * (along + SPIKE_ROOT)
    if hanging and hair_ref is not None and not hair_ref.in_skin(c0 + shift):
        hit = hair_ref.bvh.find_nearest(c0)
        if hit[0] is not None and (hit[0] - c0).length > 0.025:
            gather_from = hit[0] - hit[1] * 0.012
            if not hair_ref.in_skin(gather_from):
                gather_from = hit[0] + hit[1] * 0.012
            log('[hair] %s: tied off the head (%.0f mm from the hair body): a bunch gathers it from the head' % (name, (hit[0] - c0).length * 1000))
            hit = (None,)
            gather = gather_from
        else:
            gather = None
        if hit[0] is not None:
            inward = hit[0] - hit[1] * 0.012
            if not hair_ref.in_skin(inward):
                inward = hit[0] + hit[1] * 0.012
            bend = inward - c0
            shift = bend
            for w_, ring in zip((0.8, 0.55, 0.3, 0.1), rings[0:4]):
                for v in ring:
                    v.co += bend * w_
    c_root = c0 + shift
    if hanging and gather is not None and (c0 - gather).length > 0.02:
        # a bent tube from the hair body to the tail top, the tail's first ring on the bisector
        g_dir = (c0 - gather).normalized()
        a_r = max((v.co - c0).dot(e1) for v in rings[0])
        b_r = max((v.co - c0).dot(e2) for v in rings[0])

        def ring_frame(direction):
            f1 = e1 - direction * e1.dot(direction)
            if f1.length < 1e-3:
                f1 = direction.cross(Vector((0.0, 1.0, 0.0)))
            f1.normalize()
            return f1, direction.cross(f1).normalized()
        bis = (g_dir + d)
        bis = bis.normalized() if bis.length > 1e-6 else d
        k1_, k2_ = ring_frame(bis)
        stretch = 1.0 / max(0.5, abs(bis.dot(d)))
        for j, v in enumerate(rings[0]):
            ang = 2 * math.pi * j / 6
            v.co = c0 + k1_ * (a_r * stretch * math.cos(ang)) + k2_ * (b_r * math.sin(ang))
        half = math.acos(max(-1.0, min(1.0, g_dir.dot(d)))) * 0.5
        reach = max(a_r, b_r) * math.tan(min(math.radians(75), half)) + 0.003
        keep_rings = [rings[0]]
        n_drop = 0
        for r_ in rings[1:]:
            c_r = sum((v.co for v in r_), Vector()) / len(r_)
            if (c_r - c0).dot(d) < reach:
                for v in r_:
                    bm.verts.remove(v)
                n_drop += 1
            else:
                keep_rings.append(r_)
        rings[:] = keep_rings
        if n_drop:
            log("[hair] %s: %d tail ring(s) inside the mitre's reach (%.0f mm) dropped" % (name, n_drop, reach * 1000))
        g1, g2 = ring_frame(g_dir)
        g_len = (c0 - gather).length
        n_b = max(1, int(g_len / 0.035))
        while n_b > 1 and g_len / (n_b + 1) < reach:
            n_b -= 1
        for i_ in range(n_b):
            t_ = 1.0 - (i_ + 1) / float(n_b + 1)
            sc_ = 1.0 - 0.35 * (1.0 - t_)
            c_ = gather + (c0 - gather) * t_
            ring_ = [bm.verts.new(c_ + g1 * (a_r * sc_ * math.cos(2 * math.pi * j / 6)) + g2 * (b_r * sc_ * math.sin(2 * math.pi * j / 6))) for j in range(6)]
            rings.insert(0, ring_)
        end_ = [bm.verts.new(gather + g1 * (a_r * 0.5 * math.cos(2 * math.pi * j / 6)) + g2 * (b_r * 0.5 * math.sin(2 * math.pi * j / 6))) for j in range(6)]
        rings.insert(0, end_)
    elif lock:
        # a lock's first section is its root
        pass
    else:
        root = [bm.verts.new(c_root + (v.co - c0) * root_flare) for v in rings[0]]
        rings.insert(0, root)
    c_last = sum((v.co for v in rings[-1]), Vector()) / len(rings[-1])
    if (tip3 - c_last).dot(d) < 0.004:
        tip3 = c_last + d * 0.004
    tip_v = bm.verts.new(tip3)
    bm.faces.new(rings[0][::-1])
    for A, B in zip(rings, rings[1:]):
        for j in range(6):
            bm.faces.new([A[j], A[(j + 1) % 6], B[(j + 1) % 6], B[j]])
    for j in range(6):
        bm.faces.new([rings[-1][j], rings[-1][(j + 1) % 6], tip_v])
    ob, nf = close_and_emit(name, bm, dict(props, s3d_kind='spike', s3d_separate=int(separate)), log)
    log('[hair] %-13s L %.0f mm  base (%.3f %.3f %.3f) tip (%.3f %.3f %.3f)  %d faces  [%s%s]' % (
        name, L3 * 1000, base3.x, base3.y, base3.z, tip3.x, tip3.y, tip3.z, nf, src, ', lock' if lock else ''))
    return ob


def pair_spurs(fr, sd, log=print):
    """Pair front and side spurs by tip and base height, cheapest first."""
    def down(sp):
        dz = sp['tip']['z'] - sp['base']['z']
        du = sp['tip']['u'] - sp['base']['u']
        return dz < -0.01 and abs(dz) > 0.5 * abs(du)
    def vclass(sp):
        v = (sp['tip']['z'] - sp['base']['z']) / max(1e-6, sp['length_m'])
        return 'up' if v > 0.3 else ('down' if v < -0.3 else 'level')
    cands = []
    for i, f in enumerate(fr):
        for j, s in enumerate(sd):
            cf, cs = vclass(f), vclass(s)
            if cf != cs and 'level' not in (cf, cs):
                continue
            c = abs(f['tip']['z'] - s['tip']['z']) + 0.5 * abs(f['base']['z'] - s['base']['z'])
            c += 0.05 * abs(math.log(max(1e-3, f['length_m']) / max(1e-3, s['length_m'])))
            cands.append((c, i, j))
    cands.sort()
    used_f, used_s, pairs = set(), set(), []
    for c, i, j in cands:
        if c > PAIR_MAX or i in used_f or j in used_s:
            continue
        used_f.add(i); used_s.add(j)
        pairs.append((i, j, c))
        log('[hair] pair front %d <-> side %d  cost %.3f' % (i, j, c))
    return pairs, used_f, used_s


def plate(name, P, body, props, log=print):
    """Build a hair plate from the side silhouette, as wide as drawn in front and sunk into the skull."""
    poly = [(float(y), float(z)) for y, z in P['poly']]
    if len(poly) >= 2 and abs(poly[0][0] - poly[-1][0]) < 1e-6 and abs(poly[0][1] - poly[-1][1]) < 1e-6:
        poly = poly[:-1]
    clean = []
    for q in poly:
        if not clean or (abs(q[0] - clean[-1][0]) > 1e-4 or abs(q[1] - clean[-1][1]) > 1e-4):
            clean.append(q)
    poly = clean
    if len(poly) < 3:
        return None
    th = sorted(P['thick'], key=lambda t: t['z'])
    z_lo, z_hi = min(z for _, z in poly), max(z for _, z in poly)

    def width_at(z):
        if not th:
            return 0.02
        if z <= th[0]['z']:
            w = th[0]['w']
        elif z >= th[-1]['z']:
            w = th[-1]['w']
        else:
            w = th[-1]['w']
            for a, b in zip(th, th[1:]):
                if a['z'] <= z <= b['z']:
                    t = (z - a['z']) / max(1e-9, b['z'] - a['z'])
                    w = a['w'] + (b['w'] - a['w']) * t
                    break
        f = (z - z_lo) / max(1e-6, z_hi - z_lo)
        return max(0.006, w * (1.0 - 0.55 * max(0.0, f - 0.6) / 0.4))

    def x_at(z):
        if not th:
            return 0.0
        q = min(th, key=lambda t: abs(t['z'] - z))
        return q.get('x', 0.0)
    bm = bmesh.new()
    vs = [bm.verts.new(Vector((0.0, y, z))) for y, z in poly]
    try:
        bm.faces.new(vs)
    except ValueError:
        bm.free()
        return None
    long_e = [e for e in bm.edges if e.calc_length() > 0.03]
    if long_e:
        bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], ngon_method='BEAUTY')
    # crest notches are design: mark them as creases for the judge
    cr = bm.verts.layers.int.new('s3d_crease')
    for v in bm.verts:
        v[cr] = 1
    front = [v for v in bm.verts]
    geom = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:] + bm.edges[:] + bm.verts[:])
    back = [g for g in geom['geom'] if isinstance(g, bmesh.types.BMVert)]
    for v in front:
        v.co.x = x_at(v.co.z) + 0.5 * width_at(v.co.z)
    for v in back:
        v.co.x = x_at(v.co.z) - 0.5 * width_at(v.co.z)
    n_sunk = 0
    for v in bm.verts:
        hit = body.bvh.ray_cast(Vector((v.co.x, v.co.y, v.co.z + 0.5)), Vector((0, 0, -1)), 1.0)
        if hit[0] is None:
            continue
        z_sk = hit[0].z
        if v.co.z <= z_sk + 0.02:
            v.co.z = min(v.co.z, z_sk - 0.004)
            n_sunk += 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if signed_volume(bm) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:], ngon_method='BEAUTY')
    ob, _nf = close_and_emit(name, bm, props, log)
    log('[hair] %s: plate %s, %d faces, %.0f..%.0f mm thick, %d vertices sunk into the skull' % (
        name, P['cls'], len(ob.data.polygons), 1000 * min(width_at(z) for _, z in poly), 1000 * max(width_at(z) for _, z in poly), n_sunk))
    return ob


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, out_blend = argv[:4]
    C = json.load(open(char_path))
    M = json.load(open(map_path))
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    body_ob = bpy.data.objects['Body']
    body = BodyRef(body_ob)
    H = C['hair']
    sk = C['skull']
    z_hair = sk['hairline_z']
    cap_f = H.get('cap', {}).get('front', [])
    cap_s = H.get('cap', {}).get('side', [])
    dome_y = M['dome']['y']
    props = dict(s3d_part='hair', s3d_class='hair')
    made = []

    plates_made = []
    for k_, P in enumerate(H.get('plates', [])):
        pob = plate('Hair_Plate_%d' % k_, P, body, dict(s3d_part='hair', s3d_class=P['cls'], s3d_kind='plate', s3d_over='Hair,Body'), log=print)
        if pob is not None:
            plates_made.append(pob)
    sheet = Sheet(char_path)
    if not H.get('cap'):
        from carve import triangulate
        for ob in plates_made:
            triangulate(ob, log=print)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        print('[hair] %d objects (plates only) -> %s' % (len(plates_made), out_blend))
        return
    hob = hair_body(H, sk, body, M, z_hair, props, log=print, sheet=sheet, C_roles=C['roles'])
    if hob is None:
        raise SystemExit('[hair] no hair body could be read (need side core + front/back core)')
    made.append(hob)
    hair_ref = BodyRef(hob)

    skull_hw0 = 0.5 * sk['w_mm'] / 1000.0
    ppm_ = C['calibration']['ppm']

    def thin(sp):
        # a strand under 8 mm wide is drawn by the texture
        if not sp.get('px') or not sp.get('length_m'):
            return False
        return sp['px'] / max(1.0, sp['length_m'] * ppm_) / ppm_ < 0.008
    fr = [f for f in H.get('spurs', {}).get('front', []) if len(f.get('poly', [])) >= 3
          and not (f['length_m'] < 0.03 and abs(f['tip']['u']) < skull_hw0) and not thin(f)]
    sd = [s for s in H.get('spurs', {}).get('side', []) if len(s.get('poly', [])) >= 3 and not thin(s)]
    n_thin = sum(1 for f in H.get('spurs', {}).get('front', []) if thin(f)) + sum(1 for s_ in H.get('spurs', {}).get('side', []) if thin(s_))
    if n_thin:
        print('[hair] %d strand(s) thinner than 8 mm left to the texture' % n_thin)
    pairs, used_f, used_s = pair_spurs(fr, sd, log=print)
    k = 0
    skull_hw = 0.5 * sk['w_mm'] / 1000.0
    for i, j, c in pairs:
        ob = spike('Hair_Spike_%d' % k, fr[i], sd[j], None, None, props, hair_ref=hair_ref, lock=is_lock(fr[i], skull_hw), log=print, skull=body)
        if ob is not None:
            made.append(ob); k += 1
    partner = {i: j for i, j, c in pairs}
    for i, f in enumerate(fr):
        if i in used_f:
            continue
        twin = None
        for i2, j2 in partner.items():
            f2 = fr[i2]
            if abs(f2['tip']['u'] + f['tip']['u']) < 0.03 and abs(f2['tip']['z'] - f['tip']['z']) < 0.03 and abs(f2['length_m'] - f['length_m']) < 0.15 * max(f['length_m'], f2['length_m']):
                twin = j2
                break
        if twin is not None:
            ob = spike('Hair_Spike_%d' % k, f, sd[twin], None, None, props, hair_ref=hair_ref, lock=is_lock(f, skull_hw), log=print, skull=body)
            if ob is not None:
                made.append(ob); k += 1
            continue
        sy = table_at(cap_s, f['base']['z'])
        y_fix = 0.5 * (sy[0] + sy[1]) if sy else dome_y
        ob = spike('Hair_Spike_%d' % k, f, None, None, y_fix, props, hair_ref=hair_ref, lock=is_lock(f, skull_hw), log=print, skull=body)
        if ob is not None:
            made.append(ob); k += 1
    # a side-only spur: fringe or nape spike on the midline, otherwise a side lock and its twin
    sk_front = dome_y - 0.5 * M['dome']['d']
    sk_back = dome_y + 0.5 * M['dome']['d']
    for j, s in enumerate(sd):
        if j in used_s:
            continue
        fx = table_at(cap_f, s['base']['z'])
        x_mid = 0.5 * (fx[0] + fx[1]) if fx else 0.0
        ty = s['tip']['u']
        if ty < sk_front + 0.01 or ty > sk_back - 0.01:
            ob = spike('Hair_Spike_%d' % k, None, s, x_mid, None, props, hair_ref=hair_ref, log=print)
            if ob is not None:
                made.append(ob); k += 1
            continue
        se = skull_extent(body, M['head'], 0.5 * (s['base']['z'] + s['tip']['z']))
        hw = 0.5 * (se[1] - se[0]) if se else skull_hw0
        x_vis = (hw + 0.004) * (1.0 if C['datum'].get('side_sign', 1.0) > 0 else -1.0)
        ob = spike('Hair_Spike_%d' % k, None, s, x_vis, None, props, hair_ref=hair_ref, lock=True, log=print, skull=body)
        if ob is not None:
            made.append(ob); k += 1
    # lift spikes out of the skin as a whole: neighbours within 15 mm move by the same lift
    from mathutils.kdtree import KDTree
    for ob in made[1:]:
        verts = ob.data.vertices
        lifts = {}
        for i, v in enumerate(verts):
            co = body.lift_out(v.co.copy(), CLEAR)
            if co is not None:
                lifts[i] = co - v.co
        if not lifts:
            continue
        kd = KDTree(len(verts))
        for i, v in enumerate(verts):
            kd.insert(v.co, i)
        kd.balance()
        move = {}
        for i, lv in lifts.items():
            for (_, j, d_) in kd.find_range(verts[i].co, 0.015):
                if j not in move or move[j].length < lv.length:
                    move[j] = lv
        for j, mv in move.items():
            verts[j].co = verts[j].co + mv
        ob.data.update()

    pin_edge_to_skin(hob, body.bvh, max_dist=0.012, lift=0.0025, log=print)
    fix_selfx(hob, log=print)
    kept = made[1:]
    for k_, ob in enumerate(kept):
        ob['s3d_kind'] = 'spike'
    made = plates_made + [hob] + kept
    # triangulate first: a non-planar quad's halves can cross a neighbour
    from carve import triangulate
    for ob in made:
        triangulate(ob, log=print)
        fix_selfx(ob, log=print)

    tris = 0
    for ob in made:
        t = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        tris += t
        print('[hair] %-14s %4d tris' % (ob.name, t))
    print('[hair] %d objects, %d tris -> %s' % (len(made), tris, out_blend))
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)


if __name__ == '__main__':
    main()
