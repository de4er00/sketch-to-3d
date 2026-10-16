"""Build accessories without a recipe: thin bands as ribbons, the rest as visual hulls."""
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

RIBBON_W = 0.036  # metres, thinner everywhere after erosion is a band
RIBBON_T = 0.003
RIBBON_LIFT = 0.0025
MIN_PX_MM = 12


def integral(m):
    return np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def box_survivors(m, e):
    """Pixels of m whose (2e+1)^2 box lies wholly inside m."""
    H, W = m.shape
    I = integral(m)
    k = 2 * e + 1
    if H < k or W < k:
        return np.zeros_like(m)
    s = I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
    out = np.zeros_like(m)
    out[e:H - e, e:W - e] = s == k * k
    return out


def components(m, min_side_px):
    from carve import label8
    lab = label8(m)
    out = []
    for k in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < min_side_px * min_side_px:
            continue
        out.append((ys, xs))
    return out


def is_ribbon(ys, xs, ppm, name):
    n = name.lower()
    import re
    if re.search(r'strap|belt|band|ribbon|cord|lace|chain|string|wire|sash', n):
        return True
    if re.search(r'backpack|bag|pouch|pack|helmet|hat|cap|box|case|bottle|bow', n):
        return False
    r0, c0 = ys.min(), xs.min()
    sub = np.zeros((ys.max() - r0 + 1, xs.max() - c0 + 1), bool)
    sub[ys - r0, xs - c0] = True
    e = max(2, int(round(0.5 * RIBBON_W * ppm)))
    surv = box_survivors(sub, e).sum()
    area = len(ys)
    return surv < 0.05 * area and area >= 3 * (2 * e) ** 2


class Scene:
    """Outermost built surface for ray casts, optionally without accessory volumes."""
    def __init__(self, exclude_cls, no_volumes=False):
        bm = bmesh.new()
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.get('s3d_class') == exclude_cls or not len(o.data.polygons):
                continue
            if no_volumes and o.get('s3d_part') == 'accessory' and o.get('s3d_kind') == 'volume':
                continue
            t = bmesh.new(); t.from_mesh(o.data); t.transform(o.matrix_world)
            for f in t.faces:
                try:
                    bm.faces.new([bm.verts.new(v.co) for v in f.verts])
                except ValueError:
                    pass
            t.free()
        bm.normal_update()
        self.bvh = BVHTree.FromBMesh(bm) if len(bm.faces) else None
        bm.free()

    def hit(self, origin, direction, dist=3.0):
        if self.bvh is None:
            return None, None
        loc, nrm, _i, _d = self.bvh.ray_cast(origin, direction, dist)
        if loc is None:
            return None, None
        if nrm is None or nrm.length < 1e-6:
            nrm = -direction.normalized()  # degenerate normal: face the camera
        if nrm.dot(direction) > 0:
            nrm = -nrm
        return loc, nrm.normalized()

    def nearest(self, p):
        if self.bvh is None:
            return None
        return self.bvh.find_nearest(p)[0]


def view_frame(sheet, view):
    if view == 'front':
        mid = sheet.pan['front']['midline']
        return (lambda c: (c - mid) / sheet.ppm), -1.5, Vector((0, 1, 0))
    mid = sheet.pan['back']['midline']
    return (lambda c: (mid - c) / sheet.ppm), 1.5, Vector((0, -1, 0))


def ribbon(name, view, ys, xs, sheet, scene, props, log=print):
    """A strip RIBBON_T thick cast onto the surface under the band, sampled every 4 mm."""
    x_of, y0, d = view_frame(sheet, view)
    ppm = sheet.ppm
    h_px, w_px = ys.max() - ys.min(), xs.max() - xs.min()
    along_rows = h_px >= w_px
    step = max(2, int(round(0.004 * ppm)))
    samples = []
    if along_rows:
        for r in range(ys.min(), ys.max() + 1, step):
            cs = xs[ys == r]
            if len(cs) < 2:
                continue
            # the widest run, so a fork keeps its main branch
            cs = np.sort(cs)
            breaks = np.nonzero(np.diff(cs) > 2)[0]
            runs = np.split(cs, breaks + 1)
            run = max(runs, key=len)
            samples.append((float(r), 0.5 * (run[0] + run[-1]), 0.5 * (run[-1] - run[0])))
    else:
        for c in range(xs.min(), xs.max() + 1, step):
            rs = ys[xs == c]
            if len(rs) < 2:
                continue
            rs = np.sort(rs)
            breaks = np.nonzero(np.diff(rs) > 2)[0]
            runs = np.split(rs, breaks + 1)
            run = max(runs, key=len)
            samples.append((0.5 * (run[0] + run[-1]), float(c), 0.5 * (run[-1] - run[0])))
    if len(samples) < 2:
        return None
    bm = bmesh.new()
    rings = []
    n_miss = 0
    for (r, c, hw) in samples:
        hw_m = max(0.003, hw / ppm)
        X, Z = x_of(c), (sheet.floor - r) / ppm
        if along_rows:
            pl = (X - hw_m, Z); pr = (X + hw_m, Z)
        else:
            pl = (X, Z + hw_m); pr = (X, Z - hw_m)
        cen_hit, cen_n = scene.hit(Vector((X, y0, Z)), d)
        if cen_hit is None:
            n_miss += 1
            continue
        pts = []
        for (px, pz) in (pl, pr):
            h, n = scene.hit(Vector((px, y0, pz)), d)
            if h is None or abs(h.y - cen_hit.y) > 0.06:
                h = Vector((px, cen_hit.y, pz)); n = cen_n
            pts.append((h, n))
        (hl, nl), (hr, nr) = pts
        ring = (hl + nl * (RIBBON_LIFT + RIBBON_T), hr + nr * (RIBBON_LIFT + RIBBON_T),
                hl + nl * RIBBON_LIFT, hr + nr * RIBBON_LIFT)
        if rings and max((a - b).length for a, b in zip(ring, rings[-1])) < 0.0015:
            continue
        if (ring[0] - ring[1]).length < 0.002:
            continue
        rings.append(ring)
    if len(rings) < 2:
        bm.free()
        return None
    V = [[bm.verts.new(p) for p in ring] for ring in rings]
    for a, b in zip(V, V[1:]):
        for (i, j) in ((0, 1), (3, 2), (2, 0), (1, 3)):
            try:
                bm.faces.new([a[i], a[j], b[j], b[i]])
            except ValueError:
                pass
    for ring, flip in ((V[0], False), (V[-1], True)):
        try:
            bm.faces.new([ring[0], ring[1], ring[3], ring[2]][::-1] if flip else [ring[0], ring[1], ring[3], ring[2]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # a thin closed strip: the signed volume decides the winding
    vol = 0.0
    for f in bm.faces:
        vs = [v.co for v in f.verts]
        for i in range(1, len(vs) - 1):
            vol += vs[0].dot(vs[i].cross(vs[i + 1])) / 6.0
    if vol < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    from shell import emit
    ob = emit(name, bm, dict(props, s3d_kind='ribbon', s3d_view=view))
    log('[attach] %s: ribbon in the %s view, %d samples (%d off the scene), %.0f mm wide' % (
        name, view, len(rings), n_miss, 2000 * np.median([s[2] for s in samples]) / ppm))
    return ob


TOL_FRAC = 0.005  # an outline vertex must move the silhouette by max(2 mm, 0.5 % of the size)
SIZE_M = [1.6]


def simple_polygon(poly):
    """True when no two non-adjacent edges cross."""
    n = len(poly)
    if n < 4:
        return True

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def cross(a, b, c, d):
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if cross(a, b, c, d):
                return False
    return True


def outline(ys, xs, ppm):
    from accessories import trace_outline, star_outline, simplify_ring, convex_hull
    tol = max(0.002, TOL_FRAC * SIZE_M[0]) * ppm
    pts2 = np.stack([xs, ys], 1).astype(float)
    hull = trace_outline(pts2, tol)
    if len(hull) < 3 or len(hull) > 48:
        hull = star_outline(pts2, 32, tol)
    if len(hull) < 3 or not simple_polygon(hull):
        hull = simplify_ring(convex_hull(pts2), 1.5 * tol)
    return hull


def host_of(sheet, view, ys, xs, cls):
    """The class the component touches most, in a 6 px ring."""
    lab = sheet.lab
    H, W = lab.shape
    r0, r1 = max(0, ys.min() - 6), min(H, ys.max() + 7)
    c0, c1 = max(0, xs.min() - 6), min(W, xs.max() + 7)
    sub = np.zeros((r1 - r0, c1 - c0), bool)
    sub[ys - r0, xs - c0] = True
    from carve import dilate8
    g = sub.copy()
    for _ in range(6):
        g = dilate8(g)
    ring = lab[r0:r1, c0:c1][g & ~sub]
    ids = {v: k for k, v in sheet.ids.items()}
    counts = {}
    for v in ring.tolist():
        n_ = ids.get(int(v))
        if n_ and n_ != cls:
            counts[n_] = counts.get(n_, 0) + 1
    return max(counts, key=counts.get) if counts else None


def volume(name, view, ys, xs, side_comp, sheet, scene, props, log=print):
    from accessories import prism, visual_hull
    ppm = sheet.ppm
    x_of, y0, d = view_frame(sheet, view)
    host = host_of(sheet, view, ys, xs, props.get('s3d_class', ''))
    hair_cls = set((sheet.C.get('roles') or {}).get('hair', [])) if hasattr(sheet, 'C') else set()
    props = dict(props, s3d_host=host or '', s3d_tied=int(host in hair_cls))
    hull = outline(ys, xs, ppm)
    if len(hull) < 3:
        return None
    f_poly = [Vector((x_of(cx), 0.0, (sheet.floor - ry) / ppm)) for (cx, ry) in hull]
    xs_ = [q.x for q in f_poly]
    if side_comp is not None:
        sy, sx = side_comp
        s_hull = outline(sy, sx, ppm)
        if len(s_hull) >= 3:
            s_poly = [Vector((0.0, (c_px - sheet.dcol) / ppm * sheet.side_sign, (sheet.floor - r_px) / ppm)) for (c_px, r_px) in s_hull]
            ys_ = [q.y for q in s_poly]
            bm_a = prism(f_poly, 1, min(ys_) - 0.004, max(ys_) + 0.004)
            bm_b = prism(s_poly, 0, min(xs_) - 0.004, max(xs_) + 0.004)
            ob, ok = visual_hull(name, bm_a, bm_b, dict(props, s3d_kind='volume', s3d_view=view), log=log)
            log('[attach] %s: volume from the %s view x side, depth %.0f..%.0f mm%s' % (name, view, 1000 * min(ys_), 1000 * max(ys_), '' if ok else ' (single prism)'))
            return ob
    # no side outline: a slab half its width deep
    cen = Vector((float(np.mean(xs_)), 0.0, float(np.mean([q.z for q in f_poly]))))
    hit, n = scene.hit(Vector((cen.x, y0, cen.z)), d)
    y_face = hit.y if hit is not None else 0.0
    depth = 0.5 * (max(xs_) - min(xs_))
    if view == 'front':
        bm = prism(f_poly, 1, y_face - depth - 0.002, y_face - 0.002)
    else:
        bm = prism(f_poly, 1, y_face + 0.002, y_face + depth + 0.002)
    from shell import emit
    ob = emit(name, bm, dict(props, s3d_kind='volume', s3d_view=view))
    log('[attach] %s: slab from the %s view alone, %.0f mm deep' % (name, view, 1000 * depth))
    return ob


def touch(ob, scene, log=print):
    """Slide a volume that touches nothing to the nearest surface, 3 mm apart."""
    if scene.bvh is None:
        return
    best = None
    for v in ob.data.vertices:
        p = scene.nearest(v.co)
        if p is None:
            continue
        dd = (p - v.co).length
        if best is None or dd < best[0]:
            best = (dd, p - v.co)
    if best is not None and best[0] > 0.02:
        mv = best[1] - best[1].normalized() * 0.003
        for v in ob.data.vertices:
            v.co = v.co + mv
        ob.data.update()
        log('[attach] %s: slid %.0f mm to touch the scene' % (ob.name, 1000 * mv.length))


def stand_off(ob, cls, rigid, log=print):
    from carve import clear_from_meshes
    others = [o for o in bpy.data.objects if o.type == 'MESH' and o.get('s3d_class') != cls and o is not ob]
    clear_from_meshes(ob, others, gap=0.0025 if not rigid else 0.003, log=log, rigid=rigid)


def settle(ob, cls, gap=0.003, log=print):
    """Move a volume pushed clear back toward the nearest surface until `gap` away."""
    dg = bpy.context.evaluated_depsgraph_get()
    others = [o for o in bpy.data.objects if o.type == 'MESH' and o.get('s3d_class') != cls and o is not ob]
    trees = [BVHTree.FromObject(o, dg) for o in others]
    base = [v.co.copy() for v in ob.data.vertices]

    def place(offset):
        for v, c in zip(ob.data.vertices, base):
            v.co = c + offset
        ob.data.update()

    def crosses():
        mine = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
        return any(mine.overlap(t) for t in trees)
    total = Vector()
    for _ in range(3):
        best = None
        for v in ob.data.vertices:
            for t in trees:
                hit = t.find_nearest(v.co)
                if hit[0] is not None and (best is None or hit[3] < best[0]):
                    best = (hit[3], hit[0] - v.co)
        if best is None or best[0] <= gap + 0.001:
            break
        d = best[1].normalized()
        lo, hi = 0.0, best[0] - gap
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            place(total + d * mid)
            if crosses():
                hi = mid
            else:
                lo = mid
        total = total + d * lo
        place(total)
    if total.length > 0.0005:
        log('[attach] %s: settled %.1f mm back onto the surface' % (ob.name, 1000 * total.length))


def build_class(C, sheet, cls, props, log=print):
    ppm = sheet.ppm
    SIZE_M[0] = float(C.get('height_m') or 1.6)
    scene = Scene(cls)
    cid = sheet.ids.get(cls)
    out = []
    painted = []
    if cid is None:
        return out
    min_side = int(round(MIN_PX_MM / 1000.0 * ppm))
    comps = {}
    for view in ('front', 'back', 'side'):
        if view not in sheet.panels:
            continue
        c0, c1 = sheet.panels[view]
        m = np.zeros(sheet.lab.shape, bool)
        m[:, c0:c1] = sheet.lab[:, c0:c1] == cid
        comps[view] = components(m, min_side)
    side_used = set()
    k = 0
    for view in ('front', 'back'):
        for (ys, xs) in comps.get(view, []):
            name = '%s_%d' % (cls.capitalize(), k)
            if is_ribbon(ys, xs, ppm, cls):
                # a band on a surface is texture
                painted.append(view)
                ob = None
            else:
                r_lo, r_hi = int(ys.min()), int(ys.max())
                best, best_len = None, 0
                for j, (sy, sx) in enumerate(comps.get('side', [])):
                    if j in side_used:
                        continue
                    lo_, hi_ = max(r_lo, int(sy.min())), min(r_hi, int(sy.max()))
                    if hi_ - lo_ >= 0.5 * min(r_hi - r_lo, int(sy.max()) - int(sy.min())) and len(sy) > best_len:
                        best, best_len = j, len(sy)
                side_comp = comps['side'][best] if best is not None else None
                ob = volume(name, view, ys, xs, side_comp, sheet, scene, props, log=log)
                if ob is not None:
                    if best is not None:
                        side_used.add(best)
                    touch(ob, scene, log=log)
            if ob is not None:
                if ob.get('s3d_kind') == 'volume':
                    stand_off(ob, cls, rigid=True, log=log)
                    settle(ob, cls, log=log)
                out.append(ob)
                k += 1
    n_side_only = len(comps.get('side', [])) - len(side_used)
    if not out and not painted:
        n_px = sum(int((sheet.lab[:, sheet.panels[v][0]:sheet.panels[v][1]] == cid).sum()) for v in comps)
        if n_px and not any(comps.values()):
            painted.append('front')
            log('[attach] %s: %d px, under %d mm everywhere - texture, no geometry' % (cls, n_px, MIN_PX_MM))
    if n_side_only > 0 and not out and not painted:
        big = max((max(sy.max() - sy.min(), sx.max() - sx.min()) for (sy, sx) in comps.get('side', [])), default=0) / ppm
        area = sum(len(sy) for (sy, sx) in comps.get('side', [])) / (ppm * ppm)
        if big < 0.025 or area < 0.0016:
            painted.append('side')
            log('[attach] %s: only small pieces (%.0f mm) in the side view - texture, no geometry' % (cls, 1000 * big))
        else:
            log('[attach] %s: drawn in the side view only (%d piece(s)) - nothing built from that alone' % (cls, n_side_only))
    if painted:
        body = bpy.data.objects.get('Body')
        if body is not None:
            prev = [q for q in (body.get('s3d_painted') or '').split(',') if q]
            if cls not in prev:
                body['s3d_painted'] = ','.join(prev + [cls])
        log('[attach] %s: a band - painted onto the surfaces under it (%s view%s), no geometry' % (cls, '/'.join(sorted(set(painted))), 's' if len(set(painted)) > 1 else ''))
    return out
