"""Build a rigid prop from prop.json: hulls, solids of revolution and prisms, then one atlas.

    blender -b --factory-startup --python sketch3d/build/prop_build.py -- <prop.json> <out.blend>
"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Vector

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from shell import emit                                     # noqa: E402
from accessories import prism, visual_hull, trace_outline, star_outline, simplify_ring, convex_hull   # noqa: E402
from carve import label8                                    # noqa: E402
import polish as PL                                         # noqa: E402

LATHE_TOL = 0.20
MIN_MM = 6  # under 6 x 6 mm or 0.4 % of the size is a speck
import re
# two views cannot tell a cylinder from a box, so revolve only round names or circular top views
ROUND_WORDS = r'round|disc|disk|record|wheel|ring|handle|grip|hilt|pipe|tube|pole|post|rod|bottle|barrel|can|cup|ball|sphere|knob|pommel|cylinder|coin|button|lens|tyre|tire|drum|bolt|nut|screw|cone|vase|pot|mug|lamp|bulb|orb|gem|bead|helmet|dome'


def round_named(name):
    words = ROUND_WORDS.replace(chr(92) + 'b', '')
    return re.search(r'\b(' + words + r')\b', name.lower().replace('_', ' ')) is not None



def fill_holes(m):
    """Fill holes not connected to the image border: a part drawn on a face hides it, not removes it."""
    bg = label8(~m)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]).tolist())) - {0}
    holes = (bg > 0) & ~np.isin(bg, list(border))
    return m | holes


def filled(ys, xs, close_px=0):
    """The component with holes filled and notches under 2 x close_px closed -> (ys, xs)."""
    from carve import dilate8
    pad = close_px + 1
    r0, c0 = ys.min(), xs.min()
    sub = np.zeros((ys.max() - r0 + 2 * pad + 1, xs.max() - c0 + 2 * pad + 1), bool)
    sub[ys - r0 + pad, xs - c0 + pad] = True
    if close_px > 0:
        g = sub.copy()
        for _ in range(close_px):
            g = dilate8(g)
        h = ~g
        for _ in range(close_px):
            h = dilate8(h)
        sub = sub | ~h
    sub = fill_holes(sub)
    fy, fx = np.nonzero(sub)
    return fy + r0 - pad, fx + c0 - pad


def absorb_occluders(view_lab, ys, xs, cid, min_frac=0.5):
    """Grow the component by other-class blobs whose border lies mostly on it -> (ys, xs)."""
    from carve import dilate8
    H, W = view_lab.shape
    comp = np.zeros((H, W), bool)
    comp[ys, xs] = True
    others = (view_lab > 0) & (view_lab != cid)
    r0, r1 = max(0, ys.min() - 4), min(H, ys.max() + 5)
    c0, c1 = max(0, xs.min() - 4), min(W, xs.max() + 5)
    sub = others[r0:r1, c0:c1]
    lab = label8(sub)
    grown = comp.copy()
    n_abs = 0
    for k in range(1, lab.max() + 1):
        blob = np.zeros((H, W), bool)
        blob[r0:r1, c0:c1] = lab == k
        ring = dilate8(blob) & ~blob
        n_ring = ring.sum()
        if n_ring and (ring & comp).sum() >= min_frac * n_ring:
            grown |= blob
            n_abs += 1
    gy, gx = np.nonzero(grown)
    return gy, gx, n_abs


def components(m, min_px):
    lab = label8(m)
    out = []
    for k in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) >= min_px * min_px:
            out.append((ys, xs))
    return out


def holes_of(ys, xs, ppm):
    r0, c0 = ys.min(), xs.min()
    sub = np.zeros((ys.max() - r0 + 3, xs.max() - c0 + 3), bool)
    sub[ys - r0 + 1, xs - c0 + 1] = True
    bg = label8(~sub)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]).tolist())) - {0}
    out = []
    for k in range(1, bg.max() + 1):
        if k in border:
            continue
        hy, hx = np.nonzero(bg == k)
        if len(hy) < (0.006 * ppm) ** 2:
            continue
        h = outline(hy + r0 - 1, hx + c0 - 1, ppm)
        if len(h) >= 3:
            out.append(h)
    return out


def prism_holes(outer3, holes3, axis, a0, a1):
    bm = bmesh.new()
    loops = []
    for poly in [outer3] + holes3:
        vs = [bm.verts.new(q) for q in poly]
        es = []
        for i in range(len(vs)):
            try:
                es.append(bm.edges.new((vs[i], vs[(i + 1) % len(vs)])))
            except ValueError:
                pass
        loops.append(es)
    res = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=[e for es in loops for e in es])
    faces = [f for f in res.get('geom', []) if isinstance(f, bmesh.types.BMFace)]
    if not faces:
        bm.free()
        return None
    for v in bm.verts:
        v.co[axis] = a0
    geom = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:] + bm.edges[:] + bm.verts[:])
    new_vs = [g for g in geom['geom'] if isinstance(g, bmesh.types.BMVert)]
    for v in new_vs:
        v.co[axis] = a1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def stroke_px(ys, xs):
    from attach import box_survivors
    r0, c0 = ys.min(), xs.min()
    sub = np.zeros((ys.max() - r0 + 1, xs.max() - c0 + 1), bool)
    sub[ys - r0, xs - c0] = True
    e = 1
    while e < 200 and box_survivors(sub, e).sum() >= 0.02 * len(ys):
        e += 1
    return 2 * e - 1


def thin_structure(ys, xs, ppm, size_m):
    """Strokes: under 5 % survives erosion by 1.5 % of the size, yet it spans over 6 %."""
    from attach import box_survivors
    r0, c0 = ys.min(), xs.min()
    sub = np.zeros((ys.max() - r0 + 1, xs.max() - c0 + 1), bool)
    sub[ys - r0, xs - c0] = True
    e = max(2, int(round(0.015 * size_m * ppm)))
    span = max(sub.shape) / ppm
    # outline length per area: ~50 for a 10:1 bar, over 100 for a frame, 16 for a box
    hull = outline(ys, xs, ppm)
    per = sum(math.hypot(hull[i][0] - hull[i - 1][0], hull[i][1] - hull[i - 1][1]) for i in range(len(hull))) if len(hull) >= 3 else 0.0
    compact = per * per / max(1.0, len(ys))
    return box_survivors(sub, e).sum() < 0.05 * len(ys) and span > 0.06 * size_m and compact >= 80


SIZE_M = [1.0]


def outline(ys, xs, ppm):
    """Outline with a vertex only where it moves the silhouette by max(2 mm, 0.5 % of the size)."""
    tol = max(0.002, 0.005 * SIZE_M[0]) * ppm
    pts2 = np.stack([xs, ys], 1).astype(float)
    hull = trace_outline(pts2, tol)
    if len(hull) < 3 or len(hull) > 64:
        hull = star_outline(pts2, 48, tol)
    if len(hull) < 3:
        hull = simplify_ring(convex_hull(pts2), 1.5 * tol)
    return hull


def row_widths(ys, xs):
    out = {}
    for r in np.unique(ys):
        cs = xs[ys == r]
        out[int(r)] = (int(cs.min()), int(cs.max()))
    return out


def axis_wander(W_):
    rows = sorted(W_)
    wmax = max(W_[r][1] - W_[r][0] for r in rows)
    cs = [0.5 * (W_[r][0] + W_[r][1]) for r in rows if (W_[r][1] - W_[r][0]) >= 0.5 * wmax]
    if len(cs) < 3:
        return 1.0
    return float(np.percentile(cs, 90) - np.percentile(cs, 10)) / max(1.0, wmax)


def circular(ys, xs):
    w = xs.max() - xs.min() + 1; h = ys.max() - ys.min() + 1
    fill = len(ys) / float(w * h)
    return 0.68 <= fill <= 0.88 and 0.8 <= w / float(h) <= 1.25


def simplify_profile(prof, tol=0.0004, max_gap=0.04):
    """Douglas-Peucker over a profile [(t, r1[, r2])], with a ring at least every max_gap."""
    if len(prof) <= 2:
        return prof
    keep = [0, len(prof) - 1]

    def dev(i, a, b):
        pa, pb, p = prof[a], prof[b], prof[i]
        den = pb[0] - pa[0]
        t = (p[0] - pa[0]) / den if abs(den) > 1e-9 else 0.0
        return max(abs(p[k] - (pa[k] + (pb[k] - pa[k]) * t)) for k in range(1, len(p)))
    stack = [(0, len(prof) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        i_max = max(range(a + 1, b), key=lambda i: dev(i, a, b))
        if dev(i_max, a, b) > tol:
            keep.append(i_max)
            stack.append((a, i_max)); stack.append((i_max, b))
        elif abs(prof[b][0] - prof[a][0]) > max_gap:
            i_mid = (a + b) // 2
            keep.append(i_mid)
            stack.append((a, i_mid)); stack.append((i_mid, b))
    return [prof[i] for i in sorted(set(keep))]


def lathe_z(name, wf, ws, z_of, x_of, y_of, ppm, props, log=print):
    """Revolve about the vertical axis with an elliptic section from the front and side half-widths."""
    rows = sorted(set(wf) & set(ws))
    step = max(1, int(round(0.003 * ppm)))
    sel = rows[::step]
    if sel[-1] != rows[-1]:
        sel.append(rows[-1])
    xc = float(np.mean([x_of(0.5 * (wf[r][0] + wf[r][1] + 1)) for r in rows]))
    yc = float(np.mean([y_of(0.5 * (ws[r][0] + ws[r][1] + 1)) for r in rows]))
    prof = [(z_of(r + 0.5), max(0.0008, 0.5 * (wf[r][1] - wf[r][0] + 1) / ppm), max(0.0008, 0.5 * (ws[r][1] - ws[r][0] + 1) / ppm)) for r in sel]
    prof = simplify_profile(prof, tol=max(0.002, 0.005 * SIZE_M[0], 1.5 / ppm))
    r_max = max(max(p[1], p[2]) for p in prof)
    n = 12 if r_max < 0.015 else (16 if r_max < 0.04 else 24)
    bm = bmesh.new()
    rings = [[bm.verts.new(Vector((xc + rx * math.cos(2 * math.pi * k / n), yc + ry * math.sin(2 * math.pi * k / n), z))) for k in range(n)] for z, rx, ry in prof]
    for a, b in zip(rings, rings[1:]):
        for k in range(n):
            try:
                bm.faces.new([a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]])
            except ValueError:
                pass
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        try:
            bm.faces.new(ring[::-1] if flip else ring)
        except ValueError:
            pass
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob = emit(name, bm, dict(props, s3d_kind='lathe_z'))
    log('[prop] %s: lathe about z, %d rings x %d, rx %.0f ry %.0f mm at the widest, z %.3f..%.3f' % (name, len(prof), n, 1000 * max(p[1] for p in prof), 1000 * max(p[2] for p in prof), prof[0][0], prof[-1][0]))
    return ob


def lathe_y(name, ys_f, xs_f, sy, sx, z_of, x_of, y_of, ppm, props, log=print):
    """Revolve about the depth axis: a disc or knob facing the viewer, r(y) from the side view."""
    xc = float(x_of(0.5 * (xs_f.min() + xs_f.max() + 1)))
    zc = float(z_of(0.5 * (ys_f.min() + ys_f.max() + 1)))
    cols = sorted(np.unique(sx).tolist())
    step = max(1, int(round(0.003 * ppm)))
    sel = cols[::step]
    if sel[-1] != cols[-1]:
        sel.append(cols[-1])
    prof = []
    for c in sel:
        rs = sy[sx == c]
        prof.append((y_of(c + 0.5), max(0.0008, 0.5 * (rs.max() - rs.min() + 1) / ppm)))
    prof = simplify_profile(prof, tol=max(0.002, 0.005 * SIZE_M[0], 1.5 / ppm))
    r_max = max(p[1] for p in prof)
    n = 12 if r_max < 0.015 else (16 if r_max < 0.04 else 24)
    bm = bmesh.new()
    rings = [[bm.verts.new(Vector((xc + r * math.cos(2 * math.pi * k / n), y, zc + r * math.sin(2 * math.pi * k / n)))) for k in range(n)] for y, r in prof]
    for a, b in zip(rings, rings[1:]):
        for k in range(n):
            try:
                bm.faces.new([a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]])
            except ValueError:
                pass
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        try:
            bm.faces.new(ring[::-1] if flip else ring)
        except ValueError:
            pass
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob = emit(name, bm, dict(props, s3d_kind='lathe_y'))
    log('[prop] %s: lathe about y (a disc / knob facing the viewer), %d rings x %d, r %.0f mm, y %.3f..%.3f' % (name, len(prof), n, 1000 * r_max, prof[0][0], prof[-1][0]))
    return ob


def lathe(name, wf, ws, z_of, x_of, y_of, ppm, props, log=print):
    rows = sorted(set(wf) & set(ws))
    step = max(1, int(round(0.003 * ppm)))
    sel = rows[::step]
    if sel[-1] != rows[-1]:
        sel.append(rows[-1])
    xc = float(np.mean([x_of(0.5 * (wf[r][0] + wf[r][1] + 1)) for r in rows]))
    yc = float(np.mean([y_of(0.5 * (ws[r][0] + ws[r][1] + 1)) for r in rows]))
    prof = []
    for r in sel:
        rad = 0.5 * max(wf[r][1] - wf[r][0] + 1, ws[r][1] - ws[r][0] + 1) / ppm
        prof.append((z_of(r + 0.5), max(0.0008, rad)))
    r_max = max(p[1] for p in prof)
    n = 12 if r_max < 0.015 else (16 if r_max < 0.04 else 24)
    bm = bmesh.new()
    rings = []
    for z, rad in prof:
        ring = [bm.verts.new(Vector((xc + rad * math.cos(2 * math.pi * k / n), yc + rad * math.sin(2 * math.pi * k / n), z))) for k in range(n)]
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(n):
            try:
                bm.faces.new([a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]])
            except ValueError:
                pass
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        try:
            bm.faces.new(ring[::-1] if flip else ring)
        except ValueError:
            pass
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob = emit(name, bm, dict(props, s3d_kind='lathe'))
    log('[prop] %s: lathe, %d rings x %d, r %.1f..%.1f mm, z %.3f..%.3f' % (name, len(prof), n, 1000 * min(p[1] for p in prof), 1000 * r_max, prof[0][0], prof[-1][0]))
    return ob


def paint_atlas(made, painted_cls, P, lab, pan, col_of, atlas_png, log=print):
    """Smart-project each object into its own atlas cell and paint faces from the view they face."""
    sys.path.insert(0, os.path.join(os.path.dirname(D), 'finish'))
    import uv as UV
    RES = UV.RES
    ppm = P['ppm']
    front = pan['front']; side = pan.get('side'); top = pan.get('top')
    names = {c['id']: c['name'] for c in P['classes']}
    n = max(1, len(made))
    k = int(math.ceil(math.sqrt(n)))
    cell = 1.0 / k
    for i, ob in enumerate(made):
        bpy.context.view_layer.objects.active = ob
        for o in bpy.data.objects:
            o.select_set(o is ob)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        uvl = ob.data.uv_layers.active
        u0, v0 = (i % k) * cell, (i // k) * cell
        for d in uvl.data:
            d.uv = (u0 + d.uv[0] * cell * 0.94 + 0.03 * cell, v0 + d.uv[1] * cell * 0.94 + 0.03 * cell)
    atlas = np.zeros((RES, RES, 3), np.uint8)

    def label_at(view, pt):
        if view == 'front':
            r = front['r1'] + 1 - pt.z * ppm; c = front['c0'] + front['cx'] - front['c0'] + pt.x * ppm
        elif view == 'side' and side:
            r = front['r1'] + 1 - pt.z * ppm; c = side['cx'] + float(P.get('side_sign', 1)) * pt.y * ppm
        elif view == 'top' and top:
            r = top['cy'] + (-pt.y if P.get('top_front_down', True) else pt.y) * ppm; c = top['cx'] + pt.x * ppm
        else:
            return None
        r, c = int(round(r)), int(round(c))
        if 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1]:
            v = int(lab[r, c])
            return names.get(v)
        return None
    for ob in made:
        cname = ob['s3d_class']
        base = np.array(col_of.get(cname, (200, 200, 200)), np.uint8)
        md = ob.data
        uv = md.uv_layers.active.data
        mw = ob.matrix_world
        for pg in md.polygons:
            nrm = (mw.to_3x3() @ pg.normal).normalized()
            if abs(nrm.y) >= abs(nrm.x) and abs(nrm.y) >= abs(nrm.z):
                view = 'front' if nrm.y < 0 else None
            elif abs(nrm.x) >= abs(nrm.z):
                view = 'side'
            else:
                view = 'top' if nrm.z > 0 else None
            pts = [(uv[li].uv[0] * RES, (1.0 - uv[li].uv[1]) * RES) for li in pg.loop_indices]
            Pw = [mw @ md.vertices[md.loops[li].vertex_index].co for li in pg.loop_indices]

            def fn(pt, _view=view):
                if _view is None or not painted_cls:
                    return None
                kk = label_at(_view, pt)
                if kk is None or kk == cname or kk not in painted_cls:
                    return None
                return np.array(col_of.get(kk, (200, 200, 200)), np.uint8)
            for i in range(1, len(pts) - 1):
                UV.tri_fill_fn(atlas, pts[0], pts[i], pts[i + 1], Pw[0], Pw[i], Pw[i + 1], fn, base)
    alpha = atlas.any(2)
    atlas, alpha = UV.dilate(atlas, alpha, UV.GUTTER)
    img = bpy.data.images.new('prop_atlas', RES, RES, alpha=False)
    px = np.ones((RES, RES, 4), np.float32)
    px[:, :, :3] = atlas[::-1].astype(np.float32) / 255.0
    img.pixels = px.ravel()
    img.filepath_raw = atlas_png
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    log('[prop] atlas %s: %d objects in a %dx%d grid' % (os.path.basename(atlas_png), len(made), k, k))


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    prop_path, out_blend = argv[:2]
    P = json.load(open(prop_path))
    lab = np.load(P['labels'])
    ppm = P['ppm']
    SIZE_M[0] = float(P['size_m'])
    pan = {p['name']: p for p in P['panels']}
    front = pan['front']
    z_of = lambda r: (front['r1'] + 1 - r) / ppm  # the front figure's bottom is z 0
    x_of = lambda c: (c - front['cx']) / ppm
    side = pan.get('side')
    sgn = float(P.get('side_sign', 1))  # +1: the side view's front is on its left (-y)
    y_of = (lambda c: sgn * (c - side['cx']) / ppm) if side else (lambda c: 0.0)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    min_px = int(round(max(MIN_MM / 1000.0, 0.01 * P['size_m']) * ppm))
    order = sorted(P['classes'], key=lambda c: -int((lab[:, front['c0']:front['c1']] == c['id']).sum()))
    # a class drawn only in front lies on the class it borders most, small ones first
    front_lab = lab[:, front['c0']:front['c1']]
    side_lab = lab[:, side['c0']:side['c1']] if side else None
    front_lab_full = np.zeros(lab.shape, lab.dtype); front_lab_full[:, front['c0']:front['c1']] = front_lab
    side_lab_full = np.zeros(lab.shape, lab.dtype)
    if side:
        side_lab_full[:, side['c0']:side['c1']] = side_lab
    build_mask = {c['id']: (front_lab == c['id']) for c in P['classes']}
    front_only = [c for c in P['classes'] if side is not None and not (side_lab == c['id']).any() and (front_lab == c['id']).any()]
    names_ = {q['id']: q['name'] for q in P['classes']}
    flat = set()
    for c in sorted(P['classes'], key=lambda q: int((front_lab == q['id']).sum())):
        m = front_lab == c['id']
        if not m.any():
            continue
        ring = np.zeros_like(m)
        ring[1:, :] |= m[:-1, :]; ring[:-1, :] |= m[1:, :]; ring[:, 1:] |= m[:, :-1]; ring[:, :-1] |= m[:, 1:]
        ring &= ~m
        touch = front_lab[ring]
        on_face = c in front_only or (len(touch) and (touch == 0).mean() <= 0.25)
        touch = touch[(touch > 0) & (touch != c['id'])]
        if not on_face or not len(touch):
            continue
        cnt = np.bincount(touch)
        hosts = [h for h in range(len(cnt)) if cnt[h] >= 0.2 * len(touch)]
        rows = np.arange(m.shape[0])[:, None]
        for h in hosts:
            hr = np.nonzero((front_lab == h).any(1))[0]
            if len(hr):
                build_mask[h] = build_mask[h] | (m & (rows >= hr.min()) & (rows <= hr.max()))
        if side is not None and c not in front_only:
            host_side = np.isin(side_lab, hosts)
            own_side = side_lab == c['id']
            tol = max(3, int(round(0.01 * ppm)))
            out_px = 0
            for r in np.nonzero(own_side.any(1))[0]:
                hc = np.nonzero(host_side[max(0, r - tol):r + tol + 1].any(0))[0]
                oc = np.nonzero(own_side[r])[0]
                out_px += len(oc) if not len(hc) else int(((oc < hc.min() - tol) | (oc > hc.max() + tol)).sum())
            if out_px <= 0.02 * max(1, own_side.sum()):
                flat.add(c['id'])
        print('[prop] %s lies on %s%s: its outline joins the host%s' % (c['name'], ', '.join(names_[h] for h in hosts),
              ' (front only)' if c in front_only else '', ', painted' if c['id'] in flat else ''))
    made = []
    painted_cls = set()
    for c in order:
        cid, cname = c['id'], c['name']
        if cid in flat:
            painted_cls.add(cname)
            continue
        props = dict(s3d_part='prop', s3d_class=cname)
        mf = np.zeros(lab.shape, bool); mf[:, front['c0']:front['c1']] = build_mask[cid]
        ms = np.zeros(lab.shape, bool)
        if side:
            ms[:, side['c0']:side['c1']] = lab[:, side['c0']:side['c1']] == cid
        comps_f = components(mf, min_px)
        comps_s = components(ms, min_px) if side else []
        if not comps_f and not comps_s:
            print('[prop] %s: not drawn' % cname)
            continue
        used_s = set()
        k = 0
        comps_f.sort(key=lambda q: -len(q[0]))
        for (ys, xs) in comps_f:
            name = '%s_%d' % (cname.capitalize(), k)
            r_lo, r_hi = int(ys.min()), int(ys.max())
            match = []
            for j, (sy, sx) in enumerate(comps_s):
                if j in used_s:
                    continue
                lo_, hi_ = max(r_lo, int(sy.min())), min(r_hi, int(sy.max()))
                if hi_ - lo_ >= 0.5 * min(r_hi - r_lo, int(sy.max()) - int(sy.min())):
                    match.append(j)
            hull_f = outline(ys, xs, ppm)
            if len(hull_f) < 3:
                continue
            ob = None
            best = match[0] if match else None
            if best is not None:
                sy = np.concatenate([comps_s[j][0] for j in match]); sx = np.concatenate([comps_s[j][1] for j in match])
                used_s.update(match)
                if len(match) > 1:
                    rf = np.zeros(lab.shape, bool)
                    for r in np.unique(sy):
                        cs = sx[sy == r]
                        rf[r, cs.min():cs.max() + 1] = True
                    sy, sx = np.nonzero(rf)
                wf, ws = row_widths(ys, xs), row_widths(sy, sx)
                # revolve about depth for a round front, about z for straight axes with a section within 3:1, else hull
                w_f = xs.max() - xs.min() + 1; w_s = sx.max() - sx.min() + 1
                aspect = max(w_f, w_s) / float(max(1, min(w_f, w_s)))
                wand_f, wand_s = axis_wander(wf), axis_wander(ws)
                is_round = round_named(cname)
                print('[prop] %s: form test - %s by name, front %s, axis wander %.2f / %.2f, section aspect %.1f' % (name, 'round' if is_round else 'boxy', 'round' if circular(ys, xs) else 'not round', wand_f, wand_s, aspect))
                if is_round and circular(ys, xs) and wand_s <= 0.25:
                    ob = lathe_y(name, ys, xs, sy, sx, z_of, x_of, y_of, ppm, props, log=print)
                elif is_round and wand_f <= 0.25 and wand_s <= 0.25 and aspect <= 3.0:
                    ob = lathe_z(name, wf, ws, z_of, x_of, y_of, ppm, props, log=print)
                elif thin_structure(ys, xs, ppm, P['size_m']):
                    # thin structure: a prism of the front outline as deep as its stroke
                    f_poly = [Vector((x_of(cx), 0.0, z_of(ry))) for (cx, ry) in hull_f]
                    hf = [[Vector((x_of(cx), 0.0, z_of(ry))) for (cx, ry) in h] for h in holes_of(ys, xs, ppm)]
                    widths = [ws[r][1] - ws[r][0] + 1 for r in ws]
                    t = max(0.004, min(float(np.median(widths)), 2.0 * stroke_px(ys, xs)) / ppm)
                    yc = float(np.median([y_of(0.5 * (ws[r][0] + ws[r][1] + 1)) for r in ws]))
                    bm = prism_holes(f_poly, hf, 1, yc - 0.5 * t, yc + 0.5 * t) if hf else prism(f_poly, 1, yc - 0.5 * t, yc + 0.5 * t)
                    ob = emit(name, bm, dict(props, s3d_kind='frame'))
                    print('[prop] %s: thin structure - the front outline (%d points, %d holes) %.0f mm deep at y %.3f' % (name, len(hull_f), len(hf), 1000 * t, yc))
                else:
                    cl = int(round(0.02 * P['size_m'] * ppm))
                    ys, xs, n_af = absorb_occluders(front_lab_full, ys, xs, cid)
                    sy, sx, n_as = absorb_occluders(side_lab_full, sy, sx, cid)
                    if n_af or n_as:
                        print('[prop] %s: %d + %d part(s) drawn on it join its silhouette' % (name, n_af, n_as))
                    ys, xs = filled(ys, xs, cl)
                    sy, sx = filled(sy, sx, cl)
                    hull_f = outline(ys, xs, ppm)
                    hull_s = outline(sy, sx, ppm)
                    f_poly = [Vector((x_of(cx), 0.0, z_of(ry))) for (cx, ry) in hull_f]
                    s_poly = [Vector((0.0, y_of(cx), z_of(ry))) for (cx, ry) in hull_s]
                    ys_ = [q.y for q in s_poly]; xs_ = [q.x for q in f_poly]
                    hf = [[Vector((x_of(cx), 0.0, z_of(ry))) for (cx, ry) in h] for h in holes_of(ys, xs, ppm)]
                    hs = [[Vector((0.0, y_of(cx), z_of(ry))) for (cx, ry) in h] for h in holes_of(sy, sx, ppm)]
                    bm_a = prism_holes(f_poly, hf, 1, min(ys_) - 0.002, max(ys_) + 0.002) if hf else None
                    bm_b = prism_holes(s_poly, hs, 0, min(xs_) - 0.002, max(xs_) + 0.002) if hs else None
                    if bm_a is None:
                        bm_a = prism(f_poly, 1, min(ys_) - 0.002, max(ys_) + 0.002)
                    if bm_b is None:
                        bm_b = prism(s_poly, 0, min(xs_) - 0.002, max(xs_) + 0.002)
                    ob, ok = visual_hull(name, bm_a, bm_b, dict(props, s3d_kind='hull'), log=print)
                    print('[prop] %s: visual hull front x side (%d x %d points%s), y %.3f..%.3f%s' % (name, len(hull_f), len(hull_s), (', %d + %d holes' % (len(hf), len(hs))) if (hf or hs) else '', min(ys_), max(ys_), '' if ok else ' (front prism only)'))
            else:
                painted_cls.add(cname)
                ob = None
            if ob is not None:
                made.append(ob)
                k += 1
        for j, (sy, sx) in enumerate(comps_s):
            if j not in used_s and len(sy) >= (0.01 * ppm) ** 2:
                print('[prop] %s: a side-only piece (%d px) has no front counterpart - not built' % (cname, len(sy)))
    for ob in made:
        bm = bmesh.new(); bm.from_mesh(ob.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        vol = 0.0
        for f in bm.faces:
            vs = [v.co for v in f.verts]
            for i in range(1, len(vs) - 1):
                vol += vs[0].dot(vs[i].cross(vs[i + 1])) / 6.0
        if vol < 0:
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(ob.data); bm.free(); ob.data.update()
    for ob in made:
        PL.polish(ob, log=print)
    ac = P.get('art_colours') or {}
    col_of = {}
    for c in P['classes']:
        col_of[c['name']] = ac.get(c['name']) or c['colour']
    atlas_png = os.path.join(os.path.dirname(out_blend), 'prop_atlas.png')
    paint_atlas(made, painted_cls, P, lab, pan, col_of, atlas_png, log=print)
    img = bpy.data.images.load(atlas_png)
    mat = bpy.data.materials.new('M_prop')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.interpolation = 'Closest'
    if bsdf is not None:
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = 0.85
    for ob in made:
        ob.data.materials.clear()  # a Boolean can leave an empty slot first
        ob.data.materials.append(mat)
        for pg in ob.data.polygons:
            pg.material_index = 0
    if painted_cls:
        print('[prop] painted (texture, no geometry): %s' % ', '.join(sorted(painted_cls)))
    json.dump(sorted(painted_cls), open(os.path.join(os.path.dirname(out_blend), 'prop_painted.json'), 'w'))
    tris = sum(sum(len(p.vertices) - 2 for p in ob.data.polygons) for ob in made)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[prop] %d objects, %d tris -> %s' % (len(made), tris, out_blend))


if __name__ == '__main__':
    main()
