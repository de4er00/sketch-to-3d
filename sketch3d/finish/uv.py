"""UVs, the flat-colour body atlas, the face cell and unlit materials.

    blender -b --factory-startup --python sketch3d/finish/uv.py -- <char.json> <body.json> <char.blend> <outdir>
"""
import bpy, bmesh, os, sys, json, math
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))
from carve import Sheet   # noqa: E402

RES = 1024
GUTTER = 4


def class_colour(C, cname):
    # the part map's colours are labels: paint with the art's colours
    ac = C.get('art_colours') or {}
    if cname in ac:
        return tuple(int(v) for v in ac[cname])
    hair = (C.get('roles') or {}).get('hair') or []
    if cname == 'hair' and hair and hair[0] in ac:
        return tuple(int(v) for v in ac[hair[0]])
    pal = C['palette']['colours']
    for p in C['parts']:
        if p['name'] == cname:
            return tuple(int(v) for v in pal[p['colours'][0]])
    return (200, 200, 200)


def band_colour(C, cname, z, base):
    """Colour of the nearest height band at z, else base."""
    bands = (C.get('art_bands') or {}).get(cname)
    if not bands:
        return base
    best = min(bands, key=lambda b: 0.0 if b[1] <= z <= b[0] else min(abs(z - b[0]), abs(z - b[1])))
    return np.array(best[2], np.asarray(base).dtype)


def tri_fill(img, p0, p1, p2, col):
    xs = [p0[0], p1[0], p2[0]]
    ys = [p0[1], p1[1], p2[1]]
    x0, x1 = max(0, int(math.floor(min(xs)))), min(RES - 1, int(math.ceil(max(xs))))
    y0, y1 = max(0, int(math.floor(min(ys)))), min(RES - 1, int(math.ceil(max(ys))))
    if x1 < x0 or y1 < y0:
        return
    X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    if abs(d) < 1e-9:
        return
    a = ((p1[0] - X) * (p2[1] - Y) - (p2[0] - X) * (p1[1] - Y)) / d
    b = ((p2[0] - X) * (p0[1] - Y) - (p0[0] - X) * (p2[1] - Y)) / d
    c = 1 - a - b
    eps = -0.002
    m = (a >= eps) & (b >= eps) & (c >= eps)
    sub = img[y0:y1 + 1, x0:x1 + 1]
    sub[m] = col


SS = 2  # 2 x 2 samples per pixel


def tri_fill_fn(img, p0, p1, p2, P0, P1, P2, fn, base, ss=None):
    """Fill a UV triangle with fn(world point), or `base` where it returns None, ss x ss samples per pixel."""
    ss = SS if ss is None else ss
    xs = [p0[0], p1[0], p2[0]]
    ys = [p0[1], p1[1], p2[1]]
    x0, x1 = max(0, int(math.floor(min(xs)))), min(RES - 1, int(math.ceil(max(xs))))
    y0, y1 = max(0, int(math.floor(min(ys)))), min(RES - 1, int(math.ceil(max(ys))))
    if x1 < x0 or y1 < y0:
        return
    X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    if abs(d) < 1e-9:
        return
    a = ((p1[0] - X) * (p2[1] - Y) - (p2[0] - X) * (p1[1] - Y)) / d
    b = ((p2[0] - X) * (p0[1] - Y) - (p0[0] - X) * (p2[1] - Y)) / d
    c = 1 - a - b
    eps = -0.002
    m = (a >= eps) & (b >= eps) & (c >= eps)
    sub = img[y0:y1 + 1, x0:x1 + 1]
    ys_, xs_ = np.nonzero(m)
    if ss <= 1:
        for yy, xx in zip(ys_, xs_):
            aa, bb, cc = a[yy, xx], b[yy, xx], c[yy, xx]
            pt = P0 * aa + P1 * bb + P2 * cc
            col = fn(pt)
            sub[yy, xx] = base if col is None else col
        return
    dax = ((p1[1] - p2[1])) / d; dbx = ((p2[1] - p0[1])) / d
    day = ((p2[0] - p1[0])) / d; dby = ((p0[0] - p2[0])) / d
    offs = [((i + 0.5) / ss - 0.5, (j + 0.5) / ss - 0.5) for i in range(ss) for j in range(ss)]
    base_f = np.array(base, np.float32)
    for yy, xx in zip(ys_, xs_):
        acc = np.zeros(3, np.float32)
        for (ox, oy) in offs:
            aa = a[yy, xx] + dax * ox + day * oy
            bb = b[yy, xx] + dbx * ox + dby * oy
            cc = 1.0 - aa - bb
            pt = P0 * aa + P1 * bb + P2 * cc
            col = fn(pt)
            acc += base_f if col is None else np.asarray(col, np.float32)
        sub[yy, xx] = np.clip(acc / len(offs), 0, 255).astype(np.uint8)


def dilate(img, alpha, n):
    for _ in range(n):
        src = img.copy()
        a = alpha.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(src, (dy, dx), (0, 1))
            sa = np.roll(a, (dy, dx), (0, 1))
            take = (~alpha) & sa
            img[take] = sh[take]
            alpha = alpha | take
    return img, alpha


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, outdir = [os.path.abspath(a) for a in argv[:4]]
    C = json.load(open(char_path))
    M = json.load(open(map_path))
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    scene = bpy.context.scene
    body = bpy.data.objects['Body']
    meshes = [o for o in scene.objects if o.type == 'MESH']
    head_ids = set(i for r in M['head'] for i in r)
    hairline = C['skull']['hairline_z']
    fb = C['face']['box_m']
    face_faces = set()
    me = body.data
    for p in me.polygons:
        if all(v in head_ids for v in p.vertices) and p.normal.y < -0.45:
            zc = sum(me.vertices[v].co.z for v in p.vertices) / len(p.vertices)
            if zc <= hairline + 0.012 and zc >= fb[2] - 0.02:
                face_faces.add(p.index)
    # the face goes to cell (0, 0) of a 3x3 grid, same scale in u and v
    cell = 1.0 / 3.0
    marg = 0.10
    w_box, h_box = fb[1] - fb[0], fb[3] - fb[2]
    scale = min(cell * (1 - 2 * marg) / w_box, cell * (1 - 2 * marg) / h_box)
    cx, cz = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3])
    u_c, v_c = cell * 0.5, cell * 0.5

    def face_uv(x, z):
        return (u_c + (x - cx) * scale, v_c + (z - cz) * scale)
    def unlit(name, img_path, colour=None):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        tex = nt.nodes.new('ShaderNodeTexImage')
        img = bpy.data.images.load(img_path) if os.path.exists(img_path) else bpy.data.images.new(os.path.basename(img_path), RES, RES)
        img.filepath = img_path
        tex.image = img
        tex.interpolation = 'Closest'
        tex.extension = 'REPEAT'
        nt.links.new(tex.outputs['Color'], em.inputs['Color'])
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
        if colour:
            mat.diffuse_color = (colour[0] / 255.0, colour[1] / 255.0, colour[2] / 255.0, 1.0)
        return mat
    body_png = os.path.join(outdir, 'body_atlas.png')
    face_png = os.path.join(outdir, 'face_atlas.png')
    atlas = np.zeros((RES, RES, 3), np.uint8)
    alpha = np.zeros((RES, RES), bool)
    if not os.path.exists(face_png):
        import struct, zlib
        sk = class_colour(C, C['roles']['skin'][0])
        raw = b''.join(b'\x00' + bytes(sk) * RES for _ in range(RES))
        def chunk(t, d):
            return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', RES, RES, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
        open(face_png, 'wb').write(png)
    mats = {}
    report = dict(objects={}, face_faces=len(face_faces))
    for o in meshes:
        o.data.materials.clear()
        if not o.data.uv_layers:
            o.data.uv_layers.new(name='UVMap')
    # pack each object into [0,1] alone, then place it in its own treemap tile by area
    def area_of(o):
        return sum(p.area for p in o.data.polygons)
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.006, correct_aspect=True, scale_to_bounds=False)
        try:
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.pack_islands(rotate=True, scale=True, margin_method='ADD', margin=0.006)
        except TypeError:
            bpy.ops.uv.pack_islands(rotate=True, margin=0.006)
        bpy.ops.object.mode_set(mode='OBJECT')
        o.select_set(False)
    items = sorted([(area_of(o), o) for o in meshes], key=lambda t: -t[0])

    def treemap(items, rect):
        if len(items) == 1:
            return {items[0][1].name: rect}
        tot = sum(a for a, _ in items)
        acc, k = 0.0, 0
        while k < len(items) - 1 and acc + items[k][0] < 0.5 * tot:
            acc += items[k][0]
            k += 1
        k = max(1, k)
        left, right = items[:k], items[k:]
        f = sum(a for a, _ in left) / tot
        x0, y0, w, h = rect
        if w >= h:
            r1, r2 = (x0, y0, w * f, h), (x0 + w * f, y0, w * (1 - f), h)
        else:
            r1, r2 = (x0, y0, w, h * f), (x0, y0 + h * f, w, h * (1 - f))
        out = treemap(left, r1)
        out.update(treemap(right, r2))
        return out
    tiles = treemap(items, (0.0, 0.0, 1.0, 1.0))
    pad = GUTTER / RES
    for o in meshes:
        x0, y0, w, h = tiles[o.name]
        uv = o.data.uv_layers.active.data
        for l in uv:
            u, v = l.uv
            l.uv = (x0 + pad + u * (w - 2 * pad), y0 + pad + v * (h - 2 * pad))
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    uvl = bm.loops.layers.uv.active
    for fi in face_faces:
        f = bm.faces[fi]
        for l in f.loops:
            l[uvl].uv = face_uv(l.vert.co.x, l.vert.co.z)
    bm.to_mesh(me)
    bm.free()
    report['tiles'] = {k: [round(v_, 4) for v_ in v] for k, v in tiles.items()}
    # flats per class, rim faces as ink, a little contact shading (12 % at most)
    from mathutils.bvhtree import BVHTree
    bpy.context.evaluated_depsgraph_get()
    scene_bm = bmesh.new()
    for o in meshes:
        tmp = bmesh.new()
        tmp.from_mesh(o.data)
        tmp.transform(o.matrix_world)
        for f in tmp.faces:
            scene_bm.faces.new([scene_bm.verts.new(v.co) for v in f.verts])
        tmp.free()
    scene_bvh = BVHTree.FromBMesh(scene_bm)
    scene_bm.free()
    import random
    rnd = random.Random(7)
    dirs = []
    for _ in range(24):
        while True:
            v = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)))
            if 1e-3 < v.length <= 1.0:
                dirs.append(v.normalized())
                break

    def occlusion(co, n):
        hits = 0
        for d in dirs:
            dd = d if d.dot(n) > 0 else -d
            loc, _n, _i, _dist = scene_bvh.ray_cast(co + n * 0.002, dd, 0.12)
            if loc is not None:
                hits += 1
        return hits / len(dirs)
    ink = np.array((28, 26, 30), np.uint8)
    # shoe faces take the class the sheet draws in the view they face
    sheet = Sheet(char_path)
    shoe_cls = list(C['roles'].get('shoe', [])) + list(C['roles'].get('sole', []))
    group_of = {c: 'shoe' for c in shoe_cls}
    col_of = {c: np.array(class_colour(C, c), np.uint8) for c in list(C['parts_by_name']) if c in group_of} if 'parts_by_name' in C else {c: np.array(class_colour(C, c), np.uint8) for c in shoe_cls}
    lab_ids = {v: k for k, v in sheet.ids.items()}
    painted = [k for k, e in (C.get('garments') or {}).items() if e.get('painted')]
    wrapping = set(k for k in painted if not C['garments'][k].get('decal')) | set((C.get('hair') or {}).get('painted', []))
    painted += [q for q in (body.get('s3d_painted') or '').split(',') if q and q not in painted]
    painted += [q for q in (C.get('hair') or {}).get('painted', []) if q not in painted]
    for k in painted:
        group_of[k] = 'body'
        col_of[k] = np.array(class_colour(C, k), np.uint8)

    det_path = os.path.join(os.path.dirname(char_path), 'details.npy')
    marks = {q['id']: q for q in (C.get('art_details') or [])}
    det_img = np.load(det_path) if marks and os.path.exists(det_path) else None
    mark_cls = set(q['cls'] for q in marks.values())

    def mark_at(view, pt, cname):
        if det_img is None or view is None or cname not in mark_cls:
            return None
        q = pt
        if view == 'side':
            q = Vector((abs(pt.x) if sheet.side_sign > 0 else -abs(pt.x), pt.y, pt.z))
        r, c = sheet.project(view, q)
        if not (0 <= r < det_img.shape[0] and 0 <= c < det_img.shape[1]) or not det_img[r, c]:
            return None
        m = marks.get(int(det_img[r, c]))
        if m is None or m['cls'] != cname or m['view'] != view:
            return None
        return np.array(m['rgb'], np.float32)

    def facing(n):
        """The view a face looks squarely toward."""
        if n.y < -0.5:
            return 'front'
        if n.y > 0.5:
            return 'back'
        return 'side' if abs(n.x) > 0.5 else None

    def drawn_class_at(view, pt):
        r, c = sheet.project(view, pt)
        if 0 <= r < sheet.lab.shape[0] and 0 <= c < sheet.lab.shape[1]:
            return lab_ids.get(int(sheet.lab[r, c]))
        return None
    buzz_rgb = np.array(C['buzz']['rgb'], np.uint8) if C.get('buzz') and sheet.buzz is not None else None
    z_chin = C['landmarks']['chin']['z'] - 0.02

    def buzz_at(view, pt):
        if sheet.buzz is None:
            return False
        if view == 'side':
            pt = Vector((abs(pt.x) if sheet.side_sign > 0 else -abs(pt.x), pt.y, pt.z))
        r, c = sheet.project(view, pt)
        return 0 <= r < sheet.buzz.shape[0] and 0 <= c < sheet.buzz.shape[1] and bool(sheet.buzz[r, c])
    for o in meshes:
        cname = o.get('s3d_class', 'skin')
        col = np.array(class_colour(C, cname), np.float32)
        md = o.data
        uv = md.uv_layers.active.data
        rim_layer = md.attributes.get('s3d_rim')
        rim = [int(v.value) for v in rim_layer.data] if rim_layer is not None else None
        n_tri = 0
        proj = bool(painted)
        mw = o.matrix_world
        head_buzz = (o is body) and buzz_rgb is not None
        for p in md.polygons:
            if o is body and p.index in face_faces:
                continue
            pts = [(uv[li].uv[0] * RES, (1.0 - uv[li].uv[1]) * RES) for li in p.loop_indices]
            if rim is not None and p.index < len(rim) and rim[p.index]:
                c = ink
            else:
                ao = occlusion(mw @ p.center, (mw.to_3x3() @ p.normal).normalized())
                c = np.clip(band_colour(C, cname, (mw @ p.center).z, col) * (1.0 - 0.12 * ao), 0, 255).astype(np.uint8)
            if head_buzz and p.center.z > z_chin and p.index not in face_faces:
                n = (mw.to_3x3() @ p.normal).normalized()
                view = 'side' if abs(n.x) > abs(n.y) and abs(n.x) > 0.4 else ('back' if n.y > 0.4 else 'front')
                shade = 1.0 - 0.12 * ao
                bz = np.clip(buzz_rgb.astype(np.float32) * shade, 0, 255).astype(np.uint8)

                def fnb(pt, _view=view, _bz=bz):
                    return _bz if buzz_at(_view, pt) else None
                P = [mw @ md.vertices[md.loops[li].vertex_index].co for li in p.loop_indices]
                for i in range(1, len(pts) - 1):
                    tri_fill_fn(atlas, pts[0], pts[i], pts[i + 1], P[0], P[i], P[i + 1], fnb, c)
                    n_tri += 1
                continue
            if proj and not (rim is not None and p.index < len(rim) and rim[p.index]):
                n = (mw.to_3x3() @ p.normal).normalized()
                view = 'side' if abs(n.x) > abs(n.y) and abs(n.x) > 0.4 else ('back' if n.y > 0.4 else 'front')
                if n.z < -0.7:
                    view = None  # the underside: no view draws it
                facing_ = facing(n)
                shade = 1.0 - 0.12 * ao

                def fn(pt, _view=view, _shade=shade):
                    # a wrapping class covers the body from any view, elsewhere only the facing view counts
                    views_ = ([_view] if _view else [])
                    if o is body:
                        views_ += [v_ for v_ in ('front', 'side', 'back') if v_ != _view and v_ in sheet.panels]
                    for v_ in views_:
                        q = pt
                        if v_ == 'side':
                            q = Vector((abs(pt.x) if sheet.side_sign > 0 else -abs(pt.x), pt.y, pt.z))
                        k = drawn_class_at(v_, q)
                        if k is None or k not in group_of or k == cname:
                            continue
                        if k not in painted:
                            continue
                        if v_ != _view and k not in wrapping:
                            continue
                        mk = mark_at(v_, pt, k) if v_ == _view and _view == facing_ else None
                        if mk is not None:
                            return np.clip(mk * _shade, 0, 255).astype(np.uint8)
                        return np.clip(band_colour(C, k, pt.z, col_of[k]).astype(np.float32) * _shade, 0, 255).astype(np.uint8)
                    return None
                P = [mw @ md.vertices[md.loops[li].vertex_index].co for li in p.loop_indices]
                for i in range(1, len(pts) - 1):
                    tri_fill_fn(atlas, pts[0], pts[i], pts[i + 1], P[0], P[i], P[i + 1], fn, c)
                    n_tri += 1
                continue
            view_m = facing((mw.to_3x3() @ p.normal).normalized()) if cname in mark_cls and not (rim is not None and p.index < len(rim) and rim[p.index]) else None
            if view_m is not None:
                shade = 1.0 - 0.12 * occlusion(mw @ p.center, (mw.to_3x3() @ p.normal).normalized())

                def fn_mark(pt, _view=view_m, _shade=shade):
                    mk = mark_at(_view, pt, cname)
                    return None if mk is None else np.clip(mk * _shade, 0, 255).astype(np.uint8)
                P = [mw @ md.vertices[md.loops[li].vertex_index].co for li in p.loop_indices]
                for i in range(1, len(pts) - 1):
                    tri_fill_fn(atlas, pts[0], pts[i], pts[i + 1], P[0], P[i], P[i + 1], fn_mark, c)
                    n_tri += 1
                continue
            for i in range(1, len(pts) - 1):
                tri_fill(atlas, pts[0], pts[i], pts[i + 1], c)
                n_tri += 1
        report['objects'][o.name] = dict(cls=cname, colour=[int(v) for v in col], tris=n_tri)
    alpha = atlas.any(2)
    atlas, alpha = dilate(atlas, alpha, GUTTER)
    fill = float(alpha.mean())
    import struct, zlib

    def write_png(path, arr):
        H, W, _ = arr.shape
        raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(H))

        def chunk(t, d):
            return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')
        open(path, 'wb').write(png)
    write_png(body_png, atlas)
    # create materials after painting: an image loaded earlier keeps the placeholder pixels
    for o in meshes:
        cname = o.get('s3d_class', 'skin')
        col = class_colour(C, cname)
        if cname not in mats:
            mats[cname] = unlit('M_' + cname, body_png, col)
        o.data.materials.append(mats[cname])
        if o is body:
            mats['face'] = unlit('M_face', face_png, col)
            o.data.materials.append(mats['face'])
            for p in me.polygons:
                p.material_index = 1 if p.index in face_faces else 0
    scene.render.engine = 'BLENDER_EEVEE'
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    for o in list(scene.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o, do_unlink=True)
    json.dump(dict(cell=cell, scale_uv_per_m=scale, centre_m=[cx, cz], u_c=u_c, v_c=v_c, box_m=fb, res=RES, margin=marg),
              open(os.path.join(outdir, 'face_cell.json'), 'w'), indent=1)
    report['body_fill'] = fill
    json.dump(report, open(os.path.join(outdir, 'uv_report.json'), 'w'), indent=1)
    out_blend = os.path.join(outdir, 'char_tex.blend')
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[uv] %d objects, face faces %d, body atlas fill %.1f%% -> %s' % (len(meshes), len(face_faces), 100 * fill, out_blend))


if __name__ == '__main__':
    main()
