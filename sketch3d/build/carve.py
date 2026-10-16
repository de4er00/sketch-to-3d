"""Trim built shells to the drawn class masks in the sheet views, plus mesh clean-up passes."""
import bmesh, os, json, math
import bmesh.utils
import numpy as np
from mathutils import Vector

PULL_MAX = 0.025  # an outside vertex farther than this from the drawn edge loses its faces


def seam_verts(bm):
    sm = bm.verts.layers.int.get('s3d_seam')
    if sm is None:
        return set()
    return set(v for v in bm.verts if v[sm])


def label8(m):
    """Label 8-connected components with union-find on runs (Blender's Python has no scipy)."""
    H, W = m.shape
    lab = np.zeros((H, W), np.int32)
    parent = [0]
    runs_prev = []
    nxt = 1
    for r in range(H):
        row = m[r]
        runs = []
        idx = np.nonzero(row)[0]
        if len(idx):
            starts = [int(idx[0])]
            ends = []
            for a, b in zip(idx[:-1], idx[1:]):
                if b != a + 1:
                    ends.append(int(a) + 1); starts.append(int(b))
            ends.append(int(idx[-1]) + 1)
            for c0, c1 in zip(starts, ends):
                lab_ = 0
                for (p0, p1, pl) in runs_prev:
                    if p0 <= c1 and p1 >= c0:
                        root = pl
                        while parent[root] != root:
                            root = parent[root]
                        if lab_ == 0:
                            lab_ = root
                        elif root != lab_:
                            parent[root] = lab_
                if lab_ == 0:
                    lab_ = nxt; parent.append(nxt); nxt += 1
                runs.append((c0, c1, lab_))
                lab[r, c0:c1] = lab_
        runs_prev = runs
    final = {}
    np.zeros_like(lab)
    k = 0
    for v in range(1, nxt):
        root = v
        while parent[root] != root:
            root = parent[root]
        if root not in final:
            k += 1
            final[root] = k
    lut = np.zeros(nxt, np.int32)
    for v in range(1, nxt):
        root = v
        while parent[root] != root:
            root = parent[root]
        lut[v] = final[root]
    return lut[lab]


def dilate8(m):
    out = m.copy()
    out[1:, :] |= m[:-1, :]; out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    out[1:, 1:] |= m[:-1, :-1]; out[:-1, :-1] |= m[1:, 1:]
    out[1:, :-1] |= m[:-1, 1:]; out[:-1, 1:] |= m[1:, :-1]
    return out


class Sheet:
    def __init__(self, char_json):
        C = json.load(open(char_json))
        d = os.path.dirname(os.path.abspath(char_json))
        self.C = C
        cal = C['calibration']
        self.ppm = cal['ppm']
        self.floor = cal['floor_row']
        self.pan = {p['name']: p for p in cal['panels']}
        lj = json.load(open(os.path.join(d, 'labels.json')))
        self.ids = lj['ids']
        self.panels = {k: tuple(v) for k, v in lj['panels'].items()}
        self.lab = np.load(os.path.join(d, 'labels.npy'))
        ink_p = os.path.join(d, 'labels_ink.npy')
        self.ink = np.load(ink_p).astype(bool) if os.path.exists(ink_p) else np.zeros(self.lab.shape, bool)
        buzz_p = os.path.join(d, 'labels_buzz.npy')
        self.buzz = np.load(buzz_p).astype(bool) if os.path.exists(buzz_p) else None
        if self.buzz is not None:
            for _ in range(2):
                self.buzz = dilate8(self.buzz)
        self.dcol = C['datum']['side_col']
        self.side_sign = C['datum'].get('side_sign', 1.0)
        self._masks = {}

    def mask(self, view, classes, tol_px=2, drawn_only=False, smooth_px=0, close_px=0):
        """Class mask of `view` dilated by tol_px."""
        key = (view, tuple(classes), tol_px, drawn_only, smooth_px, close_px)
        if key not in self._masks:
            m = np.isin(self.lab, [self.ids[c] for c in classes if c in self.ids])
            if drawn_only:
                m = m & ~self.ink
            c0, c1 = self.panels[view]
            mm = np.zeros_like(m)
            mm[:, c0:c1] = m[:, c0:c1]
            if smooth_px > 0 or close_px > 0:
                r = int(smooth_px)

                def grow(a, n):
                    for _ in range(n):
                        a = dilate8(a)
                    return a

                def shrink(a, n):
                    return ~grow(~a, n)
                # close by the larger radius (gaps between ribbons), open by the smaller (pixel crumbs)
                rc = max(r, int(close_px))
                if rc > 0:
                    mm = shrink(grow(mm, rc), rc)
                if r > 0:
                    mm = grow(shrink(mm, r), r)
            for _ in range(int(tol_px)):
                mm = dilate8(mm)
            self._masks[key] = mm
        return self._masks[key]

    def mask_under(self, view, own, occluders, tol_px=2, smooth_px=5, close_px=0, gate_px=0, over=None):
        """Mask of `own` plus occluder pixels inside its outline, so cloth continues under a bow or a strand."""
        key = ('under', view, tuple(own), tuple(occluders), tol_px, smooth_px, close_px, gate_px, tuple(over or ()))
        if key not in self._masks:
            base = self.mask(view, own, 0, smooth_px=smooth_px, close_px=close_px)
            if occluders and gate_px > 0:
                gate = self.mask(view, own, 0, smooth_px=0, close_px=gate_px)
                occ = self.mask(view, occluders, 0)
                base = base | (occ & gate)
                # an occluder bordered 60 % by this garment sits on it, if it is worn over it
                own0 = self.mask(view, own, 0)
                occ_over = self.mask(view, [c for c in occluders if c in (over or [])], 0) if over else np.zeros_like(occ)
                lab_o = label8(occ_over & ~own0)
                for k in range(1, lab_o.max() + 1):
                    blob = lab_o == k
                    if blob.sum() < 25:
                        continue
                    border = dilate8(blob) & ~blob
                    n_b = border.sum()
                    if n_b and (border & (own0 | base)).sum() >= 0.6 * n_b:
                        base = base | blob
                r = int(smooth_px)
                if r > 0:
                    def grow(a, n):
                        for _ in range(n):
                            a = dilate8(a)
                        return a
                    base = ~grow(~grow(base, r), r)
            mm = base
            for _ in range(int(tol_px)):
                mm = dilate8(mm)
            self._masks[key] = mm
        return self._masks[key]

    def project(self, view, p):
        row = self.floor - p.z * self.ppm
        if view == 'front':
            col = self.pan['front']['midline'] + p.x * self.ppm
        elif view == 'back':
            col = self.pan['back']['midline'] - p.x * self.ppm
        else:
            col = self.dcol + p.y * self.ppm * self.side_sign
        return int(round(row)), int(round(col))

    def view_dir(self, view):
        if view == 'front':
            return Vector((0.0, -1.0, 0.0))
        if view == 'back':
            return Vector((0.0, 1.0, 0.0))
        # the side panel shows the left flank when +u runs toward the back
        return Vector((1.0, 0.0, 0.0)) if self.side_sign > 0 else Vector((-1.0, 0.0, 0.0))


def _pieces(bm):
    seen, pieces = set(), []
    for f in bm.faces:
        if f in seen:
            continue
        stack, piece = [f], []
        seen.add(f)
        while stack:
            q = stack.pop()
            piece.append(q)
            for e in q.edges:
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        pieces.append(piece)
    return pieces


def clip_to_views(ob, body_bvh, sheet, classes, tol_px=2, views=('front', 'side', 'back'), log=print, min_piece=6, exempt=None, region=None, pull_max=PULL_MAX, drawn_only=False, smooth_px=5, close_mm=0.0, own=None, gate_mm=0.0, over=None, split_edges=True):
    """Pull vertices outside the drawn mask to its edge, or cut them away when pull_max is 0."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    stats = {}
    for view in views:
        if view not in sheet.panels:
            continue
        if own:
            m = sheet.mask_under(view, own, [c for c in classes if c not in own], tol_px, smooth_px=smooth_px,
                                 close_px=int(round(close_mm * sheet.ppm)), gate_px=int(round(gate_mm * sheet.ppm)), over=over)
        else:
            m = sheet.mask(view, classes, tol_px, drawn_only=drawn_only, smooth_px=smooth_px, close_px=int(round(close_mm * sheet.ppm)))
        vd = sheet.view_dir(view)

        def probe_pt(co):
            if view == 'side':
                return Vector((abs(co.x) if vd.x > 0 else -abs(co.x), co.y, co.z))
            return co

        def in_mask(co):
            r, c = sheet.project(view, probe_pt(co))
            return 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and bool(m[r, c])

        def occluded(co):
            p = probe_pt(co)
            return body_bvh.ray_cast(p + vd * 0.0005, vd, 2.0)[0] is not None

        def inside(co):
            return occluded(co) or in_mask(co)
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        # a view judges a vertex only if one of its faces faces that view
        def out_dir(f):
            # outward is away from the body's nearest point, not the winding
            c = f.calc_center_median()
            hit = body_bvh.find_nearest(c) if body_bvh is not None else (None,)
            if hit[0] is not None and (c - hit[0]).length > 0.001:
                d = (c - hit[0]).normalized()
                if (c - hit[0]).length < 0.06:
                    return d if d.dot(f.normal) >= 0 else -f.normal if f.normal.dot(d) < -0.3 else d
            return f.normal

        def faces_view(v):
            return any(out_dir(f).dot(vd) > 0.25 for f in v.link_faces)
        ins = {v: (inside(v.co) or not faces_view(v)) for v in bm.verts}
        # pull targets must be inside the drawn mask, not just hidden by the body
        in_m = {v: in_mask(v.co) for v in bm.verts}
        n_ex = 0
        if exempt is not None or region is not None:
            for f in bm.faces:
                if (exempt is not None and exempt(f)) or (region is not None and not region(f)):
                    n_ex += 1
                    for v in f.verts:
                        ins[v] = True
        # seam vertices count as inside: never pulled or cut
        sm_ = bm.verts.layers.int.get('s3d_seam')
        if sm_ is not None:
            for v in bm.verts:
                if v[sm_]:
                    ins[v] = True
                    in_m[v] = True
        n_out = sum(1 for v in bm.verts if not ins[v])
        if n_out == 0:
            stats[view] = (0, 0, 0)
            log('[carve]   %s: nothing outside the drawn mask (%d faces exempt)' % (view, n_ex))
            continue
        # split crossing edges where they leave the mask
        n_split = 0
        for e in (list(bm.edges) if split_edges else []):
            va, vb = e.verts
            if ins[va] == ins[vb]:
                continue
            vin, vout = (va, vb) if ins[va] else (vb, va)
            if not in_m.get(vin, False):
                continue
            a_, b_ = vin.co.copy(), vout.co.copy()
            lo, hi = 0.0, 1.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if in_mask(a_.lerp(b_, mid)):
                    lo = mid
                else:
                    hi = mid
            t = max(0.02, min(0.98, lo))
            if split_edges == 'smart':
                pc = a_.lerp(b_, lo)
                ed = (b_ - a_)
                if view in ('front', 'back'):
                    nrm_ = Vector((-ed.z, 0.0, ed.x))
                else:
                    nrm_ = Vector((0.0, -ed.z, ed.y))
                if nrm_.length > 1e-9:
                    nrm_ = nrm_.normalized() * (4.0 / sheet.ppm)
                    if in_mask(pc + nrm_) != in_mask(pc - nrm_):
                        continue
            fac = t if e.verts[0] is vin else 1.0 - t
            ne, nv = bmesh.utils.edge_split(e, e.verts[0], fac)
            ins[nv] = True
            in_m[nv] = in_mask(nv.co) or occluded(nv.co) and in_m.get(vin, False)
            n_split += 1
        # pull outside vertices toward the shell's local axis, drop faces where the axis is outside too
        all_co = [(v.co.copy()) for v in bm.verts]
        pulled, far = 0, set()
        moves = {}
        for v in [q for q in bm.verts if not ins.get(q, True)]:
            # first choice: slide along the surface toward an inside neighbour
            adj = [e.other_vert(v) for e in v.link_edges]
            r_v, c_v = sheet.project(view, probe_pt(v.co))
            adj = [u for u in adj if ins.get(u, False) and in_m.get(u, False)
                   and max(abs(sheet.project(view, probe_pt(u.co))[0] - r_v), abs(sheet.project(view, probe_pt(u.co))[1] - c_v)) > 3]
            if adj:
                u = min(adj, key=lambda q: (q.co - v.co).length)
                a_, b_ = u.co.copy(), v.co.copy()
                lo, hi = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    if in_mask(a_.lerp(b_, mid)):
                        lo = mid
                    else:
                        hi = mid
                dest = a_.lerp(b_, max(0.0, lo - 0.02))
                if (dest - v.co).length <= pull_max:
                    moves[v] = dest
                    continue
            near = [c for c in all_co if abs(c.z - v.co.z) < 0.012 and abs(c.x - v.co.x) < 0.09 and abs(c.y - v.co.y) < 0.15]
            if len(near) < 4:
                far.add(v)
                continue
            axis = sum(near, Vector()) / len(near)
            a_, b_ = axis, v.co.copy()
            if not in_mask(a_):
                far.add(v)
                continue
            lo, hi = 0.0, 1.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if in_mask(a_.lerp(b_, mid)):
                    lo = mid
                else:
                    hi = mid
            dest = a_.lerp(b_, max(0.0, lo - 0.02))
            if (dest - v.co).length > pull_max:
                far.add(v)
            else:
                moves[v] = dest
        doomed, rebuilt = [], []
        for f in list(bm.faces):
            vs = [l.vert for l in f.loops]
            if not any(v in far for v in vs):
                continue
            keep = [v for v in vs if v not in far]
            doomed.append(f)
            if len(keep) >= 3:
                rebuilt.append((keep, f))
        for keep, f in rebuilt:
            try:
                nf = bm.faces.new(keep)
                nf.copy_from(f)
            except ValueError:
                pass
        n_gone = len(doomed) - len(rebuilt)
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context='FACES_ONLY')
        far_v = [v for v in far if v.is_valid]
        if far_v:
            bmesh.ops.delete(bm, geom=far_v, context='VERTS')
        for v, dest in moves.items():
            if v.is_valid:
                v.co = dest
                ins[v] = True
                pulled += 1
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        n_far = len(far)
        stats[view] = (n_split, pulled, n_gone)
        log('[carve]   %s: %d faces exempt, %d verts outside, %d edges split, %d verts pulled in, %d far verts / %d faces removed' % (view, n_ex, n_out, n_split, pulled, n_far, n_gone))
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0006)
    bmesh.ops.dissolve_degenerate(bm, dist=0.0006, edges=bm.edges[:])
    flat = [f for f in bm.faces if f.calc_area() < 2e-7]
    if flat:
        bmesh.ops.delete(bm, geom=flat, context='FACES')
    dust = [f for pc in _pieces(bm) if len(pc) < min_piece for f in pc]
    if dust:
        bmesh.ops.delete(bm, geom=dust, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # relax the cut edge so it reads as a curve
    frozen = seam_verts(bm)
    edge_v = set(v for v in bm.verts if v.is_valid and v not in frozen and any(e.is_boundary for e in v.link_edges))
    for _ in range(5):
        new_co = {}
        for v in edge_v:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new_co[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
        for v, co in new_co.items():
            v.co = co
    # dissolve boundary vertices on nearly straight runs or between edges under 4 mm
    n_diss = 0
    for _ in range(2):
        doomed = []
        for v in bm.verts:
            if not v.is_valid or v in frozen:
                continue
            be = [e for e in v.link_edges if e.is_boundary]
            if len(be) != 2 or len(v.link_edges) > 4:
                continue
            a = be[0].other_vert(v).co - v.co
            b = be[1].other_vert(v).co - v.co
            if a.length < 1e-9 or b.length < 1e-9:
                continue
            ang = a.angle(b)
            short = a.length < 0.004 and b.length < 0.004
            if ang > math.radians(165.0) or short:
                doomed.append(v)
        if not doomed:
            break
        bmesh.ops.dissolve_verts(bm, verts=doomed)
        n_diss += len(doomed)
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    n_skin = 0
    if body_bvh is not None:
        for v in bm.verts:
            if v in frozen:
                continue
            if in_skin(body_bvh, v.co):
                hit_ = body_bvh.find_nearest(v.co)
                if hit_[0] is not None:
                    v.co = hit_[0] + hit_[1] * 0.003
                    n_skin += 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    log('[carve] %s clipped: %s; %d dust faces; %d edge slivers dissolved; %d lifted off the skin; %d faces left' % (ob.name, ', '.join('%s %d splits / %d pulled / %d removed' % (k, a, b, c) for k, (a, b, c) in stats.items()), len(dust), n_diss, n_skin, len(ob.data.polygons)))
    return stats


def in_skin(body_bvh, co):
    dd, pp, n_ = Vector((0.4472, 0.5774, 0.6832)), co.copy(), 0
    for _ in range(24):
        hit_ = body_bvh.ray_cast(pp, dd, 100.0)
        if hit_[0] is None:
            break
        n_ += 1
        pp = hit_[0] + dd * 1e-5
    return n_ % 2 == 1


def remove_fins(ob, log=print):
    """Delete the smallest face at each non-manifold edge, then pieces under 4 faces."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n_v0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0004)
    if len(bm.verts) != n_v0:
        log('[carve] %s: %d near-double vert(s) merged' % (ob.name, n_v0 - len(bm.verts)))
    n = 0
    for _ in range(8):
        bad = [e for e in bm.edges if len(e.link_faces) > 2]
        if not bad:
            break
        doomed = set()
        for e in bad:
            doomed.add(min(e.link_faces, key=lambda q: q.calc_area()))
        bmesh.ops.delete(bm, geom=list(doomed), context='FACES')
        n += len(doomed)
    dust = [f for pc in _pieces(bm) if len(pc) < 4 for f in pc]
    if dust:
        bmesh.ops.delete(bm, geom=dust, context='FACES')
        n += len(dust)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # split bow-tie vertices so each fan keeps its own corner
    n_bow = 0
    for v in [q for q in bm.verts if not q.is_manifold and q.link_faces]:
        if not v.is_valid:
            continue
        bnd = [e for e in v.link_edges if e.is_boundary]
        if len(bnd) >= 4:
            try:
                new_vs = bmesh.utils.vert_separate(v, bnd[:2])
                for nv in list(new_vs) + [v]:
                    if nv.is_valid and nv.link_faces:
                        c = sum((f.calc_center_median() for f in nv.link_faces), Vector()) / len(nv.link_faces)
                        d = c - nv.co
                        if d.length > 1e-6:
                            nv.co = nv.co + d.normalized() * 0.0008
                n_bow += 1
            except Exception:
                pass
    wire = [e for e in bm.edges if not e.link_faces]
    if wire:
        bmesh.ops.delete(bm, geom=wire, context='EDGES')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    if n or n_bow:
        log('[carve] %s: %d fin / dust face(s) removed, %d bow-tie vertices split' % (ob.name, n, n_bow))
    return n


def triangulate(ob, log=print):
    """Triangulate by the shortest diagonal so the audit sees what the renderer draws."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n0 = len(bm.faces)
    quads = [f for f in bm.faces if len(f.verts) > 3]
    if quads:
        bmesh.ops.triangulate(bm, faces=quads, quad_method='SHORT_EDGE', ngon_method='BEAUTY')
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    for pg in ob.data.polygons:
        pg.use_smooth = False
    return len(ob.data.polygons) - n0


def fix_selfx(ob, log=print, rounds=6):
    """Untangle intersecting non-adjacent faces, a few rounds."""
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    frozen_ = seam_verts(bm)
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0008)
    bmesh.ops.dissolve_degenerate(bm, dist=0.0008, edges=bm.edges[:])
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big)
    n1 = len(bm.verts)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    if n0 != n1:
        log('[carve] %s: %d near-double vert(s) merged' % (ob.name, n0 - n1))
    n_total = 0
    for _ in range(rounds):
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.faces.ensure_lookup_table()
        vs, fs = [v.co.copy() for v in bm.verts], []
        bm.verts.ensure_lookup_table()
        for f in bm.faces:
            fs.append([v.index for v in f.verts])
        bvh = BVHTree.FromPolygons(vs, fs, all_triangles=False)
        pairs = bvh.overlap(bvh)
        doomed = set()
        moves = {}
        for a, b in pairs:
            if a == b:
                continue
            fa, fb = bm.faces[a], bm.faces[b]
            if set(fa.verts) & set(fb.verts):
                continue
            doomed.add(min((fa, fb), key=lambda f: f.calc_area()))
            # untangle, never shrink: move the vertices behind the other face's plane just in front of it
            for f, g in ((fa, fb), (fb, fa)):
                n = g.normal
                if n.length < 1e-9:
                    continue
                cg = g.calc_center_median()
                for v in f.verts:
                    if v in frozen_:
                        continue
                    dpt = (v.co - cg).dot(n)
                    if -0.02 < dpt < 0.0:
                        mv = n * (-dpt + 0.001)
                        if v not in moves or moves[v].length < mv.length:
                            moves[v] = mv
        if not doomed:
            bm.free()
            break
        for v, mv in moves.items():
            v.co = v.co + mv
        n_total += len(doomed)
        bm.to_mesh(ob.data)
        bm.free()
        ob.data.update()
    if n_total:
        log('[carve] %s: %d self-intersecting face(s) treated' % (ob.name, n_total))
    return n_total


def orient_outward(ob, body_bvh, log=print):
    """Recalculate normals and flip pieces that face the body."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    n_rev = 0
    score = 0.0
    for f in bm.faces:
        c = f.calc_center_median()
        hit = body_bvh.find_nearest(c)
        if hit[0] is None:
            continue
        d = c - hit[0]
        if d.length < 0.0005:
            continue
        score += f.normal.dot(d.normalized()) * f.calc_area()
    if score < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        n_rev = len(bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    if n_rev:
        log('[carve] %s: %d face(s) reversed so the shell faces away from the body' % (ob.name, n_rev))
    return n_rev


def clear_from_meshes(ob, others, gap=0.003, log=print, rounds=3, rigid=False):
    """Push `ob` out to `gap` off the outer side of `others`, as a whole when `rigid`."""
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    trees = []
    for o in others:
        if o is None or o.type != 'MESH' or not len(o.data.polygons):
            continue
        t = bmesh.new(); t.from_mesh(o.data); t.transform(o.matrix_world)
        t.normal_update()
        trees.append((o.name, BVHTree.FromBMesh(t)))
        t.free()
    if not trees:
        return 0
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    frozen = seam_verts(bm)
    n_total = 0
    for _ in range(rounds):
        moves = {}
        worst = Vector((0, 0, 0))
        samples = [(v, v.co) for v in bm.verts] + [(f, f.calc_center_median()) for f in bm.faces]
        for owner, co in samples:
            for name_, tree in trees:
                loc, nrm, _i, dist = tree.find_nearest(co, gap * 3)
                if loc is None or nrm is None or nrm.length < 1e-6:
                    continue
                d = (co - loc).dot(nrm)
                if d >= gap:
                    continue
                mv = nrm * (gap - d)
                if rigid:
                    if mv.length > worst.length:
                        worst = mv
                    continue
                vs = [owner] if isinstance(owner, bmesh.types.BMVert) else list(owner.verts)
                for v in vs:
                    if v in frozen:
                        continue
                    if v not in moves or moves[v].length < mv.length:
                        moves[v] = mv
        if rigid:
            if worst.length < 1e-5:
                break
            for v in bm.verts:
                v.co += worst
            n_total += 1
            continue
        if not moves:
            break
        for v, mv in moves.items():
            v.co += mv
        n_total += len(moves)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    if n_total:
        log('[carve] %s: %s so it stands off %s' % (ob.name, ('moved as a whole %d time(s)' % n_total) if rigid else ('%d vertex move(s)' % n_total), ', '.join(n for n, _ in trees)))
    return n_total


def faces_clear_skin(ob, body, clearance=0.003, log=print, rounds=3):
    """Lift faces whose centroid or edge midpoints are inside the skin."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    frozen = seam_verts(bm)
    n_total = 0
    for _ in range(rounds):
        moves = {}
        for f in bm.faces:
            vs = list(f.verts)
            samples = [f.calc_center_median()] + [0.5 * (vs[i].co + vs[(i + 1) % len(vs)].co) for i in range(len(vs))]
            for q in samples:
                if not body.in_skin(q):
                    continue
                hit = body.bvh.find_nearest(q)
                if hit[0] is None:
                    continue
                d = hit[0] - q
                if d.length < 1e-9:
                    continue
                mv = d + d.normalized() * clearance
                for v in vs:
                    if v in frozen:
                        continue
                    if v not in moves or moves[v].length < mv.length:
                        moves[v] = mv
        if not moves:
            break
        for v, mv in moves.items():
            v.co = v.co + mv
        n_total += len(moves)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    if n_total:
        log('[carve] %s: %d vertex move(s) so no face passes through the skin' % (ob.name, n_total))
    return n_total


def pin_edge_to_skin(ob, body_bvh, max_dist=0.015, lift=0.0006, log=print, taper=0.0):
    """Snap boundary vertices within max_dist onto the skin so the edge ends in a crease."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n = 0
    edge_v = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    for v in edge_v:
        hit = body_bvh.find_nearest(v.co)
        if hit[0] is None or (hit[0] - v.co).length > max_dist:
            continue
        v.co = hit[0] + hit[1] * lift
        n += 1
    if taper > 0:
        edge_set = set(edge_v)
        inner = set()
        for v in edge_v:
            for e in v.link_edges:
                u = e.other_vert(v)
                if u not in edge_set:
                    inner.add(u)
        for v in inner:
            hit = body_bvh.find_nearest(v.co)
            if hit[0] is None or (hit[0] - v.co).length > max_dist:
                continue
            target = hit[0] + hit[1] * taper
            if (v.co - hit[0]).length > taper:
                v.co = 0.5 * (v.co + target)
                n += 1
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    log('[carve] %s: %d edge verts pinned to the skin' % (ob.name, n))
    return n
