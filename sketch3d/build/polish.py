"""Final clean-up of every built mesh except the body: crumbs, needles, doubles, pits.

    blender -b --factory-startup --python sketch3d/build/polish.py -- <char.json> <body.json> <in.blend> <out.blend>
"""
import bpy
import bmesh
import sys
import math
from mathutils import Vector

NEEDLE_DEG = 4.0
NEEDLE_MM2 = 6.0
CRUMB_FACES = 4
CRUMB_MM2 = 60.0
DOUBLE_M = 0.0012
PIT_M = 0.008
SKIN_GAP_M = 0.0015


def seam_set(bm):
    sm = bm.verts.layers.int.get('s3d_seam')
    return set(v for v in bm.verts if sm is not None and v[sm]) if sm is not None else set()


def pieces(bm):
    seen = set()
    out = []
    for f in bm.faces:
        if f in seen:
            continue
        stack, pc = [f], []
        seen.add(f)
        while stack:
            g = stack.pop()
            pc.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
        out.append(pc)
    return out


def remove_degenerate(bm):
    dead = [f for f in bm.faces if f.calc_area() * 1e6 < 0.05]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')
    bmesh.ops.dissolve_degenerate(bm, dist=0.0003, edges=bm.edges[:])
    return len(dead)


def remove_crumbs(bm):
    pcs = pieces(bm)
    if len(pcs) <= 1:
        return 0
    main = max(pcs, key=lambda pc: sum(f.calc_area() for f in pc))
    dust = []
    for pc in pcs:
        if pc is main:
            continue
        if len(pc) < CRUMB_FACES or sum(f.calc_area() for f in pc) * 1e6 < CRUMB_MM2:
            dust.extend(pc)
    if dust:
        bmesh.ops.delete(bm, geom=dust, context='FACES')
    return len(dust)


def merge_doubles(bm, frozen):
    """Merge vertices within DOUBLE_M (at most 4 % of the piece size), keeping seam vertices."""
    from mathutils.kdtree import KDTree
    verts = [v for v in bm.verts]
    if not verts:
        return 0
    lo = Vector((min(v.co.x for v in verts), min(v.co.y for v in verts), min(v.co.z for v in verts)))
    hi = Vector((max(v.co.x for v in verts), max(v.co.y for v in verts), max(v.co.z for v in verts)))
    DOUBLE = min(DOUBLE_M, 0.04 * (hi - lo).length)
    kd = KDTree(len(verts))
    for i, v in enumerate(verts):
        kd.insert(v.co, i)
    kd.balance()
    done = set()
    n = 0
    for i, v in enumerate(verts):
        if v in done or not v.is_valid:
            continue
        near = [verts[j] for (_, j, d) in kd.find_range(v.co, DOUBLE) if j != i and verts[j].is_valid and verts[j] not in done]
        if not near:
            continue
        group = [v] + near
        keep = next((q for q in group if q in frozen), None)
        target = keep.co.copy() if keep is not None else sum((q.co for q in group), Vector()) / len(group)
        for q in group:
            done.add(q)
        try:
            bmesh.ops.pointmerge(bm, verts=group, merge_co=target)
            n += len(group) - 1
        except Exception:
            pass
    return n


def needle_faces(bm):
    out = []
    for f in bm.faces:
        if len(f.verts) != 3 or f.calc_area() * 1e6 < NEEDLE_MM2:
            continue
        vs = f.verts
        worst, k_sharp = 180.0, 0
        for i in range(3):
            a, b, c = vs[i - 1].co, vs[i].co, vs[(i + 1) % 3].co
            u, w = a - b, c - b
            if u.length < 1e-9 or w.length < 1e-9:
                continue
            ang = math.degrees(u.angle(w))
            if ang < worst:
                worst, k_sharp = ang, i
        if worst < NEEDLE_DEG:
            out.append((f, k_sharp))
    return out


def fix_needles(bm, frozen, rounds=4, split_ok=True):
    n_total = 0
    for _ in range(rounds):
        nf = needle_faces(bm)
        if not nf:
            break
        handled = set()
        # a sliver with all corners on the open edge is a crumb
        rim_slivers = [f for f, k in nf if f.is_valid and all(v.is_boundary for v in f.verts) and not any(v in frozen for v in f.verts)]
        if rim_slivers:
            for f in rim_slivers:
                handled.update(f.verts)
            bmesh.ops.delete(bm, geom=rim_slivers, context='FACES_ONLY')
            n_total += len(rim_slivers)
        for f, k in nf:
            if not f.is_valid or any(v in handled for v in f.verts):
                continue
            vs = list(f.verts)
            edges = sorted(f.edges, key=lambda e: e.calc_length())
            e0 = edges[0]
            # a long sliver: move the vertex opposite the longest edge onto it
            e_long = edges[-1]
            sharp = next(v for v in vs if v not in e_long.verts)
            a, b = e_long.verts
            n_frozen0 = sum(1 for v in e0.verts if v in frozen)
            if e0.calc_length() < 0.005 or (e0.calc_length() < 0.010 and n_frozen0 == 1):
                va, vb = e0.verts
                keep = va if va in frozen else (vb if vb in frozen else None)
                co = keep.co.copy() if keep is not None else 0.5 * (va.co + vb.co)
                try:
                    bmesh.ops.pointmerge(bm, verts=[va, vb], merge_co=co)
                    handled.update([va, vb])
                    n_total += 1
                except Exception:
                    pass
                continue
            if not split_ok:
                continue
            ab = b.co - a.co
            if ab.length < 1e-9:
                continue
            t = max(0.02, min(0.98, (sharp.co - a.co).dot(ab) / ab.length_squared))
            e_opp = next((e for e in f.edges if set(e.verts) == {a, b}), None)
            if e_opp is None:
                continue
            try:
                ne, nv = bmesh.utils.edge_split(e_opp, a, t)
                bmesh.ops.pointmerge(bm, verts=[sharp, nv], merge_co=(sharp.co.copy() if sharp in frozen else nv.co.copy()))
                handled.update([sharp, a, b])
                n_total += 1
            except Exception:
                pass
        bmesh.ops.dissolve_degenerate(bm, dist=0.0004, edges=bm.edges[:])
        big = [g for g in bm.faces if len(g.verts) > 3]
        if big:
            bmesh.ops.triangulate(bm, faces=big, quad_method='SHORT_EDGE', ngon_method='BEAUTY')
    return n_total


def fill_pits(bm, frozen, skin=None, rounds=3):
    """Raise vertices sunk more than PIT_M below their neighbours' median."""
    n_total = 0
    for _ in range(rounds):
        bm.normal_update()
        conc = {}
        for v in bm.verts:
            if v in frozen or v.is_boundary or len(v.link_edges) < 3:
                continue
            nb = [e.other_vert(v) for e in v.link_edges]
            c = sum((u.co for u in nb), Vector()) / len(nb)
            conc[v] = (c - v.co).dot(v.normal)
        moves = {}
        for v, d in conc.items():
            nbc = sorted(conc[u] for u in (e.other_vert(v) for e in v.link_edges) if u in conc)
            med = nbc[len(nbc) // 2] if nbc else 0.0
            excess = d - max(0.0, med)
            if excess > PIT_M:
                moves[v] = v.co + v.normal * excess
        if not moves:
            break
        for v, co in moves.items():
            if skin is not None:
                hit = skin.find_nearest(co)
                if hit[0] is not None and (co - hit[0]).dot(hit[1]) < SKIN_GAP_M:
                    co = hit[0] + hit[1] * SKIN_GAP_M
            v.co = co
        n_total += len(moves)
    return n_total


def health(bm):
    """(non-manifold edges, intersecting face pairs): what a pass must not worsen."""
    from mathutils.bvhtree import BVHTree
    nm = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    b2 = bm.copy()
    bmesh.ops.triangulate(b2, faces=b2.faces[:])
    b2.faces.ensure_lookup_table()
    t = BVHTree.FromBMesh(b2, epsilon=0.0)
    n = 0
    seen = set()
    for a, b in t.overlap(t):
        if a == b:
            continue
        fa, fb = b2.faces[a], b2.faces[b]
        if set(fa.verts) & set(fb.verts):
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            n += 1
    b2.free()
    return nm, n


def polish(ob, skin=None, log=print):
    """Apply each step only if it adds no non-manifold edge or self-intersection."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    closed = not any(e.is_boundary for e in bm.edges)
    counts = dict(degenerate=0, crumbs=0, doubles=0, needles=0, pits=0)
    reverted = []

    def step(name, fn):
        nonlocal bm
        h0 = health(bm)
        trial = bm.copy()
        n = fn(trial)
        if not n:
            trial.free()
            return
        h1 = health(trial)
        if h1[0] > h0[0] or h1[1] > h0[1]:
            trial.free()
            reverted.append(name)
            return
        bm.free()
        bm = trial
        counts[name] += n
    step('degenerate', lambda b: remove_degenerate(b))
    step('crumbs', lambda b: remove_crumbs(b))
    step('doubles', lambda b: merge_doubles(b, seam_set(b)))
    # closed solids keep their topology: only short edges collapse
    step('needles', lambda b: fix_needles(b, seam_set(b), split_ok=not closed))
    step('pits', lambda b: fill_pits(b, seam_set(b), skin))
    step('degenerate', lambda b: remove_degenerate(b))
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    left = len(needle_faces(bm))
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    for pg in ob.data.polygons:
        pg.use_smooth = False
    if any(counts.values()) or left or reverted:
        log('[polish] %-16s degenerate %d, crumbs %d, doubles %d, needles fixed %d, pits raised %d%s%s' % (
            ob.name, counts['degenerate'], counts['crumbs'], counts['doubles'], counts['needles'], counts['pits'], (', %d left' % left) if left else '',
            (' (dropped: %s - they would have hurt the mesh)' % ', '.join(reverted)) if reverted else ''))
    return left


def needle_faces_of(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    out = needle_faces(bm)
    n = len(out)
    bm.free()
    return [None] * n


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, body_json, in_blend, out_blend = argv[:4]
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    body = next((o for o in bpy.data.objects if o.type == 'MESH' and o.get('s3d_part') == 'body'), None)
    skin = None
    if body is not None:
        from mathutils.bvhtree import BVHTree
        skin = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    n_left = 0
    for ob in list(bpy.data.objects):
        if ob.type != 'MESH' or ob.get('s3d_part') == 'body':
            continue
        if len(ob.data.polygons) == 0:
            print('[polish] %-16s empty - removed' % ob.name)
            bpy.data.objects.remove(ob, do_unlink=True)
            continue
        n_left += polish(ob, skin=skin, log=print)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[polish] done, %d needle(s) left -> %s' % (n_left, out_blend))


if __name__ == '__main__':
    main()
