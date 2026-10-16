"""Check mesh hygiene of every object in a .blend.

    blender -b --factory-startup --python sketch3d/judge/audit.py -- <char.blend> <report.json>
"""
import json
import os
import sys

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

FLOAT_MM = 5.0  # mm, a part farther than this from the rest is floating
DEGENERATE_M2 = 1e-9


def islands(bm):
    seen, out = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, isl = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            isl.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        out.append(isl)
    return out


def world_bmesh(o, dg):
    bm = bmesh.new()
    bm.from_object(o, dg)
    bm.transform(o.matrix_world)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def audit_object(o, bm):
    r = dict(tris=sum(len(f.verts) - 2 for f in bm.faces), faces=len(bm.faces))
    tree = BVHTree.FromBMesh(bm)
    pairs = tree.overlap(tree)
    r['selfx'] = sum(1 for a, b in pairs if a < b and not (set(v.index for v in bm.faces[a].verts) & set(v.index for v in bm.faces[b].verts)))
    r['nonmanifold'] = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    r['boundary'] = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    r['degenerate'] = sum(1 for f in bm.faces if f.calc_area() < DEGENERATE_M2)
    r['loose'] = sum(1 for v in bm.verts if not v.link_faces)
    inv = 0
    for isl in islands(bm):
        closed = all(len(e.link_faces) == 2 for f in isl for e in f.edges)
        if closed and metrics.signed_volume([[tuple(l.vert.co) for l in f.loops] for f in isl]) < 0:
            inv += 1
    r['inverted'] = inv
    m = o.matrix_world
    r['transform'] = int(any(abs(m[i][j] - (1.0 if i == j else 0.0)) > 1e-6 for i in range(4) for j in range(4)))
    return r, tree


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    blend, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    bpy.ops.wm.open_mainfile(filepath=blend)
    dg = bpy.context.evaluated_depsgraph_get()
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH' and len(o.data.polygons)]
    rep, trees, verts = {}, {}, {}
    for o in objs:
        bm = world_bmesh(o, dg)
        rep[o.name], trees[o.name] = audit_object(o, bm)
        verts[o.name] = [v.co.copy() for v in bm.verts]
        bm.free()
    def closest(src, tree, best):
        """Return the nearest distance from sampled vertices of src to tree, stopping at contact."""
        for co in verts[src][::max(1, len(verts[src]) // 400)]:
            hit = tree.find_nearest(co)
            if hit[0] is not None and hit[3] < best:
                best = hit[3]
                if best < 1e-4:
                    break
        return best

    for o in objs:
        # both ways: a flat face can rest on a vertex of the other part
        best = 1e9
        for other in trees:
            if other == o.name:
                continue
            best = closest(o.name, trees[other], best)
            best = closest(other, trees[o.name], best)
            if best < 1e-4:
                break
        rep[o.name]['gap_mm'] = round(best * 1000.0, 1) if best < 1e8 else None
    fails = []
    for name, r in sorted(rep.items()):
        bad = [k for k in ('selfx', 'nonmanifold', 'degenerate', 'loose', 'inverted', 'transform') if r[k]]
        if r['gap_mm'] is not None and r['gap_mm'] > FLOAT_MM and len(objs) > 1:
            bad.append('floating')
        for k in bad:
            fails.append('%s: %s %s' % (name, k, r.get(k, r['gap_mm'])))
        print('%s %-22s tris %5d selfx %d nonmanifold %d degenerate %d loose %d inverted %d gap %s mm' % (
            'FAIL' if bad else 'ok  ', name, r['tris'], r['selfx'], r['nonmanifold'], r['degenerate'], r['loose'],
            r['inverted'], r['gap_mm']))
    json.dump(dict(objects=rep, fails=fails), open(out, 'w'), indent=1)
    print('[audit] %d objects, %d failing gates -> %s' % (len(objs), len(fails), out))


if __name__ == '__main__':
    main()
