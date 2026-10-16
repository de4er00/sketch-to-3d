"""Surface checks the audit misses: dents, needles, creases, symmetry, concavity.

    blender -b --factory-startup --python sketch3d/judge/quality.py -- <char.blend> <out.json> [--sym-classes a,b]
"""
import bpy, bmesh, os, sys, json, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

DENT_MM = 8.0
NEEDLE_DEG = 4.0
NEEDLE_MM2 = 6.0
CREASE_DEG = 110.0
CONCAVE_DEG = 2.0
SYM_MM = 3.0
SYM_MAX_MM = 12.0


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    blend, out_path = argv[0], argv[1]
    sym_classes = None
    if '--sym-classes' in argv:
        sym_classes = set(argv[argv.index('--sym-classes') + 1].split(','))
    bpy.ops.wm.open_mainfile(filepath=blend)
    rep = {}
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.transform(ob.matrix_world)
        bm.normal_update()
        part = ob.get('s3d_part', '')
        kind = ob.get('s3d_kind', '')
        r = dict(part=part, cls=ob.get('s3d_class', ''), kind=kind, verts=len(bm.verts), faces=len(bm.faces))
        # a dent is one vertex sunk below its ring, rim vertices are skipped
        rim_layer = bm.faces.layers.int.get('s3d_rim')
        rim_v = set()
        if rim_layer is not None:
            for f in bm.faces:
                if f[rim_layer]:
                    rim_v.update(f.verts)
        # creases the builder marked (armpit, crotch, nape) are not dents
        cr_layer = bm.verts.layers.int.get('s3d_crease')
        if cr_layer is not None:
            rim_v.update(v for v in bm.verts if v[cr_layer])
        conc = {}
        for v in bm.verts:
            if v in rim_v:
                continue
            nb = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) not in rim_v]
            if len(nb) < 3 or v.is_boundary:
                continue
            c = sum((u.co for u in nb), Vector()) / len(nb)
            conc[v] = (c - v.co).dot(v.normal)
        dents = []
        for v, d in conc.items():
            nbc = sorted(conc[u] for u in (e.other_vert(v) for e in v.link_edges) if u in conc)
            med = nbc[len(nbc) // 2] if nbc else 0.0
            excess = d - max(0.0, med)
            if excess * 1000.0 > DENT_MM:
                dents.append(dict(mm=round(excess * 1000.0, 1), at=[round(q, 3) for q in v.co]))
        dents.sort(key=lambda q: -q['mm'])
        r['dents'] = dents[:20]
        r['n_dents'] = len(dents)
        needles = 0
        for f in bm.faces:
            if f.calc_area() * 1e6 < NEEDLE_MM2:
                continue
            vs = f.verts
            n = len(vs)
            amin = 180.0
            for i in range(n):
                a, b, c = vs[i - 1].co, vs[i].co, vs[(i + 1) % n].co
                u, w = a - b, c - b
                if u.length < 1e-9 or w.length < 1e-9:
                    continue
                ang = math.degrees(u.angle(w))
                amin = min(amin, ang)
            if amin < NEEDLE_DEG:
                needles += 1
        r['needles'] = needles
        creases = []
        if part in ('body', 'garment', 'shoe'):
            for e in bm.edges:
                if len(e.link_faces) != 2:
                    continue
                f1, f2 = e.link_faces
                ang = math.degrees(f1.normal.angle(f2.normal)) if f1.normal.length and f2.normal.length else 0.0
                if ang < 180.0 - CREASE_DEG:
                    continue
                # concave when the other face's centre lies in front of this face
                c2 = f2.calc_center_median()
                if (c2 - f1.calc_center_median()).dot(f1.normal) > 0:
                    m = 0.5 * (e.verts[0].co + e.verts[1].co)
                    creases.append(dict(deg=round(180.0 - ang, 0), at=[round(q, 3) for q in m]))
        creases.sort(key=lambda q: q['deg'])
        r['creases'] = creases[:20]
        r['n_creases'] = len(creases)
        edge_len = concave_len = 0.0
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            f1, f2 = e.link_faces
            ln = e.calc_length()
            edge_len += ln
            if f1.normal.length and f2.normal.length and metrics.folds_inward(
                    f1.normal, f1.calc_center_median(), f2.normal, f2.calc_center_median(), CONCAVE_DEG):
                concave_len += ln
        r['edge_len_m'] = round(edge_len, 4)
        r['concave_len_m'] = round(concave_len, 4)
        if sym_classes is None or r['cls'] in sym_classes or part == 'body':
            xs = [v.co.x for v in bm.verts]
            if xs and min(xs) < -0.01 and max(xs) > 0.01 and not ob.get('s3d_side'):
                bvh = BVHTree.FromBMesh(bm)
                ds = []
                for v in bm.verts:
                    m = Vector((-v.co.x, v.co.y, v.co.z))
                    hit = bvh.find_nearest(m)
                    if hit[0] is not None:
                        ds.append((hit[0] - m).length)
                if ds:
                    ds.sort()
                    r['sym_mean_mm'] = round(1000.0 * sum(ds) / len(ds), 2)
                    r['sym_p95_mm'] = round(1000.0 * ds[int(0.95 * (len(ds) - 1))], 2)
                    r['sym_max_mm'] = round(1000.0 * ds[-1], 2)
        rep[ob.name] = r
        bm.free()
    json.dump(rep, open(out_path, 'w'), indent=1)
    print('[quality] %d objects -> %s' % (len(rep), out_path))


if __name__ == '__main__':
    main()
