"""Grow a garment shell from the body's faces over a set of rings.

Each vertex moves out to the drawn section, never below R (1 - cos(pi / n)) + clearance + thick.
"""
import bpy, bmesh, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class BodyRef:
    def __init__(self, ob):
        self.ob = ob
        self.me = ob.data
        self.co = [v.co.copy() for v in self.me.vertices]
        self.nrm = [v.normal.copy() for v in self.me.vertices]
        bm = bmesh.new()
        bm.from_mesh(self.me)
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()
        self.polys = [list(p.vertices) for p in self.me.polygons]

    def in_skin(self, co):
        d, p, n = Vector((0.4472, 0.5774, 0.6832)), co.copy(), 0
        for _ in range(24):
            hit = self.bvh.ray_cast(p, d, 100.0)
            if hit[0] is None:
                break
            n += 1
            p = hit[0] + d * 1e-5
        return n % 2 == 1

    def lift_out(self, co, clearance):
        if not self.in_skin(co):
            return None
        hit = self.bvh.find_nearest(co)
        if hit[0] is None:
            return None
        d = hit[0] - co
        if d.length < 1e-9:
            return None
        return hit[0] + d.normalized() * clearance

    def ring_out(self, ring, neigh=None):
        c = sum((self.co[i] for i in ring), Vector()) / len(ring)
        t = Vector((0.0, 0.0, 1.0))
        if neigh:
            d = (sum((self.co[i] for i in neigh), Vector()) / len(neigh)) - c
            if d.length > 1e-9:
                t = d.normalized()
        dots = []
        for i in ring:
            rad = self.co[i] - c
            rad -= t * rad.dot(t)
            dots.append(self.nrm[i].dot(rad.normalized()) if rad.length > 1e-9 else 0.0)
        sgn = 1.0 if sum(dots) >= 0.0 else -1.0
        return dict((i, self.nrm[i] * sgn) for i in ring), c, t

    def ring_floor(self, ring, clearance, thick):
        n = len(ring)
        c = sum((self.co[i] for i in ring), Vector()) / n
        R = max((self.co[i] - c).length for i in ring)
        return R * (1.0 - math.cos(math.pi / n)) + clearance + thick


def emit(name, bm, props=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for p in me.polygons:
        p.use_smooth = False
    for k, v in (props or {}).items():
        ob[k] = v
    return ob


def shell(body, ringlists, name, target=None, clearance=0.0015, thick=0.0028, cols=None, props=None, log=print,
          keep_face=None, smooth=0, extra=None):
    """Return the shell object. target(z, ring) gives the drawn section in metres, or None."""
    bm, vm, outd = bmesh.new(), {}, {}
    chosen = set()
    limb_v = set()
    want, floors, limb_i = {}, {}, set()
    for i in (extra or ()):
        n_ = body.nrm[i]
        if n_.length < 1e-6:
            continue
        chosen.add(i)
        fl = clearance + thick + 0.001
        want[i] = n_ * fl
        floors[i] = fl
        outd[i] = n_.normalized()
    for rl in ringlists:
        for k, r in enumerate(rl):
            nb = rl[k + 1] if k + 1 < len(rl) else (rl[k - 1] if len(rl) > 1 else None)
            od, c, t = body.ring_out(r, nb)
            floor = body.ring_floor(r, clearance, thick)
            zc = c.z
            tg = target(zc, r) if target else None
            if tg:
                xs = [body.co[i].x for i in r]
                ys = [body.co[i].y for i in r]
                wb = max(xs) - min(xs)
                db = max(ys) - min(ys)
                cxb = 0.5 * (max(xs) + min(xs))
                cyb = 0.5 * (max(ys) + min(ys))
                sx = (tg['w'] / wb) if wb > 1e-6 else 1.0
                sy = (tg['d'] / db) if db > 1e-6 else 1.0
                if not tg.get('cap'):
                    sx, sy = max(1.0, sx), max(1.0, sy)
                cxt = tg.get('x_c', cxb) if tg.get('x_c') is not None else cxb
                cyt = tg.get('y_mid', cyb) if tg.get('y_mid') is not None else cyb
            sel = r if cols is None or id(r) not in cols else cols[id(r)]
            for i in sel:
                chosen.add(i)
                if i in want:
                    continue
                co0 = body.co[i]
                if tg:
                    co = Vector((cxt + (co0.x - cxb) * sx, cyt + (co0.y - cyb) * sy, co0.z))
                    if (co - co0).dot(od[i]) < floor:
                        co = co0 + od[i] * floor
                else:
                    co = co0 + od[i] * floor
                want[i] = co - co0
                floors[i] = floor
                if rl is not ringlists[0]:
                    limb_i.add(i)
                outd[i] = od[i]
    # smooth the offsets over neighbours so rings of different widths meet without a fold
    adj = {}
    for vs in body.polys:
        if all(i in chosen for i in vs):
            n_ = len(vs)
            for k, i in enumerate(vs):
                adj.setdefault(i, set()).update((vs[k - 1], vs[(k + 1) % n_]))
    # smooth only the radial part: whole vectors cancel across a concave junction
    rad = {i: dv.dot(outd[i]) for i, dv in want.items()}
    tan = {i: dv - outd[i] * rad[i] for i, dv in want.items()}
    for _ in range(smooth):
        new = {}
        for i, r_ in rad.items():
            nb = [rad[j] for j in adj.get(i, ()) if j in rad]
            new[i] = r_ * 0.5 + (sum(nb) / len(nb)) * 0.5 if len(nb) >= 2 else r_
        rad = new
    src_layer = bm.verts.layers.int.new('s3d_src')
    for i in want:
        co0 = body.co[i]
        co = co0 + outd[i] * max(floors[i], rad[i]) + tan[i]
        vm[i] = bm.verts.new(co)
        vm[i][src_layer] = i
        if i in limb_i:
            limb_v.add(vm[i])
    for vs in body.polys:
        if all(i in chosen for i in vs):
            if keep_face and not keep_face([body.co[i] for i in vs]):
                continue
            try:
                bm.faces.new([vm[i] for i in vs])
            except ValueError:
                pass
    for v in [v for v in bm.verts if not v.link_faces]:
        bm.verts.remove(v)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    dict((bv, i) for i, bv in vm.items() if bv.is_valid)
    bm.normal_update()
    bm.faces.layers.int.new('s3d_rim')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    n_lift = 0
    n_push = 0
    for v in bm.verts:
        co = body.lift_out(v.co.copy(), clearance)
        # a lift over 12 mm means the vertex is inside another part, not a clearance fix
        if co is not None and (co - v.co).length <= 0.012:
            n_lift += 1
            v.co = co
        hit = body.bvh.find_nearest(v.co)
        if hit[0] is not None:
            d = v.co - hit[0]
            need = clearance if v not in limb_v else 2.0 * (clearance + thick) + 0.003
            if d.length < need - 1e-5 and d.dot(hit[1]) >= 0:
                dest = hit[0] + hit[1] * need
                if (dest - v.co).length <= 0.012:
                    v.co = dest
                    n_push += 1
    # a sleeve never passes into the trunk shell at the armpit
    n_out = 0
    if limb_v:
        tb = bmesh.new()
        trunk_faces = [f for f in bm.faces if not any(v in limb_v for v in f.verts)]
        if trunk_faces:
            vmap = {}
            for f in trunk_faces:
                try:
                    tb.faces.new([vmap.setdefault(v, tb.verts.new(v.co)) for v in f.verts])
                except ValueError:
                    pass
            tb.normal_update()
            tbvh = BVHTree.FromBMesh(tb)
            for v in limb_v:
                if not v.is_valid:
                    continue
                near = tbvh.find_nearest(v.co, 0.05)
                if near[0] is None:
                    continue
                n_ = near[1]
                off = (v.co - near[0]).dot(n_)
                need_ = clearance + thick + 0.002
                if off < need_:
                    v.co = near[0] + n_ * need_
                    n_out += 1
        tb.free()
    nf = len(bm.faces)
    ob = emit(name, bm, props)
    log('[shell] %-10s %d faces%s%s%s' % (name, nf, ('  lifted %d' % n_lift) if n_lift else '', ('  pushed off the skin %d' % n_push) if n_push else '', ('  limb verts brought out of the trunk shell %d' % n_out) if n_out else ''))
    return ob


def ring_z(body, ring):
    return sum(body.co[i].z for i in ring) / len(ring)
