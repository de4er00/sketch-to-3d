"""Build garment shells over the body from its own faces.

    blender -b --factory-startup --python sketch3d/build/garments.py -- <char.json> <body.json> <in.blend> <out.blend>
"""
import bpy, bmesh, math, os, sys, json, re
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from carve import Sheet, clip_to_views, fix_selfx, remove_fins, triangulate, faces_clear_skin, clear_from_meshes, orient_outward   # noqa: E402
from shell import BodyRef, shell, ring_z   # noqa: E402

THICK = {'jacket': 0.0030, 'coat': 0.0030, 'shirt': 0.0022, 'blouse': 0.0022, 'jeans': 0.0026, 'trousers': 0.0026,
         'cuff': 0.0030, 'legwarmer': 0.0035, 'default': 0.0026}
CLEAR = {'jacket': 0.0045, 'coat': 0.0045, 'default': 0.0015}


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lexicon import layer_rank  # noqa: E402


def st_at(stations, z, keys=('w_mm', 'd_mm', 'y_mid_mm', 'x_c_mm')):
    """Interpolate a smoothed station table at z, None outside its range."""
    st = [s for s in stations if all(s.get(k) is not None for k in keys[:2])]
    if not st:
        return None
    st = sorted(st, key=lambda s: s['z'])
    def edge_bad(q, n):
        if q[keys[0]] < 0.75 * n[keys[0]]:
            return True
        for k in ('x_c_mm', 'y_mid_mm'):
            if q.get(k) is not None and n.get(k) is not None and abs(q[k] - n[k]) > 12.0:
                return True
        return False
    while len(st) >= 2 and edge_bad(st[0], st[1]):
        st = st[1:]
    while len(st) >= 2 and edge_bad(st[-1], st[-2]):
        st = st[:-1]
    if len(st) >= 3:
        for i in range(1, len(st) - 1):
            a, q, b = st[i - 1], st[i], st[i + 1]
            for k in ('y_mid_mm', 'd_mm'):
                if a.get(k) is not None and b.get(k) is not None and q.get(k) is not None:
                    lo, hi = min(a[k], b[k]), max(a[k], b[k])
                    if q[k] < lo - 15 or q[k] > hi + 15:
                        st[i] = dict(q, **{k: 0.5 * (a[k] + b[k])})
                        q = st[i]
    if len(st) >= 3:
        sm = []
        for i, q in enumerate(st):
            nb = st[max(0, i - 1):i + 2]
            q2 = dict(q)
            for k in keys[:2]:
                q2[k] = sum(n[k] for n in nb) / len(nb)
            sm.append(q2)
        st = sm
    zs = [s['z'] for s in st]
    # a ring up to 30 mm past the last station takes that station's section
    if z < zs[0] - 0.03 or z > zs[-1] + 0.03:
        return None
    z = min(max(z, zs[0]), zs[-1])
    for i in range(len(zs) - 1):
        if zs[i] <= z <= zs[i + 1]:
            t = (z - zs[i]) / max(1e-9, zs[i + 1] - zs[i])
            out = {}
            for k in keys:
                a, b = st[i].get(k), st[i + 1].get(k)
                out[k] = (a + (b - a) * t) if a is not None and b is not None else None
            return out
    return None


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, out_blend = argv[:4]
    C = json.load(open(char_path))
    M = json.load(open(map_path))
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    body_ob = bpy.data.objects['Body']
    body = BodyRef(body_ob)
    G = C['garments']
    lm = C['landmarks']
    # shells stop at the neck base, a collar above it is its own band
    trunk = M['trunk']
    legs = M['leg']
    arms = M['arm']
    z_hip = lm['hip']['z']
    z_crotch = lm['crotch']['z']
    made = []
    sheet = Sheet(char_path)

    def trunk_rings_in(z0, z1, tol=0.003):
        return [r for r in trunk if z0 - tol <= ring_z(body, r) <= z1 + tol]

    def leg_rings_in(side, z0, z1, tol=0.003):
        return [r for r in legs[side] if z0 - tol <= ring_z(body, r) <= z1 + tol]

    def target_from(stations, scale_w=1.0):
        def f(z, ring):
            s = st_at(stations, z)
            if s is None:
                return None
            return dict(w=s['w_mm'] / 1000.0 * scale_w, d=s['d_mm'] / 1000.0, y_mid=s['y_mid_mm'] / 1000.0 if s.get('y_mid_mm') is not None else None,
                        x_c=s['x_c_mm'] / 1000.0 if s.get('x_c_mm') is not None else None)
        return f

    # among torso garments overlapping in z the wider one is outer
    torso_g = {k: v for k, v in G.items() if v['kind'] in ('torso', 'skirt') and v.get('stations')}

    def mean_w(g, z0, z1):
        ws = [s['w_mm'] for s in g['stations'] if z0 <= s['z'] <= z1 and s.get('w_mm')]
        return sum(ws) / len(ws) if ws else 0.0
    outer_of = {}
    for a, ga in torso_g.items():
        za0, za1 = sorted(ga['z'])
        for b, gb in torso_g.items():
            if a == b:
                continue
            zb0, zb1 = sorted(gb['z'])
            lo, hi = max(za0, zb0), min(za1, zb1)
            if hi - lo < 0.03:
                continue
            ra, rb = layer_rank(a), layer_rank(b)
            if ra is not None and rb is not None:
                if rb > ra:
                    if a not in outer_of or (layer_rank(outer_of[a]) or 99) > rb:
                        outer_of[a] = b
            elif mean_w(gb, lo, hi) > 1.15 * mean_w(ga, lo, hi) and a not in outer_of:
                outer_of[a] = b
    if outer_of:
        print('[garments] layering (inner -> outer): %s' % outer_of)
    hair_cls = list(C['roles'].get('hair', []))

    def occluders_of(cname):
        # pixels of other classes count as this garment only inside its closed outline
        skin_cls = set(C['roles'].get('skin', ['skin']))
        others = [g for g in G if g != cname and g not in skin_cls] + [c for c in hair_cls if c != cname]
        seen, out = set(), [cname]
        for c in others:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def over_of(cname):
        acc = [g for g, e in G.items() if e['kind'] == 'accessory']
        out = list(acc) + list(hair_cls)
        kind_ = G.get(cname, {}).get('kind')
        if kind_ in ('trousers', 'legs', 'arms', 'skirt'):
            out += [g for g, e in G.items() if e['kind'] in ('torso', 'cape') and g != cname]
        c = cname
        while c in outer_of:
            c = outer_of[c]
            out.append(c)
        return [q for q in dict.fromkeys(out) if q != cname]

    # build what is worn under first, so a later band wraps over it
    order_kind = {'trousers': 0, 'skirt': 1, 'torso': 2, 'cape': 2, 'arms': 3, 'legs': 4}
    items = sorted(G.items(), key=lambda kv: (order_kind.get(kv[1]['kind'], 5), -min(kv[1]['rows']) if kv[1]['kind'] == 'arms' and kv[1].get('rows') else 0))
    worn_over = {}
    for cname, g in items:
        kind = g['kind']
        if kind == 'accessory' or cname in C['roles'].get('shoe', []) + C['roles'].get('sole', []):
            continue
        if g.get('painted') and kind != 'arms':
            print('[garments] %s: tight - painted, no shell' % cname)
            continue
        z0, z1 = sorted(g['z'])
        thick = THICK.get(cname, THICK['default'])
        clear = CLEAR.get(cname, CLEAR['default'])
        Name = cname.capitalize()
        if kind in ('torso', 'skirt', 'cape'):
            z_hem = None
            if kind == 'torso' and re.search(r'collar|cape|shawl|capelet', cname.lower()):
                # collar or cape: a band over the neck, then a bell over the shoulders
                z_hem = z0
                z0 = max(z0, z1 - 0.006)
                kind = 'cape'
            if kind == 'skirt':
                # skirt: a torso shell to the hip, then a bell to the hem
                z_hem = z0
                z0 = max(z0, z_hip - 0.004)
                z_waist = lm.get('waist', {}).get('z')
                st_g = [q for q in g.get('stations', []) if q.get('w_mm')]
                tb_ = [q for q in C['stations']['trunk'] if q.get('w_body_mm')]
                if z_waist is not None and st_g and tb_:
                    q_hip = min(st_g, key=lambda q: abs(q['z'] - z_hip))
                    b_hip = min(tb_, key=lambda q: abs(q['z'] - z_hip))
                    if abs(q_hip['z'] - z_hip) < 0.04 and q_hip['w_mm'] > 1.2 * (b_hip['w_body_mm'] + 2000 * thick):
                        z0 = max(z_hem, min(z_waist, z1) - 0.004)
                        print('[garments] %s: flares from the waist (drawn %.0f mm at the hip vs body %.0f)' % (cname, q_hip['w_mm'], b_hip['w_body_mm']))
            sleeve_lists = []
            for side in ('R', 'L'):
                A = C['stations']['arm_' + side]
                mine = [s for s in A['stations'] if s['cls'] == cname]
                if mine:
                    # a sleeve runs at least 60 mm or 15 % down the arm, a shorter touch is the shoulder
                    s_w = A['landmarks']['wrist']['s_px'] / C['calibration']['ppm']
                    s_lo = min(q['s_m'] for q in mine)
                    s_hi = max(q['s_m'] for q in mine)
                    if s_lo > 0.06 or (s_hi - s_lo) < max(0.05, 0.15 * s_w):
                        print('[garments] %s: on the arm %s only from %.0f to %.0f mm - a shoulder, not a sleeve' % (cname, side, s_lo * 1000, s_hi * 1000))
                        continue
                    wrist_z = A['landmarks']['wrist']['z']
                    zs_cls = [s['z'] for s in mine]
                    z_end = min(zs_cls) if zs_cls else wrist_z
                    ar = [r for r in arms[side] if ring_z(body, r) >= z_end - 0.012]
                    if len(ar) >= 2:
                        sleeve_lists.append(ar)
            if sleeve_lists:
                # the trunk shell reaches the arm hole's top so the sleeve root lies on its rim
                z_loop_top = max(body.co[i].z for ar in sleeve_lists for i in ar[0])
                above = [ring_z(body, r) for r in trunk if ring_z(body, r) > z_loop_top + 0.001]
                z_hole_top = (min(above) if above else z_loop_top) + 0.002
                if z_hole_top > z1:
                    print('[garments] %s: top raised %.3f -> %.3f to cover the arm hole' % (cname, z1, z_hole_top))
                    z1 = z_hole_top
            rings = trunk_rings_in(z0, z1)
            if len(rings) < 2 and kind in ('cape', 'skirt'):
                zs_ = sorted(trunk, key=lambda r: abs(ring_z(body, r) - z1))
                rings = sorted(zs_[:2], key=lambda r: -ring_z(body, r))
            if not rings:
                print('[garments] %s: no trunk rings in z %.3f..%.3f' % (cname, z0, z1))
                continue
            st = g.get('stations', [])
            _tgt_trunk = target_from(st) if kind != 'cape' else (lambda z, ring: None)
            _trunk_ids = set(id(r) for r in rings)
            _z_top = max(ring_z(body, r) for r in rings)
            _outer = G.get(outer_of.get(cname, ''), {})
            _tgt_outer = target_from(_outer.get('stations', [])) if _outer else None
            _t_outer = THICK.get(outer_of.get(cname, ''), THICK['default']) + CLEAR.get(outer_of.get(cname, ''), CLEAR['default'])

            _inner = [k for k, v in outer_of.items() if v == cname]
            grow = True
            while grow:
                grow = False
                for k, v in outer_of.items():
                    if v in _inner and k not in _inner:
                        _inner.append(k); grow = True
            _t_inner = sum(THICK.get(k, THICK['default']) + CLEAR.get(k, CLEAR['default']) for k in _inner)

            _under_obs = [o for o in bpy.data.objects if o.type == 'MESH' and o.get('s3d_part') == 'garment' and o.get('s3d_kind') in ('trousers', 'leg')
                          and o.get('s3d_class') != cname]
            _under_obs = [o for o in _under_obs if any(z0 - 0.01 <= v.co.z <= z1 + 0.01 for v in list(o.data.vertices)[::7])]
            worn_over[cname] = [o.name for o in _under_obs]

            def over_built(z, t, _obs=_under_obs, _th=thick + clear):
                for tob in _obs:
                    band = [v.co for v in tob.data.vertices if abs(v.co.z - z) < 0.014]
                    if len(band) < 8:
                        continue
                    xs = [q.x for q in band]; ys = [q.y for q in band]
                    w_min = (max(xs) - min(xs)) + 2 * _th
                    d_min = (max(ys) - min(ys)) + 2 * _th
                    if t is None:
                        t = dict(w=w_min, d=d_min, x_c=0.5 * (max(xs) + min(xs)), y_mid=0.5 * (max(ys) + min(ys)))
                    else:
                        t = dict(t, w=max(t['w'], w_min), d=max(t['d'], d_min))
                return t

            def tgt(z, ring, _f=_tgt_trunk, _ids=_trunk_ids, _fo=_tgt_outer, _to=_t_outer, _ti=_t_inner):
                if id(ring) not in _ids:
                    zc = sum(body.co[i].z for i in ring) / len(ring)
                    xs = [body.co[i].x for i in ring]; ys = [body.co[i].y for i in ring]
                    side_ = 'L' if 0.5 * (max(xs) + min(xs)) > 0 else 'R'
                    st_a = [q for q in C['stations']['arm_' + side_]['stations'] if q['cls'] == cname and q.get('w_mm')]
                    if not st_a:
                        return None
                    q = min(st_a, key=lambda q_: abs(q_['z'] - zc))
                    if abs(q['z'] - zc) > 0.03:
                        return None
                    w = q['w_mm'] / 1000.0
                    # sleeve: tight at the root, full drawn width from 60 mm down the arm
                    arm_rings = arms[side_]
                    k_ = next((i_ for i_, r_ in enumerate(arm_rings) if r_ is ring), None)
                    s_ = 0.0
                    if k_ is not None:
                        for i_ in range(1, k_ + 1):
                            ca = sum((body.co[j] for j in arm_rings[i_ - 1]), Vector()) / len(arm_rings[i_ - 1])
                            cb = sum((body.co[j] for j in arm_rings[i_]), Vector()) / len(arm_rings[i_])
                            s_ += (cb - ca).length
                    t_ = max(0.0, min(1.0, (s_ - 0.015) / 0.045))
                    w_ring = (max(xs) - min(xs)) + 2 * (thick + clear)
                    w = w_ring + (w - w_ring) * t_
                    return dict(w=w, d=w, x_c=0.5 * (max(xs) + min(xs)), y_mid=0.5 * (max(ys) + min(ys)))
                t = _f(z, ring)
                if id(ring) in _ids:
                    t = over_built(z, t)
                # the neckline hugs the body, the drawn width takes hold over 40 mm below it
                if t is not None and kind != 'skirt':
                    f_top = min(1.0, max(0.0, (_z_top - z) / 0.04))
                    if f_top < 1.0:
                        xs = [body.co[i].x for i in ring]; ys = [body.co[i].y for i in ring]
                        wb = (max(xs) - min(xs)) + 2 * (thick + clear); db = (max(ys) - min(ys)) + 2 * (thick + clear)
                        t = dict(t, w=wb + (t['w'] - wb) * f_top, d=db + (t['d'] - db) * f_top,
                                 x_c=0.5 * (max(xs) + min(xs)) if t.get('x_c') is None else t['x_c'] * f_top + 0.5 * (max(xs) + min(xs)) * (1 - f_top),
                                 y_mid=0.5 * (max(ys) + min(ys)) if t.get('y_mid') is None else t['y_mid'] * f_top + 0.5 * (max(ys) + min(ys)) * (1 - f_top))
                if t is not None and _ti > 0:
                    xs = [body.co[i].x for i in ring]
                    ys = [body.co[i].y for i in ring]
                    w_min = (max(xs) - min(xs)) + 2 * (_ti + thick + clear)
                    d_min = (max(ys) - min(ys)) + 2 * (_ti + thick + clear)
                    if t['w'] < w_min:
                        t = dict(t, w=w_min, x_c=0.5 * (max(xs) + min(xs)))
                    if t['d'] < d_min:
                        t = dict(t, d=d_min, y_mid=0.5 * (max(ys) + min(ys)))
                if t is not None:
                    xs = [body.co[i].x for i in ring]
                    ys = [body.co[i].y for i in ring]
                    w_ring = (max(xs) - min(xs)) + 2 * (thick + clear)
                    d_ring = (max(ys) - min(ys)) + 2 * (thick + clear)
                    if t['w'] < w_ring:
                        t = dict(t, w=w_ring, x_c=0.5 * (max(xs) + min(xs)))
                    if t['d'] < d_ring:
                        t = dict(t, d=d_ring, y_mid=0.5 * (max(ys) + min(ys)))
                if _fo is not None:
                    o = _fo(z, ring)
                    if o is not None:
                        xs = [body.co[i].x for i in ring]
                        ys = [body.co[i].y for i in ring]
                        lim = dict(w=max(o['w'] - 2 * _to - 0.004, (max(xs) - min(xs)) + 2 * (thick + clear)),
                                   d=max(o['d'] - 2 * _to - 0.004, (max(ys) - min(ys)) + 2 * (thick + clear)), y_mid=o.get('y_mid'), x_c=o.get('x_c'))
                        if t is None:
                            return dict(w=lim['w'], d=lim['d'], y_mid=lim['y_mid'], x_c=lim['x_c'], cap=True)
                        t = dict(t, w=min(t['w'], lim['w']), d=min(t['d'], lim['d']), y_mid=lim['y_mid'] if t.get('y_mid') is None else t['y_mid'])
                        t['cap'] = True
                return t
            gap = g.get('open_front')
            if gap:
                rows = gap['gap_mm']
                gz = {}
                cal = C['calibration']
                for r, (a, b) in rows.items():
                    z = (cal['floor_row'] - (int(r) + 0.5)) / cal['ppm']
                    gz[z] = (a / 1000.0, b / 1000.0)
                gzs = sorted(gz)

                def gap_at(z):
                    if z < gzs[0] or z > gzs[-1]:
                        return None
                    for i in range(len(gzs) - 1):
                        if gzs[i] <= z <= gzs[i + 1]:
                            t = (z - gzs[i]) / max(1e-9, gzs[i + 1] - gzs[i])
                            a0, b0 = gz[gzs[i]]; a1, b1 = gz[gzs[i + 1]]
                            return (a0 + (a1 - a0) * t, b0 + (b1 - b0) * t)
                    return gz[gzs[-1]]

            # each sleeve is its own tube rooted at the arm hole and snapped onto the trunk shell
            hole_cs = []
            for ar in sleeve_lists:
                c_ = sum((body.co[i] for i in ar[0]), Vector()) / len(ar[0])
                r_ = max((body.co[i] - c_).length for i in ar[0])
                hole_cs.append((c_, r_))
            min(ring_z(body, r) for r in rings)
            ob = shell(body, [rings], Name, target=tgt, clearance=clear, thick=thick,
                       props=dict(s3d_part='garment', s3d_class=cname, s3d_kind=kind))
            _tb = bmesh.new()
            _tb.from_mesh(ob.data)
            _sl = _tb.verts.layers.int.get('s3d_src')
            trunk_pos = {v[_sl]: v.co.copy() for v in _tb.verts if _sl is not None and v[_sl] >= 0}
            _tb.free()
            sleeve_obs = []
            for ar in sleeve_lists:
                side_ = 'L' if sum(body.co[i].x for i in ar[-1]) > 0 else 'R'
                c_root = sum((body.co[i] for i in ar[0]), Vector()) / len(ar[0])
                max((body.co[i] - c_root).length for i in ar[0])
                c_next = sum((body.co[i] for i in ar[1]), Vector()) / len(ar[1]) if len(ar) > 1 else c_root
                a_dir = (c_next - c_root)
                a_dir = a_dir.normalized() if a_dir.length > 1e-6 else Vector((1.0 if side_ == 'L' else -1.0, 0, 0))
                sob = shell(body, [ar], '%s_Sleeve_%s' % (Name, side_), target=tgt, clearance=clear, thick=thick,
                            props=dict(s3d_part='garment', s3d_class=cname, s3d_kind='sleeve', s3d_side=side_, s3d_host=Name))
                sb_ = bmesh.new()
                sb_.from_mesh(sob.data)
                sl_ = sb_.verts.layers.int.get('s3d_src')
                root_ids = set(ar[0])
                n_in = 0
                for v in sb_.verts:
                    i_ = v[sl_] if sl_ is not None else -1
                    if i_ in root_ids and i_ in trunk_pos:
                        v.co = trunk_pos[i_]
                        n_in += 1
                # the second ring must stay ahead of the root ring plane or the first band folds under the arm
                rim = [v for v in sb_.verts if (v[sl_] if sl_ is not None else -1) in root_ids]
                if len(rim) >= 3:
                    c_rim = sum((v.co for v in rim), Vector()) / len(rim)
                    n_rim = Vector((0, 0, 0))
                    order = sorted(rim, key=lambda v: math.atan2(v.co.z - c_rim.z, (v.co.y - c_rim.y)))
                    for k_ in range(len(order)):
                        n_rim += (order[k_].co - c_rim).cross(order[(k_ + 1) % len(order)].co - c_rim)
                    if n_rim.length > 1e-9:
                        n_rim.normalize()
                        if n_rim.x * (1.0 if side_ == 'L' else -1.0) < 0:
                            n_rim = -n_rim
                        n_push = 0
                        for depth, ids_ in ((0.003, set(ar[1])), (0.006, set(ar[2]) if len(ar) > 2 else set())):
                            for v in sb_.verts:
                                i_ = v[sl_] if sl_ is not None else -1
                                if i_ in ids_:
                                    d_ = (v.co - c_rim).dot(n_rim)
                                    if d_ < depth:
                                        v.co = v.co + n_rim * (depth - d_)
                                        n_push += 1
                        if n_push:
                            print('[garments] %s: sleeve %s: %d vertices of its first rings brought ahead of the arm-hole rim' % (cname, side_, n_push))
                blend_to_rim(sb_, sl_, ar)
                sb_.to_mesh(sob.data)
                sb_.free()
                sob.data.update()
                print('[garments] %s: sleeve %s as its own tube from the arm hole (%d root vertices on the trunk shell)' % (cname, side_, n_in))
                mark_seam(sob, root_ids)
                mark_seam(ob, root_ids)
                sleeve_obs.append(sob)
            made.append(ob)
            if z_hem is not None:
                skirt_flare(ob, body, st, z_hem, thick, cname, trunk=trunk, arms=(arms if kind == 'cape' else None), drape=(kind == 'cape'), drape_off=thick + _t_inner + 0.003)
            if gap:
                # lapels only on jacket-like outer garments (lexicon rank 3)
                open_front_clip(ob, body, rings, sheet, cname, gzs, thick, classes=occluders_of(cname), lapel=0.0, over=over_of(cname))
                ob['s3d_clipped'] = 1
            if kind != 'cape':
                collar(ob, z1, thick, cname, top_ids=set(rings[0]))
            made.extend(sleeve_obs)
        elif kind == 'trousers':
            st_t = g.get('stations', [])
            tr = [r for r in trunk if z0 - 1e-6 <= ring_z(body, r) <= z1 + 1e-6]
            lists = [tr] if len(tr) >= 2 else []
            leg_zs = []
            for side in ('R', 'L'):
                st_s = g.get('sides', {}).get(side, [])
                lr = leg_rings_in(side, z0, z1)
                if len(lr) >= 2:
                    lists.append(lr)
                    leg_zs += [ring_z(body, r) for r in lr]
            junction_ids = set()
            if tr:
                junction_ids.add(id(min(tr, key=lambda r: ring_z(body, r))))
            for lst in lists[1:] if (tr and len(tr) >= 2) else lists:
                if lst:
                    junction_ids.add(id(lst[0]))
            sides_st = g.get('sides', {})

            # the hip ring and both leg root rings share one scale so the junction stays smooth
            hip_ratio = None
            hip_c = (0.0, 0.0)
            if tr:
                r_hip = min(tr, key=lambda r: ring_z(body, r))
                s_hip = st_at(st_t, ring_z(body, r_hip))
                xs_h = [body.co[i].x for i in r_hip]; ys_h = [body.co[i].y for i in r_hip]
                hip_c = (0.5 * (max(xs_h) + min(xs_h)), 0.5 * (max(ys_h) + min(ys_h)))
                if s_hip is not None and (max(xs_h) - min(xs_h)) > 1e-6:
                    hip_ratio = (max(1.0, s_hip['w_mm'] / 1000.0 / (max(xs_h) - min(xs_h))),
                                 max(1.0, s_hip['d_mm'] / 1000.0 / max(1e-6, (max(ys_h) - min(ys_h)))))

            def tgt_tr(z, ring, _tr=tr, _st=st_t, _sides=sides_st, _j=junction_ids, _hr=hip_ratio, _hc=hip_c):
                if id(ring) in _j:
                    if _hr is None:
                        return None
                    xs = [body.co[i].x for i in ring]; ys = [body.co[i].y for i in ring]
                    cx_ = 0.5 * (max(xs) + min(xs)); cy_ = 0.5 * (max(ys) + min(ys))
                    return dict(w=(max(xs) - min(xs)) * _hr[0], d=(max(ys) - min(ys)) * _hr[1],
                                y_mid=_hc[1] + (cy_ - _hc[1]) * _hr[1], x_c=_hc[0] + (cx_ - _hc[0]) * _hr[0])
                if _tr and any(ring is r for r in _tr):
                    s = st_at(_st, z)
                    if s is None:
                        return None
                    return dict(w=s['w_mm'] / 1000.0, d=s['d_mm'] / 1000.0, y_mid=s['y_mid_mm'] / 1000.0 if s.get('y_mid_mm') is not None else None, x_c=None)
                side = 'L' if sum(body.co[i].x for i in ring) > 0 else 'R'
                s = st_at(_sides.get(side, []), z)
                if s is None:
                    return None
                return dict(w=s['w_mm'] / 1000.0, d=s['d_mm'] / 1000.0, y_mid=s['y_mid_mm'] / 1000.0 if s.get('y_mid_mm') is not None else None,
                            x_c=s['x_c_mm'] / 1000.0 if s.get('x_c_mm') is not None else None)
            # the crotch bridge is not cloth: offset down it would cross the thighs
            z_hipring = min(ring_z(body, r) for r in tr) if tr else z_hip

            def keep_tr(cos, _zc=z_crotch, _zh=z_hipring):
                return True
            # pelvis shell over the trouser rings and both leg root loops, the legs are tubes snapped to its rim
            leg_lists = lists[1:] if (tr and len(tr) >= 2) else lists
            pelvis_lists = [(tr if (tr and len(tr) >= 2) else []) + [lr[0] for lr in leg_lists]]
            ob = shell(body, pelvis_lists, Name, target=tgt_tr, clearance=clear, thick=thick, keep_face=keep_tr,
                       props=dict(s3d_part='garment', s3d_class=cname, s3d_kind=kind))
            _tb = bmesh.new()
            _tb.from_mesh(ob.data)
            _sl = _tb.verts.layers.int.get('s3d_src')
            pelvis_pos = {v[_sl]: v.co.copy() for v in _tb.verts if _sl is not None and v[_sl] >= 0}
            _tb.free()
            made.append(ob)
            leg_obs = []
            for lr in leg_lists:
                side_ = 'L' if sum(body.co[i].x for i in lr[0]) > 0 else 'R'
                lob = shell(body, [lr], '%s_%s' % (Name, side_), target=tgt_tr, clearance=clear, thick=thick,
                            props=dict(s3d_part='garment', s3d_class=cname, s3d_kind='leg', s3d_side=side_, s3d_host=Name))
                lb_ = bmesh.new()
                lb_.from_mesh(lob.data)
                sl_ = lb_.verts.layers.int.get('s3d_src')
                root_ids = set(lr[0])
                n_in = 0
                for v in lb_.verts:
                    i_ = v[sl_] if sl_ is not None else -1
                    if i_ in root_ids and i_ in pelvis_pos:
                        v.co = pelvis_pos[i_]
                        n_in += 1
                rim = [v for v in lb_.verts if (v[sl_] if sl_ is not None else -1) in root_ids]
                if len(rim) >= 3 and len(lr) > 1:
                    c_rim = sum((v.co for v in rim), Vector()) / len(rim)
                    n_rim = Vector((0, 0, 0))
                    order = sorted(rim, key=lambda v: math.atan2(v.co.y - c_rim.y, v.co.x - c_rim.x))
                    for k_ in range(len(order)):
                        n_rim += (order[k_].co - c_rim).cross(order[(k_ + 1) % len(order)].co - c_rim)
                    if n_rim.length > 1e-9:
                        n_rim.normalize()
                        if n_rim.z > 0:
                            n_rim = -n_rim
                        n_push = 0
                        for depth, ids_ in ((0.003, set(lr[1])), (0.006, set(lr[2]) if len(lr) > 2 else set())):
                            for v in lb_.verts:
                                i_ = v[sl_] if sl_ is not None else -1
                                if i_ in ids_:
                                    d_ = (v.co - c_rim).dot(n_rim)
                                    if d_ < depth:
                                        v.co = v.co + n_rim * (depth - d_)
                                        n_push += 1
                        if n_push:
                            print('[garments] %s: leg %s: %d vertices of its first rings brought below the pelvis rim' % (cname, side_, n_push))
                blend_to_rim(lb_, sl_, lr)
                lb_.to_mesh(lob.data)
                lb_.free()
                lob.data.update()
                print('[garments] %s: leg %s as its own tube from the pelvis (%d root vertices on its rim)' % (cname, side_, n_in))
                mark_seam(lob, root_ids)
                mark_seam(ob, root_ids)
                leg_obs.append(lob)
            # the two legs never cross the plane between them
            if len(leg_obs) == 2:
                bms = []
                for lob in leg_obs:
                    b_ = bmesh.new()
                    b_.from_mesh(lob.data)
                    sl_ = b_.verts.layers.int.get('s3d_src')
                    by = {}
                    for v in b_.verts:
                        i_ = v[sl_] if sl_ is not None else -1
                        if i_ >= 0:
                            by.setdefault(i_, []).append(v)
                    bms.append((b_, by))
                n_clamp = 0
                A_, B_ = leg_lists
                shared = set(A_[0]) & set(B_[0])
                for ra, rb in zip(A_, B_):
                    va = [v for i in ra if i not in shared for v in bms[0][1].get(i, [])]
                    vb = [v for i in rb if i not in shared for v in bms[1][1].get(i, [])]
                    if not va or not vb:
                        continue
                    ca = sum((v.co for v in va), Vector()) / len(va)
                    cb = sum((v.co for v in vb), Vector()) / len(vb)
                    mid = 0.5 * (ca.x + cb.x)
                    if ra is A_[0] and shared:
                        sh = [v for i in shared for v in bms[0][1].get(i, [])]
                        if sh:
                            mid = sum(v.co.x for v in sh) / len(sh)
                    sa = 1.0 if ca.x > cb.x else -1.0
                    for vs, sgn in ((va, sa), (vb, -sa)):
                        for v in vs:
                            if (v.co.x - mid) * sgn < 0.0015:
                                v.co = Vector((mid + sgn * 0.0015, v.co.y, v.co.z))
                                n_clamp += 1
                for (b_, by), lob in zip(bms, leg_obs):
                    b_.to_mesh(lob.data)
                    b_.free()
                    lob.data.update()
                if n_clamp:
                    print("[garments] %s: %d inner-thigh vertices held on their own side of the legs' mid-plane" % (cname, n_clamp))
            made.extend(leg_obs)
        elif kind == 'arms' and g.get('painted') and not re.search(r'glove|mitt|gauntlet', cname) and g.get('tight_mm') is not None:
            print('[garments] %s: tight band - painted on the arm, no shell' % cname)
        elif kind == 'arms':
            # arm covering: a tube over the arm rings in the class's z-range, a glove copies the hand skin
            for side in ('R', 'L'):
                st_s = g.get('sides', {}).get(side, [])
                if not st_s:
                    continue
                ar = [r for r in arms[side] if z0 - 0.006 <= ring_z(body, r) <= z1 + 0.006]
                if len(ar) < 2:
                    # a narrow band spans fewer than two rings: take the two nearest
                    zc_ = 0.5 * (z0 + z1)
                    ar = sorted(sorted(arms[side], key=lambda r: abs(ring_z(body, r) - zc_))[:2], key=lambda r: -ring_z(body, r))
                if len(ar) < 2:
                    print('[garments] %s %s: fewer than two arm rings in z %.3f..%.3f' % (cname, side, z0, z1))
                    continue
                hand_extra = []

                _arm_obs = [o for o in bpy.data.objects if o.type == 'MESH' and o.get('s3d_part') == 'garment'
                            and o.get('s3d_kind') in ('arms', 'sleeve') and o.get('s3d_side') == side and o.get('s3d_class') != cname]
                worn_over['%s_%s' % (Name, side)] = [o.name for o in _arm_obs]
                _sx = 1.0 if side == 'L' else -1.0

                def tgt_arm(z, ring, _st=st_s, _obs=_arm_obs, _th=thick + clear):
                    xs = [body.co[i].x for i in ring]; ys = [body.co[i].y for i in ring]
                    q = min(_st, key=lambda q_: abs(q_['z'] - z))
                    t = None
                    if abs(q['z'] - z) <= 0.03:
                        w = q['w_mm'] / 1000.0
                        t = dict(w=w, d=w, x_c=0.5 * (max(xs) + min(xs)), y_mid=0.5 * (max(ys) + min(ys)))
                    for tob in _obs:
                        band = [v.co for v in tob.data.vertices if abs(v.co.z - z) < 0.014 and v.co.x * _sx > 0.0]
                        if len(band) < 6:
                            continue
                        bx = [q_.x for q_ in band]; by = [q_.y for q_ in band]
                        w_min = (max(bx) - min(bx)) + 2 * _th; d_min = (max(by) - min(by)) + 2 * _th
                        if t is None:
                            t = dict(w=w_min, d=d_min, x_c=0.5 * (max(bx) + min(bx)), y_mid=0.5 * (max(by) + min(by)))
                        else:
                            t = dict(t, w=max(t['w'], w_min), d=max(t['d'], d_min))
                    return t
                ob = shell(body, [ar] if len(ar) >= 2 else [], '%s_%s' % (Name, side), target=tgt_arm if len(ar) >= 2 else None,
                           clearance=clear, thick=thick, extra=hand_extra,
                           props=dict(s3d_part='garment', s3d_class=cname, s3d_kind=kind, s3d_side=side))
                ob['s3d_clipped'] = True
                print('[garments] %s %s: %d arm ring(s)%s' % (cname, side, len(ar), ' (the hand itself is painted with it)' if g.get('painted') else ''))
                made.append(ob)
        elif kind == 'legs':
            for side in ('R', 'L'):
                st_s = g.get('sides', {}).get(side, [])
                lr = leg_rings_in(side, z0, z1)
                if len(lr) < 2:
                    zs = [ring_z(body, r) for r in legs[side]]
                    idx = [i for i, z in enumerate(zs) if z0 - 0.02 <= z <= z1 + 0.02]
                    lr = [legs[side][i] for i in idx]
                if len(lr) < 2:
                    print('[garments] %s %s: fewer than two leg rings in z %.3f..%.3f' % (cname, side, z0, z1))
                    continue
                # a legs garment over the trousers takes their section plus its cloth
                trous_obs = [o for o in bpy.data.objects if o.type == 'MESH' and (o.get('s3d_kind') in ('trousers', 'leg')) and o.get('s3d_part') == 'garment']
                _tgt_own = target_from(st_s)
                sx = 1.0 if side == 'L' else -1.0

                def tgt_legs(z, ring, _own=_tgt_own, _obs=trous_obs, _sx=sx):
                    t = _own(z, ring)
                    for tob in _obs:
                        band = [v.co for v in tob.data.vertices if abs(v.co.z - z) < 0.008 and v.co.x * _sx > 0.0]
                        if len(band) < 6:
                            continue
                        xs = [q.x for q in band]; ys = [q.y for q in band]
                        w_min = (max(xs) - min(xs)) + 2 * (thick + clear)
                        d_min = (max(ys) - min(ys)) + 2 * (thick + clear)
                        x_c = 0.5 * (max(xs) + min(xs)); y_mid = 0.5 * (max(ys) + min(ys))
                        if t is None:
                            t = dict(w=w_min, d=d_min, y_mid=y_mid, x_c=x_c)
                        else:
                            t = dict(t, w=max(t['w'], w_min), d=max(t['d'], d_min),
                                     x_c=x_c if t.get('x_c') is None else t['x_c'], y_mid=y_mid if t.get('y_mid') is None else t['y_mid'])
                    return t
                ob = shell(body, [lr], '%s_%s' % (Name, side), target=tgt_legs, clearance=clear, thick=thick,
                           props=dict(s3d_part='garment', s3d_class=cname, s3d_kind=kind, s3d_side=side))
                made.append(ob)
    for ob in made:
        if ob.get('s3d_kind') in ('trousers', 'leg') or (ob.get('s3d_kind') in ('torso', 'skirt', 'cape') and not ob.get('s3d_clipped')):
            clip_to_views(ob, body.bvh, sheet, occluders_of(ob['s3d_class']), tol_px=3, views=('front', 'back'), log=print, pull_max=0.03, over=over_of(ob['s3d_class']), split_edges='smart',
                          close_mm=0.012, own=[ob['s3d_class']], gate_mm=0.015, smooth_px=int(round(0.008 * sheet.ppm)))
    for ob in made:
        remove_fins(ob, log=print)
        triangulate(ob, log=print)
        faces_clear_skin(ob, body, log=print)
        under = worn_over.get(ob.name) or worn_over.get(ob.get('s3d_class', ''), [])
        under_obs = [bpy.data.objects.get(n) for n in under if bpy.data.objects.get(n) is not None and bpy.data.objects.get(n) is not ob]
        if under_obs:
            clear_from_meshes(ob, under_obs, gap=0.0025, log=print)
            ob['s3d_over'] = ','.join(o.name for o in under_obs)
        fix_selfx(ob, log=print)
        remove_fins(ob, log=print)
        orient_outward(ob, body.bvh, log=print)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print('[garments] %d objects -> %s' % (len(made), out_blend))


def open_front_clip(ob, body, rings, sheet, cname, gzs, thick, lapel=0.028, classes=None, over=None):
    """Cut the front opening to the drawn mask and fold lapels along it."""
    hw = max(abs(body.co[i].x) for r in rings for i in r) + 0.02
    y_c = sum(body.co[i].y for i in rings[len(rings) // 2]) / len(rings[len(rings) // 2])
    z_lo, z_hi = gzs[0], gzs[-1]

    def region(f):
        c = f.calc_center_median()
        return abs(c.x) < hw
    # a vertex 8 mm outside the drawn cloth loses its faces: the opening is cut, not pulled shut
    clip_to_views(ob, body.bvh, sheet, classes or [cname], tol_px=2, views=('front', 'back'), region=region, log=print,
                  close_mm=0.012, own=[cname], gate_mm=0.015, smooth_px=int(round(0.008 * sheet.ppm)), pull_max=0.03, over=over, split_edges='smart')
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    cand = []
    for e in bm.edges:
        if not e.is_boundary:
            continue
        c = 0.5 * (e.verts[0].co + e.verts[1].co)
        if c.y < y_c and abs(c.x) < 0.5 * hw and z_lo - 0.01 <= c.z <= z_hi + 0.03:
            cand.append(e)
    # lapels only along slanted cut edges, a vertical cut gets none
    cand = [e for e in cand if abs(e.verts[0].co.x - e.verts[1].co.x) > 0.45 * abs(e.verts[0].co.z - e.verts[1].co.z) and (e.verts[0].co - e.verts[1].co).length > 0.012]
    made = {}
    if lapel <= 0:
        cand = []
    for e in cand:
        side = 1.0 if 0.5 * (e.verts[0].co.x + e.verts[1].co.x) > 0 else -1.0
        for v in e.verts:
            if v in made:
                continue
            t = (v.co.z - z_lo) / max(1e-6, z_hi - z_lo)
            depth = lapel * max(0.15, min(1.0, t)) ** 1.2
            if depth < 0.008:
                continue
            d = Vector((side * math.cos(math.radians(20)), -math.sin(math.radians(20)), 0.18)).normalized()
            made[v] = bm.verts.new(v.co + d * depth + Vector((0.0, -(thick + 0.002), 0.0)))
    n_l = 0
    for e in cand:
        a, b = e.verts
        if a not in made or b not in made:
            continue
        try:
            bm.faces.new([a, b, made[b], made[a]])
            n_l += 1
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    n_snap = snap_opening(bm, body, sheet, cname, y_c, hw, z_lo, z_hi)
    bm.to_mesh(me)
    bm.free()
    print('[garments] %s: open front clipped to the drawing, %d lapel faces, %d edge vertices moved to the drawn edge' % (cname, n_l, n_snap))


def snap_opening(bm, body, sheet, cname, y_c, hw, z_lo, z_hi, max_move=0.05, clearance=0.003):
    """Move vertices on vertical stretches of the cut to the drawn inner edge, off the skin."""
    mask = sheet.mask('front', [cname], tol_px=0)
    mid = sheet.pan['front']['midline']
    ppm = sheet.ppm
    moved = 0
    for v in bm.verts:
        if not v.is_boundary or v.co.y >= y_c or abs(v.co.x) >= 0.5 * hw or not (z_lo - 0.01 <= v.co.z <= z_hi):
            continue
        runs = [e for e in v.link_edges if e.is_boundary]
        if not any(abs(e.verts[0].co.z - e.verts[1].co.z) > abs(e.verts[0].co.x - e.verts[1].co.x) for e in runs):
            continue
        row, _ = sheet.project('front', v.co)
        if not 0 <= row < mask.shape[0]:
            continue
        side = 1 if v.co.x > 0 else -1
        cols = np.arange(int(mid), int(mid + side * max_move * 2 * ppm), side)
        cols = cols[(cols >= 0) & (cols < mask.shape[1])]
        hit = np.nonzero(mask[row, cols])[0]
        if not len(hit):
            continue
        x_t = (cols[hit[0]] - mid) / ppm
        if abs(x_t - v.co.x) < 0.001 or abs(x_t - v.co.x) > max_move:
            continue
        co = Vector((x_t, v.co.y, v.co.z))
        lifted = body.lift_out(co, clearance)
        v.co = lifted if lifted is not None else co
        moved += 1
    return moved


def open_front(ob, body, rings, gap_at, gzs, thick, cname, lapel=0.028):
    """Cut the front open along the drawn gap with two vertical planes and fold lapels."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    y_c = sum(body.co[i].y for i in rings[len(rings) // 2]) / len(rings[len(rings) // 2])
    z_lo, z_hi = gzs[0], gzs[-1]
    gl = sum(gap_at(z)[0] for z in gzs) / len(gzs)
    gr = sum(gap_at(z)[1] for z in gzs) / len(gzs)
    front = [f for f in bm.faces if sum((v.co.y for v in f.verts), 0.0) / len(f.verts) < y_c]
    geom = list(set(sum([list(f.verts) + list(f.edges) + [f] for f in front], [])))
    for x in (gl, gr):
        res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(x, 0, 0), plane_no=(1, 0, 0), dist=1e-5)
        geom = [g for g in res['geom'] if g.is_valid] + [g for g in geom if g.is_valid and g not in res['geom']]
        geom = list(set(geom))
    for z, no in ((z_lo, (0, 0, 1)), (z_hi, (0, 0, 1))):
        res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z), plane_no=no, dist=1e-5)
        geom = list(set([g for g in res['geom'] if g.is_valid] + [g for g in geom if g.is_valid]))
    dead = []
    for f in bm.faces:
        c = sum((v.co for v in f.verts), Vector()) / len(f.verts)
        if c.y < y_c and gl < c.x < gr and z_lo <= c.z <= z_hi:
            dead.append(f)
    bmesh.ops.delete(bm, geom=dead, context='FACES')
    cand = []
    for e in bm.edges:
        if not e.is_boundary:
            continue
        c = 0.5 * (e.verts[0].co + e.verts[1].co)
        if c.y < y_c and (abs(c.x - gl) < 0.004 or abs(c.x - gr) < 0.004) and z_lo - 0.002 <= c.z <= z_hi + 0.002:
            cand.append(e)
    made = {}
    for e in cand:
        side = 1.0 if 0.5 * (e.verts[0].co.x + e.verts[1].co.x) > 0 else -1.0
        for v in e.verts:
            if v in made:
                continue
            # the lapel is a triangle: nothing at the bottom, full width at the top
            t = (v.co.z - z_lo) / max(1e-6, z_hi - z_lo)
            depth = lapel * max(0.15, min(1.0, t)) ** 1.2
            if depth < 0.008:
                continue
            d = Vector((side * math.cos(math.radians(20)), -math.sin(math.radians(20)), 0.18)).normalized()
            made[v] = bm.verts.new(v.co + d * depth + Vector((0.0, -(thick + 0.002), 0.0)))
    for e in cand:
        a, b = e.verts
        if a not in made or b not in made:
            continue
        try:
            bm.faces.new([a, b, made[b], made[a]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    print('[garments] %s: open front %.0f..%.0f mm, z %.3f..%.3f, %d faces removed, %d lapel edges' % (cname, gl * 1000, gr * 1000, z_lo, z_hi, len(dead), len(cand)))


def flat_collar(ob, body, rings, thick, cname, width=0.03, log=print):
    """Extend the neckline outward and down into a collar lying on the shoulders."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    top_ring = rings[0]
    z_top = max(body.co[i].z for i in top_ring)
    c = sum((body.co[i] for i in top_ring), Vector()) / len(top_ring)
    cand = [e for e in bm.edges if e.is_boundary and 0.5 * (e.verts[0].co.z + e.verts[1].co.z) > z_top - 0.02]
    if not cand:
        bm.free()
        return
    made = {}
    for e in cand:
        for v in e.verts:
            if v in made:
                continue
            out = Vector((v.co.x - c.x, v.co.y - c.y, 0.0))
            if out.length < 1e-6:
                out = Vector((0, -1, 0))
            out.normalize()
            w = width * (0.7 + 0.5 * max(0.0, out.y))
            made[v] = bm.verts.new(v.co + out * w + Vector((0, 0, -0.35 * w)))
    rim = bm.faces.layers.int.get('s3d_rim') or bm.faces.layers.int.new('s3d_rim')
    n = 0
    for e in cand:
        a, b = e.verts
        try:
            f = bm.faces.new([a, b, made[b], made[a]])
            f[rim] = 0
            n += 1
        except ValueError:
            pass
    outer_edges = [e for e in bm.edges if e.is_boundary and all(v in made.values() for v in e.verts)]
    made2 = {}
    for e in outer_edges:
        for v in e.verts:
            if v not in made2:
                made2[v] = bm.verts.new(v.co + Vector((0, 0, -thick)))
    for e in outer_edges:
        a, b = e.verts
        try:
            f = bm.faces.new([a, b, made2[b], made2[a]])
            f[rim] = 1
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    log('[garments] %s: flat collar, %d faces' % (cname, n))


def cap_arm_holes(ob, loops, body):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    sl = bm.verts.layers.int.get('s3d_src')
    if sl is None:
        bm.free()
        return 0
    n = 0
    for loop in loops:
        ids = set(loop)
        vs = [v for v in bm.verts if v[sl] in ids]
        if len(vs) < 3:
            continue
        c = sum((v.co for v in vs), Vector()) / len(vs)
        vset = set(vs)
        nxt = {}
        for v in vs:
            for e in v.link_edges:
                if e.is_boundary and e.other_vert(v) in vset:
                    nxt.setdefault(v, []).append(e.other_vert(v))
        order, seen_v = [vs[0]], {vs[0]}
        while True:
            cands_ = [u for u in nxt.get(order[-1], []) if u not in seen_v]
            if not cands_:
                break
            order.append(cands_[0]); seen_v.add(cands_[0])
        if len(order) < 3:
            print('[garments]   arm hole: %d loop verts, boundary walk found %d (%d boundary-linked)' % (len(vs), len(order), len(nxt)))
            continue
        vs = order
        nrm = sum((v.normal for v in vs), Vector())
        if nrm.length < 1e-6:
            nrm = Vector((1.0 if c.x > 0 else -1.0, 0, 0))
        nrm.normalize()
        cv = bm.verts.new(c - nrm * 0.004)
        cv[sl] = -1
        for k in range(len(vs)):
            try:
                bm.faces.new([vs[k], vs[(k + 1) % len(vs)], cv])
            except ValueError:
                pass
        n += 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return n


def mark_seam(ob, src_ids):
    """Flag vertices born from body ids `src_ids` as a seam that later passes must not move."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    sl = bm.verts.layers.int.get('s3d_src')
    sm = bm.verts.layers.int.get('s3d_seam') or bm.verts.layers.int.new('s3d_seam')
    n = 0
    for v in bm.verts:
        if sl is not None and v[sl] in src_ids:
            v[sm] = 1
            n += 1
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return n


def blend_to_rim(bm, sl, ring_lists, fracs=(0.7, 0.4, 0.15)):
    """Blend the rings just past a tube's root toward the root profile by `fracs`."""
    by = {}
    for v in bm.verts:
        i_ = v[sl] if sl is not None else -1
        if i_ >= 0:
            by[i_] = v
    root = [by.get(i) for i in ring_lists[0]]
    if any(v is None for v in root):
        return 0
    c0 = sum((v.co for v in root), Vector()) / len(root)
    n = 0
    for k, f in enumerate(fracs, 1):
        if k >= len(ring_lists) or len(ring_lists[k]) != len(root):
            break
        ring = [by.get(i) for i in ring_lists[k]]
        if any(v is None for v in ring):
            break
        c1 = sum((v.co for v in ring), Vector()) / len(ring)
        ax = (c1 - c0)
        if ax.length < 1e-9:
            break
        ax.normalize()
        for vr, v in zip(root, ring):
            d_r = (vr.co - c0) - ax * (vr.co - c0).dot(ax)
            d_v = (v.co - c1) - ax * (v.co - c1).dot(ax)
            v.co = v.co + (d_r - d_v) * f
            n += 1
    return n


def skirt_flare(ob, body, stations, z_hem, thick, cname, log=print, trunk=(), arms=None, drape=False, drape_off=0.006):
    """Hang a bell of rings from the shell's bottom loop down to the hem."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    b_edges = [e for e in bm.edges if e.is_boundary]
    adj = {}
    for e in b_edges:
        for v in e.verts:
            adj.setdefault(v, set()).add(e)
    seen_e = set()
    cycles = []
    for e0 in b_edges:
        if e0 in seen_e:
            continue
        cyc_v, stack = set(), [e0]
        while stack:
            e = stack.pop()
            if e in seen_e:
                continue
            seen_e.add(e)
            for v in e.verts:
                cyc_v.add(v)
                stack.extend(adj[v] - seen_e)
        cycles.append(list(cyc_v))
    cycles = [cv for cv in cycles if len(cv) >= 6 and (max(v.co.x for v in cv) - min(v.co.x for v in cv)) > 0.03]
    if not cycles:
        bm.free()
        log('[garments] %s: no bottom loop for the skirt' % cname)
        return
    loop_v = min(cycles, key=lambda cv: sum(v.co.z for v in cv) / len(cv))
    zmin = sum(v.co.z for v in loop_v) / len(loop_v)
    c = sum((v.co for v in loop_v), Vector()) / len(loop_v)
    loop_v.sort(key=lambda v: math.atan2(v.co.y - c.y, v.co.x - c.x))
    xs = [v.co.x for v in loop_v]; ys = [v.co.y for v in loop_v]
    w0, d0 = max(xs) - min(xs), max(ys) - min(ys)
    # a station under half the width above is an edge row, not the hem
    st = sorted([dict(q) for q in stations if q['z'] < zmin - 0.006 and q['z'] >= z_hem - 0.004 and q.get('w_mm') and q.get('d_mm')], key=lambda q: -q['z'])
    keep = []
    for q in st:
        prev_w = keep[-1]['w_mm'] if keep else w0 * 1000
        if q['w_mm'] < 0.5 * prev_w:
            continue
        keep.append(q)
    if not keep:
        bm.free()
        log('[garments] %s: no skirt stations below z %.3f' % (cname, zmin))
        return
    if keep[-1]['z'] > z_hem + 0.006:
        keep.append(dict(keep[-1], z=z_hem + 0.002))
    # a dip between wider stations is an occluder: take the neighbours' mean
    for i in range(1, len(keep) - 1):
        for k in ('w_mm', 'd_mm'):
            a_, b_ = keep[i - 1][k], keep[i + 1][k]
            if keep[i][k] < 0.9 * min(a_, b_):
                keep[i][k] = 0.5 * (a_ + b_)
    body_w = {}
    for r in list(trunk) + ([r for sd_ in (arms or {}) for r in arms[sd_]] if arms else []):
        cs = [body.co[i] for i in r]
        zc_ = sum(c.z for c in cs) / len(cs)
        body_w.setdefault(round(zc_, 3), [0.0, 0.0, 0.0])
        e = body_w[round(zc_, 3)]
        e[0] = max(e[0], max(abs(c.x) for c in cs) * 2); e[1] = max(e[1], max(c.y for c in cs)); e[2] = min(e[2], min(c.y for c in cs))
    zs_b = sorted(body_w)
    for q in keep:
        near = [z_ for z_ in zs_b if abs(z_ - q['z']) < 0.02]
        if not near:
            continue
        wmin = max(body_w[z_][0] for z_ in near) + 2 * thick + 0.004
        dmin = max(body_w[z_][1] - body_w[z_][2] for z_ in near) + 2 * thick + 0.004
        if q['w_mm'] < wmin * 1000:
            q['w_mm'] = wmin * 1000
        if q['d_mm'] < dmin * 1000:
            q['d_mm'] = dmin * 1000
            q['y_mid_mm'] = 0.5 * (max(body_w[z_][1] for z_ in near) + min(body_w[z_][2] for z_ in near)) * 1000
    rim = bm.faces.layers.int.get('s3d_rim') or bm.faces.layers.int.new('s3d_rim')
    rings = []
    for q in keep:
        w, d = max(q['w_mm'] / 1000.0, w0), max(q['d_mm'] / 1000.0, d0 * 0.9)
        cx = c.x + ((q.get('x_c_mm') or 0.0) / 1000.0 - c.x) * 0.5
        cy = (q.get('y_mid_mm') or c.y * 1000) / 1000.0
        ring = []
        # scale the loop, not the previous ring, so scales do not compound
        for v in loop_v:
            u = (v.co.x - c.x) / max(1e-6, 0.5 * w0)
            vv = (v.co.y - c.y) / max(1e-6, 0.5 * d0)
            ring.append(bm.verts.new((cx + u * 0.5 * w, cy + vv * 0.5 * d, q['z'])))
        rings.append(ring)
    chain = [loop_v] + rings
    n = len(loop_v)
    for A, B in zip(chain, chain[1:]):
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new([A[i], A[j], B[j], B[i]])
            except ValueError:
                pass
    hem = rings[-1]
    inner = []
    if not drape:
        inner = [bm.verts.new(v.co + (Vector((c.x, c.y, v.co.z)) - v.co).normalized() * thick) for v in hem]
        for i in range(n):
            j = (i + 1) % n
            try:
                f = bm.faces.new([hem[i], hem[j], inner[j], inner[i]])
                f[rim] = 1
            except ValueError:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # a cape drapes: vertices within 60 mm above a surface drop onto it
    n_drape = 0
    below = None
    if drape:
        sb = bmesh.new()
        for o in bpy.data.objects:
            if o.type != 'MESH' or o == ob or o.get('s3d_part') in ('hair',):
                continue
            t = bmesh.new(); t.from_mesh(o.data); t.transform(o.matrix_world)
            for f in t.faces:
                try:
                    sb.faces.new([sb.verts.new(v.co) for v in f.verts])
                except ValueError:
                    pass
            t.free()
        below = BVHTree.FromBMesh(sb) if len(sb.faces) else None
        sb.free()
        if below is not None:
            for ring in rings:
                for v in ring:
                    hit = below.ray_cast(Vector((v.co.x, v.co.y, v.co.z + 0.002)), Vector((0, 0, -1)), 0.062)
                    if hit[0] is not None and hit[1].z > -0.2:
                        z_new = hit[0].z + drape_off
                        if z_new < v.co.z - 0.001:
                            v.co.z = z_new
                            n_drape += 1
                    elif hit[0] is not None:
                        up = below.ray_cast(v.co, Vector((0, 0, 1)), 0.12)
                        if up[0] is not None:
                            v.co.z = up[0].z + drape_off
                            n_drape += 1
    # cloth is taut: a vertex in a hollow rises to its neighbours' mean height
    n_taut = 0
    if drape and below is not None:
        for _ in range(4):
            new_z = {}
            for ring in rings:
                for v in ring:
                    nb = [e.other_vert(v) for e in v.link_edges]
                    if len(nb) < 3:
                        continue
                    mz = sum(u.co.z for u in nb) / len(nb)
                    if v.co.z < mz - 0.0015:
                        new_z[v] = mz - 0.0015
            for v, z_ in new_z.items():
                v.co.z = z_
                n_taut += 1
    n_lift = 0
    for ring in rings:
        for v in ring:
            co = body.lift_out(v.co.copy(), 0.006)
            if co is not None:
                v.co = co
                n_lift += 1
    n_out = 0
    if drape and below is not None:
        chain = [loop_v] + rings
        for k_ in range(n):
            for i_ in range(1, len(chain)):
                up_, lo_ = chain[i_ - 1][k_], chain[i_][k_]
                if lo_.co.z > up_.co.z - 0.004:
                    lo_.co.z = up_.co.z - 0.004
        for _pass in range(2):
            for ring in rings:
                for v in ring:
                    near = below.find_nearest(v.co, 0.06)
                    if near[0] is None:
                        continue
                    n_ = near[1]
                    if n_.length < 1e-6:
                        continue
                    off = (v.co - near[0]).dot(n_)
                    if off < drape_off - 0.001:
                        v.co = near[0] + n_ * drape_off
                        n_out += 1

    bm.to_mesh(me)
    bm.free()
    log('[garments] %s: skirt bell of %d ring(s) from z %.3f to the hem %.3f%s%s%s%s' % (cname, len(rings), zmin, keep[-1]['z'], (', %d verts lifted off the legs' % n_lift) if n_lift else '', (', %d verts draped onto the surface below' % n_drape) if n_drape else '', (', %d raised taut over hollows' % n_taut) if n_taut else '', (', %d brought out of the layers under it' % n_out) if n_out else ''))
    log('[garments]   loop w %.0f d %.0f mm at (%.3f, %.3f); rings: %s' % (w0 * 1000, d0 * 1000, c.x, c.y, ', '.join('z %.3f w %.0f d %.0f' % (q['z'], q['w_mm'], q['d_mm']) for q in keep)))


def collar(ob, z_top_drawn, thick, cname, log=print, top_ids=None):
    """Extrude the shell's top ring up to the drawn collar top, flaring 12 deg."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    sl = bm.verts.layers.int.get('s3d_src')
    top_edges = []
    zmax = max(v.co.z for v in bm.verts)
    cand_z = sorted(v.co.z for e in bm.edges if e.is_boundary and min(v.co.z for v in e.verts) > zmax - 0.03 for v in e.verts)
    z_ring = cand_z[len(cand_z) // 2] if cand_z else zmax
    for e in bm.edges:
        if not e.is_boundary:
            continue
        if top_ids is not None and sl is not None:
            if all(v[sl] in top_ids for v in e.verts):
                top_edges.append(e)
        elif all(abs(v.co.z - z_ring) < 0.004 for v in e.verts):
            top_edges.append(e)
    def touches_cut(e):
        for v in e.verts:
            for e2 in v.link_edges:
                if e2 is not e and e2.is_boundary and e2 not in top_edges:
                    return True
        return False
    if len(top_edges) >= 4:
        top_edges = [e for e in top_edges if not touches_cut(e)]
    h = z_top_drawn - z_ring
    if not top_edges or h < 0.012:
        bm.free()
        return 0
    h = min(h, 0.060)  # a biker collar stands 40-60 mm
    verts = set(v for e in top_edges for v in e.verts)
    cx = sum((v.co.x for v in verts), 0.0) / len(verts)
    cy = sum((v.co.y for v in verts), 0.0) / len(verts)
    up = {}
    for v in verts:
        r = Vector((v.co.x - cx, v.co.y - cy, 0.0))
        r = r.normalized() if r.length > 1e-6 else Vector((0, -1, 0))
        up[v] = bm.verts.new(v.co + Vector((0, 0, h)) + r * (h * math.tan(math.radians(12))))
    for e in top_edges:
        a, b = e.verts
        try:
            bm.faces.new([a, b, up[b], up[a]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    log('[garments] %s: stand collar %.0f mm on %d edges' % (cname, h * 1000, len(top_edges)))
    return len(top_edges)


def lapels(ob, body, rings, gap_at, thick, cname, depth=0.028):
    """Fold flaps 55 deg outward along the open front's cut edges."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    cy_front = min(body.co[i].y for i in rings[0])
    cand = []
    for e in bm.edges:
        if not e.is_boundary:
            continue
        c = 0.5 * (e.verts[0].co + e.verts[1].co)
        ga = gap_at(c.z)
        if ga is None:
            continue
        if c.y < cy_front + 0.06 and (abs(c.x - ga[0]) < 0.02 or abs(c.x - ga[1]) < 0.02):
            cand.append(e)
    if not cand:
        bm.free()
        return
    made = {}
    for e in cand:
        side = 1.0 if 0.5 * (e.verts[0].co.x + e.verts[1].co.x) > 0 else -1.0
        for v in e.verts:
            if v in made:
                continue
            d = Vector((side * math.cos(math.radians(55)), -math.sin(math.radians(55)), 0.0))
            made[v] = bm.verts.new(v.co + d * depth)
    for e in cand:
        a, b = e.verts
        if a not in made or b not in made:
            continue
        try:
            bm.faces.new([a, b, made[b], made[a]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    print('[garments] %s: lapels from %d cut edges' % (cname, len(cand)))


if __name__ == '__main__':
    main()
