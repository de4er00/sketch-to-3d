"""Build the body as one surface (head, trunk, legs, feet, arms, hands) from char.json.

    blender -b --factory-startup --python sketch3d/build/body.py -- <char.json> <out.blend> [--report out.json]

Metres, z up, the character faces -Y with its left at +X. Ring column 0 is the front centre.
"""
import bpy, bmesh, os, sys, json, math
from mathutils import Vector, Matrix
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from limb import extrude_loop, open_patch   # noqa: E402
import hand as HP                                         # noqa: E402

RING_MAX_GAP = 0.045
RING_MIN_GAP = 0.010
LANDMARK_TAGS = ('shoulder', 'armpit', 'waist', 'hip', 'crotch', 'neck', 'armhole_top', 'armhole_mid', 'armhole_bot', 'shoulder_line_lowered')
ARM_GAP = 0.010
NC = 16
N_IN = NC // 2 - 1
TRUNK_P = 0.76

# the normalised dome: (h, y_front_offset, radii), h from chin to crown
SKULL_T = [
    (0.95, 0.014, [0.216, 0.214, 0.206, 0.205, 0.227, 0.221, 0.216, 0.221, 0.227, 0.205, 0.206, 0.214]),
    (0.86, 0.014, [0.324, 0.323, 0.313, 0.298, 0.337, 0.337, 0.324, 0.337, 0.337, 0.298, 0.313, 0.323]),
    (0.74, 0.012, [0.384, 0.390, 0.366, 0.349, 0.387, 0.390, 0.384, 0.390, 0.387, 0.349, 0.366, 0.390]),
    (0.62, 0.010, [0.407, 0.407, 0.381, 0.359, 0.399, 0.406, 0.407, 0.406, 0.399, 0.359, 0.381, 0.407]),
    (0.50, 0.007, [0.395, 0.395, 0.376, 0.346, 0.382, 0.391, 0.395, 0.391, 0.382, 0.346, 0.376, 0.395]),
]
_TW, _TF, _TB = 0.814, 0.450, 0.397
NOSE_FALLOFF = 3.0


def sgnpow(u, p):
    return math.copysign(abs(u) ** p, u)


def resample(vals, n):
    m = len(vals)
    out = []
    for k in range(n):
        u = k * m / float(n)
        i = int(u)
        t = u - i
        out.append(vals[i % m] * (1 - t) + vals[(i + 1) % m] * t)
    return out


def angles(nc):
    """Leg ring angles in crotch-split order: 0 = outboard, 90 = front."""
    half = nc // 2
    out = [(90.0 - 180.0 * k / half) % 360.0 for k in range(half + 1)]
    inn = [(270.0 - 180.0 * k / (N_IN + 1)) % 360.0 for k in range(1, N_IN + 1)]
    return out + inn


ANG = angles(NC)
NL = len(ANG)


def shoulder_win(a, centre):
    d = ((a - centre + 180.0) % 360.0) - 180.0
    if abs(d) > 90.0:
        return 0.0
    return math.sin(2.0 * math.radians(d)) ** 2


class Body:
    def __init__(self, char, log=print):
        self.C = char
        self.log = log
        self.bm = bmesh.new()
        # layers must exist before any vertex: adding one later invalidates the BMVert references
        self.bm.verts.layers.int.new('s3d_crease')
        self.map = dict(trunk=[], head=[], leg={}, foot={}, arm={}, hand={}, z={}, landmarks={})
        lm = char['landmarks']
        self.z = {k: v['z'] for k, v in lm.items() if isinstance(v, dict) and 'z' in v}

    def ring(self, verts_co):
        return [self.bm.verts.new(Vector(c)) for c in verts_co]

    def bridge(self, a, b):
        n = len(a)
        assert len(b) == n, 'bridge: %d vs %d' % (len(a), len(b))
        for k in range(n):
            j = (k + 1) % n
            self.bm.faces.new([a[k], a[j], b[j], b[k]])

    def trunk_ring_co(self, z, hw, dp, cy, cx=0.0, p=TRUNK_P):
        out = []
        for k in range(NC):
            th = 2 * math.pi * k / NC
            out.append((cx + hw * sgnpow(math.sin(th), p), cy - dp * sgnpow(math.cos(th), p), z))
        return out

    crease_cos = []

    def build_head(self):
        C, log = self.C, self.log
        sk = C['skull']
        chin_z = sk['chin_z']
        # skull: width from the temples, depth 0.97 w, height from the brow and chin
        fp = C['face_profile']
        face_st = [s for s in C['stations']['face'] if s['w_mm'] and s['d_mm']]
        brow_z = fp.get('brow', {}).get('z', chin_z + 0.7 * (sk['cap_top_z'] - chin_z))
        dome_w = sk['w_mm'] / 1000.0
        dome_d = 0.97 * dome_w
        SK_H = max(1.97 * (brow_z - chin_z), sk['hairline_z'] - chin_z + 0.030)
        SK_H = min(SK_H, sk['cap_top_z'] - chin_z - 0.010)
        top_z = chin_z + SK_H
        SX, SY, SZ = dome_w / _TW, dome_d / (_TF + _TB), SK_H
        # the dome's lowest ring meets the drawn profile so the forehead has no step
        h_low, yf_low, rr_low = SKULL_T[-1]
        z_low = chin_z + h_low * SZ
        used = [q for q in face_st if q['z'] <= z_low - RING_MIN_GAP and q['d_mm'] >= 0.3 * q['w_mm']]
        sb = max(used, key=lambda q: q['z']) if used else (min(face_st, key=lambda q: abs(q['z'] - z_low)) if face_st else None)
        y_brow_front = (sb['y_mid_mm'] - 0.5 * sb['d_mm']) / 1000.0 if sb else -0.5 * dome_d
        front_low = -(rr_low[0] + yf_low) * SY
        y_shift = y_brow_front + 0.004 - front_low
        dome_yc = y_shift + (-_TF + _TB) * 0.5 * SY
        log('[body] skull w %.3f d %.3f (CONV 0.97 w) h %.3f (chin %.3f, top %.3f), y centre %.3f (front at z %.3f = %.3f, the profile)' % (dome_w, dome_d, SK_H, chin_z, top_z, dome_yc, z_low, y_brow_front))
        rings = []
        for (h, yf, rr) in SKULL_T:
            rr16 = resample(rr, NC)
            co = []
            for k in range(NC):
                th = 2 * math.pi * k / NC
                co.append((rr16[k] * math.sin(th) * SX, -(rr16[k] * math.cos(th) + yf) * SY + y_shift, chin_z + h * SZ))
            rings.append(self.ring(co))
        face_st = [s for s in face_st if s['z'] <= z_low - RING_MIN_GAP]
        ws = [s['w_mm'] for s in face_st]
        k = 2
        env = [min(ws[max(0, i - k):i + k + 1]) for i in range(len(ws))]
        env = [max(env[max(0, i - k):i + k + 1]) for i in range(len(env))]
        brow = fp.get('brow', {}).get('z')
        lip = fp.get('lip', {}).get('z')
        fp.get('nose_tip', {}).get('z')

        def yfront(s):
            return (s['y_mid_mm'] - 0.5 * s['d_mm']) / 1000.0

        def yback(s):
            return (s['y_mid_mm'] + 0.5 * s['d_mm']) / 1000.0
        y_brow = yfront(min(face_st, key=lambda s: abs(s['z'] - brow))) if brow else yfront(face_st[0])
        y_lip = yfront(min(face_st, key=lambda s: abs(s['z'] - lip))) if lip else yfront(face_st[-1])
        z_brow = brow if brow else face_st[0]['z']
        z_lip = lip if lip else face_st[-1]['z']
        face_rows = []
        for i, s in enumerate(face_st):
            z = s['z']
            hw = 0.5 * env[i] / 1000.0
            yc = yfront(s)
            if z_lip <= z <= z_brow and z_brow > z_lip:
                t = (z_brow - z) / (z_brow - z_lip)
                ys = y_brow + (y_lip - y_brow) * t
                ys = max(ys, yc)
            else:
                ys = yc
            # the back of the head curves forward from full depth at the temple to 0.15 at the chin
            u = (face_st[0]['z'] - z) / max(1e-6, face_st[0]['z'] - face_st[-1]['z'])
            g = 0.15 + 0.85 * math.cos(u * math.pi / 2) ** 1.2
            yb = dome_yc + 0.5 * dome_d * g
            p = 0.82 + 0.08 * u
            face_rows.append((z, hw, yb, yc, ys, p))
        for (z, hw, yb, yc, ys, p) in face_rows:
            mid, dep = 0.5 * (ys + yb), 0.5 * (yb - ys)
            co = []
            for k2 in range(NC):
                th = 2 * math.pi * k2 / NC
                s_, c_ = math.sin(th), math.cos(th)
                nz = (yc - ys) * max(0.0, c_) ** NOSE_FALLOFF
                co.append((hw * sgnpow(s_, p), mid - dep * sgnpow(c_, p) + nz, z))
            rings.append(self.ring(co))
        for a, b in zip(rings, rings[1:]):
            self.bridge(a, b)
        vcrown = self.bm.verts.new((0.0, y_shift - 0.018 * SY, chin_z + SZ))
        for k2 in range(NC):
            self.bm.faces.new([vcrown, rings[0][k2], rings[0][(k2 + 1) % NC]])
        self.map['head'] = rings
        self.map['z']['chin'] = chin_z
        self.map['z']['dome_top'] = chin_z + SZ
        self.dome = dict(w=dome_w, d=dome_d, h=SK_H, y=dome_yc, SX=SX, SY=SY, SZ=SZ, y_shift=y_shift)
        log('[body] head: %d dome rings + %d face rings' % (len(SKULL_T), len(face_rows)))
        return rings

    def build_trunk(self, head_rings):
        C, log = self.C, self.log
        st = [s for s in C['stations']['trunk'] if s.get('w_body_mm') and s.get('d_body_mm')]
        # the trapezius: three S-curve rings between the neck and the shoulder line
        neck = [s for s in st if 'neck' in s['tags']]
        line = [s for s in st if s.get('w_src', '').startswith('outline') and s['w_body_mm'] >= 2 * (neck[0]['w_body_mm'] if neck else 0)]
        if neck and line:
            n0, l0 = neck[0], line[0]
            # a shoulder line right at the neck base: lower the acromion so the slope is 45 mm
            if n0['z'] - l0['z'] < 0.045:
                z_low = n0['z'] - 0.045
                cand = [s for s in st if z_low - 0.006 <= s['z'] < n0['z'] and s.get('w_src', '').startswith('outline')] + [l0]
                widest = max(cand, key=lambda s: s['w_body_mm'])
                l0 = dict(widest, z=z_low, tags=list(l0['tags']) + ['shoulder_line_lowered'])
                st = [s for s in st if s['z'] < l0['z'] - 0.006 or s['z'] >= n0['z']] + [l0]
            st = [s for s in st if not (n0['z'] > s['z'] > l0['z'])]
            extra = []
            for f in (0.2, 0.4, 0.6, 0.8):
                rw = (l0['w_body_mm'] / n0['w_body_mm']) ** f
                rd = (l0['d_body_mm'] / n0['d_body_mm']) ** f
                extra.append(dict(z=n0['z'] + (l0['z'] - n0['z']) * f, w_body_mm=n0['w_body_mm'] * rw, d_body_mm=n0['d_body_mm'] * rd,
                                  y_mid_mm=n0['y_mid_mm'] + (l0['y_mid_mm'] - n0['y_mid_mm']) * f, x_c_mm=0.0, tags=['trapezius'], w_src='CONV: geometric slope'))
            st = sorted(st + extra, key=lambda s: -s['z'])
        # three rings span the deltoid so the opened patch is exactly the arm hole
        st = sorted(st, key=lambda q: -q['z'])
        self.arm_hole = self.arm_hole_rows(st)
        if self.arm_hole:
            z_top, z_mid, z_bot = self.arm_hole
            ref = sorted(st, key=lambda q: q['z'])

            def is_measured(q):
                src = str(q.get('w_src', ''))
                return src.startswith('outline') or src.startswith('measured')

            def interp_st(z):
                lo = [q for q in ref if q['z'] <= z]
                hi = [q for q in ref if q['z'] >= z]
                if not lo or not hi:
                    q = dict(lo[-1] if lo else hi[0])
                    q['measured'] = is_measured(q)
                    return q
                a, b = lo[-1], hi[0]
                if b['z'] - a['z'] < 1e-9:
                    q = dict(a)
                    q['measured'] = is_measured(a)
                    return q
                t = (z - a['z']) / (b['z'] - a['z'])
                q = dict(a)
                for k in ('w_body_mm', 'd_body_mm', 'y_mid_mm', 'x_c_mm'):
                    if a.get(k) is not None and b.get(k) is not None:
                        q[k] = a[k] + (b[k] - a[k]) * t
                q['z'] = z
                q['measured'] = is_measured(a) if t < 0.5 else is_measured(b)
                return q
            keep_st = []
            z_first = max(q['z'] for q in st)
            for q in st:
                if z_bot - RING_MIN_GAP < q['z'] < z_top + RING_MIN_GAP:
                    if q['z'] >= z_top and q['z'] >= z_first - 1e-9 and z_top + RING_MIN_GAP <= C['landmarks']['chin']['z'] - 0.008:
                        z_nb = z_top + RING_MIN_GAP
                        z_neck_lm = C['landmarks'].get('neck', {}).get('z')
                        if z_neck_lm is not None and z_nb <= z_neck_lm <= C['landmarks']['chin']['z'] - 0.008:
                            z_nb = z_neck_lm
                        keep_st.append(dict(q, z=z_nb, tags=list(q.get('tags', [])) + ['neck_base']))
                    continue
                keep_st.append(q)
            root = self.arm_root()
            r_del = self.deltoid_radius() or 0.0
            hw_girdle = (root[1] - 0.3 * r_del) if root else 0.0
            meas = sorted([q for q in C['stations']['trunk'] if q.get('w_body_mm') and is_measured(q)], key=lambda q: q['z'])
            if root and meas and meas[0]['z'] <= root[0] <= meas[-1]['z'] and min(abs(q['z'] - root[0]) for q in meas) <= 0.03:
                lo = [q for q in meas if q['z'] <= root[0]][-1]
                hi = [q for q in meas if q['z'] >= root[0]][0]
                t_ = 0.0 if hi['z'] - lo['z'] < 1e-9 else (root[0] - lo['z']) / (hi['z'] - lo['z'])
                hw_meas = 0.5 * (lo['w_body_mm'] + (hi['w_body_mm'] - lo['w_body_mm']) * t_) / 1000.0
                hw_girdle = min(hw_girdle, hw_meas)
            for z_, tag in ((z_top, 'armhole_top'), (z_mid, 'armhole_mid'), (z_bot, 'armhole_bot')):
                q = interp_st(z_)
                q['tags'] = [tag]
                q['w_src'] = 'CONV: arm-hole row'
                # a guessed shoulder width reaches out to the acromion, a measured one stands
                if not q.pop('measured', False) and q['w_body_mm'] < 2000.0 * hw_girdle:
                    q['w_src'] += ' widened to the acromion'
                    q['w_body_mm'] = 2000.0 * hw_girdle
                keep_st.append(q)
            # the trapezius widens by at most 1.5 x per ring
            top = max((q for q in keep_st if 'armhole_top' in q['tags']), key=lambda q: q['z'])
            above = [q for q in keep_st if q['z'] > top['z']]
            rows_h = {t: q for q in keep_st for t in q.get('tags', []) if t.startswith('armhole')}
            if all(k in rows_h for k in ('armhole_top', 'armhole_mid', 'armhole_bot')):
                rows_h['armhole_mid']['w_body_mm'] = max(rows_h['armhole_mid']['w_body_mm'], rows_h['armhole_top']['w_body_mm'], rows_h['armhole_bot']['w_body_mm'])
            # shoulder rows: depth 0.7 of the chest at the acromion, 0.85 at the hole middle
            below = [q for q in keep_st if q['z'] < z_bot - 1e-9]
            d_chest = max(below, key=lambda q: q['z'])['d_body_mm'] if below else None
            for q in keep_st:
                tags_ = q.get('tags', [])
                if any(t.startswith('armhole') for t in tags_):
                    q['d_body_mm'] = max(q['d_body_mm'], 0.55 * q['w_body_mm'])
                    if d_chest:
                        frac = 0.7 if 'armhole_top' in tags_ else (0.85 if 'armhole_mid' in tags_ else 1.0)
                        q['d_body_mm'] = max(q['d_body_mm'], frac * d_chest)
            if above:
                nk = max(above, key=lambda q: q['z'])
                nk['d_body_mm'] = max(nk['d_body_mm'], 0.9 * nk['w_body_mm'])
                ratio = max(1.0, top['w_body_mm'] / max(1e-6, nk['w_body_mm']))
                n_steps = max(1, int(math.ceil(math.log(ratio) / math.log(1.5) - 1e-9)))
                n_steps = min(n_steps, int((nk['z'] - top['z']) / RING_MIN_GAP))
                keep_st = [q for q in keep_st if not (top['z'] < q['z'] < nk['z'] and not is_measured(q))]
                for k in range(1, n_steps):
                    f = k / float(n_steps)
                    q = dict(nk)
                    q['z'] = nk['z'] + (top['z'] - nk['z']) * f
                    q['w_body_mm'] = nk['w_body_mm'] * (top['w_body_mm'] / nk['w_body_mm']) ** f
                    q['d_body_mm'] = nk['d_body_mm'] * (top['d_body_mm'] / nk['d_body_mm']) ** f
                    q['y_mid_mm'] = nk['y_mid_mm'] + (top['y_mid_mm'] - nk['y_mid_mm']) * f
                    q['x_c_mm'] = 0.0
                    q['tags'] = ['trapezius']
                    q['w_src'] = 'CONV: trapezius step %d/%d' % (k, n_steps)
                    keep_st.append(q)
                log('[body] trunk: trapezius %.0f -> %.0f mm in %d step(s)' % (nk['w_body_mm'], top['w_body_mm'], n_steps))
            st = sorted(keep_st, key=lambda q: -q['z'])
            log('[body] trunk: arm-hole rows at z %.3f / %.3f / %.3f' % (z_top, z_mid, z_bot))
        merged = []
        for q in st:
            if merged and merged[-1]['z'] - q['z'] < RING_MIN_GAP:
                q_hole = any(t.startswith('armhole') for t in q.get('tags', []))
                p_hole = any(t.startswith('armhole') for t in merged[-1].get('tags', []))
                q_is_lm = any(t in LANDMARK_TAGS for t in q.get('tags', []))
                p_is_lm = any(t in LANDMARK_TAGS for t in merged[-1].get('tags', []))
                if q_hole and not p_hole:
                    if len(merged) == 1 and q['z'] + RING_MIN_GAP <= C['landmarks']['chin']['z'] - 0.008:
                        merged[-1] = dict(merged[-1], z=q['z'] + RING_MIN_GAP, tags=list(merged[-1].get('tags', [])) + ['neck_base'])
                        merged.append(q)
                    else:
                        merged[-1] = q
                elif not p_hole and q_is_lm and not p_is_lm:
                    merged[-1] = q
                continue
            merged.append(q)
        st = merged
        # a centre jumping 8 mm against both neighbours takes their median
        for key in ('x_c_mm', 'y_mid_mm'):
            vals = [q.get(key) for q in st]
            for k in range(len(st)):
                if vals[k] is None:
                    continue
                nb = [vals[j] for j in (k - 1, k + 1) if 0 <= j < len(st) and vals[j] is not None]
                if nb and all(abs(vals[k] - v_) > 8.0 for v_ in nb):
                    st[k][key] = sorted(nb + [vals[k]])[len(nb) // 2] if len(nb) == 2 else nb[0]
                    st[k]['w_src'] = str(st[k].get('w_src', '')) + ' (centre despiked)'
        rings = []
        prev_z = None
        lm = C['landmarks']
        z_neck, z_ap = lm['neck']['z'], lm['armpit']['z']
        droop = 0.0  # the trapezius rings make the slope
        st = sorted([q for q in st if q.get('w_body_mm') and q.get('d_body_mm')], key=lambda q: -q['z'])
        dense = []
        for a, b in zip(st, st[1:]):
            dense.append(a)
            gap = a['z'] - b['z']
            if gap > RING_MAX_GAP:
                n_ins = int(gap / RING_MAX_GAP)
                for i in range(1, n_ins + 1):
                    t = i / (n_ins + 1.0)
                    q = dict(a)
                    for k in ('z', 'w_body_mm', 'd_body_mm', 'y_mid_mm', 'x_c_mm'):
                        if a.get(k) is not None and b.get(k) is not None:
                            q[k] = a[k] + (b[k] - a[k]) * t
                    q['tags'] = ['interp']
                    q['w_src'] = 'CONV: ring interpolated between drawn stations %.0f mm apart' % (gap * 1000)
                    dense.append(q)
        if st:
            dense.append(st[-1])
        st = dense
        for s in st:
            z = s['z']
            if prev_z is not None and prev_z - z < 0.003:
                continue
            hw = 0.5 * s['w_body_mm'] / 1000.0
            dp = 0.5 * s['d_body_mm'] / 1000.0
            cy = s['y_mid_mm'] / 1000.0
            cx = s['x_c_mm'] / 1000.0 if s.get('x_c_mm') is not None else 0.0
            co = self.trunk_ring_co(z, hw, dp, cy, cx)
            if z_ap - 0.01 < z < z_neck and droop > 0 and hw > 0.05:
                t = min(1.0, (z_neck - z) / max(1e-6, z_neck - z_ap))
                co = [(x, y, zz - droop * t * (abs(x) / hw) ** 1.6) for (x, y, zz) in co]
            rings.append(self.ring(co))
            if 'neck_base' in s.get('tags', []) or 'trapezius' in s.get('tags', []):
                self.crease_cos.extend(v.co.copy() for v in rings[-1])
            self.map['z'].setdefault('trunk', []).append(z)
            prev_z = z
        all_rings = [head_rings[-1]] + rings
        for a, b in zip(all_rings, all_rings[1:]):
            self.bridge(a, b)
        self.map['trunk'] = rings
        self.trunk_st = [s for s in st]
        log('[body] trunk: %d rings, z %.3f .. %.3f' % (len(rings), rings[0][0].co.z, rings[-1][0].co.z))
        return rings

    def leg_station_rows(self):
        C = self.C
        R = [s for s in C['stations']['leg_R'] if s['w_mm'] and s['d_mm']]
        L = [s for s in C['stations']['leg_L'] if s['w_mm'] and s['d_mm']]
        if not R or not L:
            self.log('[body] leg %s has no usable station: built as the mirror of the other' % ('R' if not R else 'L'))
            R, L = R or L, L or R
        zs = sorted(set([s['z'] for s in R] + [s['z'] for s in L]), reverse=True)

        def interp(tab, key, z):
            zz = [s['z'] for s in tab][::-1]
            vv = [s[key] for s in tab][::-1]
            if z <= zz[0]:
                return vv[0]
            if z >= zz[-1]:
                return vv[-1]
            for i in range(len(zz) - 1):
                if zz[i] <= z <= zz[i + 1]:
                    t = (z - zz[i]) / (zz[i + 1] - zz[i])
                    return vv[i] + (vv[i + 1] - vv[i]) * t
            return vv[-1]
        rows = []
        last = None
        for z in zs:
            if last is not None and last - z < 0.008:
                continue
            kw_ = 'w_body_mm' if all(s.get('w_body_mm') for s in R + L) else 'w_mm'
            kd_ = 'd_body_mm' if all(s.get('d_body_mm') for s in R + L) else 'd_mm'
            w = 0.5 * (interp(R, kw_, z) + interp(L, kw_, z)) / 1000.0
            d = 0.5 * (interp(R, kd_, z) + interp(L, kd_, z)) / 1000.0
            xc = 0.5 * (abs(interp(R, 'x_c_mm', z)) + abs(interp(L, 'x_c_mm', z))) / 1000.0
            ym = 0.5 * (interp(R, 'y_mid_mm', z) + interp(L, 'y_mid_mm', z)) / 1000.0
            tags = sorted(set(sum([s['tags'] for s in R + L if abs(s['z'] - z) < 1e-6], [])))
            rows.append(dict(z=z, w=w, d=d, xc=xc, ym=ym, tags=tags))
            last = z
        return rows

    def leg_ring_co(self, row, sd, ff=0.0, bf=0.0):
        z, w, d, ym, xc = row['z'], row['w'], row['d'], row['ym'], row['xc']
        out = []
        for a in ANG:
            t = math.radians(a)
            lat = (w * 0.5) * math.cos(t)
            y = ym - (d * 0.5) * math.sin(t)
            y += ff * shoulder_win(a, 90.0) + bf * shoulder_win(a, 270.0)
            out.append((sd * (xc + lat), y, z))
        return out

    def build_legs(self, trunk_rings):
        C, log, bm = self.C, self.log, self.bm
        rows = self.leg_station_rows()
        knee_z = C['landmarks'].get('knee_L', C['landmarks'].get('knee_R'))['z']
        B = trunk_rings[-1]
        Z_HIP = B[0].co.z
        hip_d = max(v.co.y for v in B) - min(v.co.y for v in B)
        hip_yc = 0.5 * (max(v.co.y for v in B) + min(v.co.y for v in B))
        hip_xc = 0.5 * (max(v.co.x for v in B) + min(v.co.x for v in B))
        _first = [r['z'] for r in rows if r['z'] < Z_HIP - 0.008]
        _gap = (Z_HIP - _first[0]) if _first else 0.04
        X_IN, Z_CR = 0.0150, Z_HIP - min(0.0120, 0.35 * _gap)
        n_blend = 2 if _gap >= 0.030 else (1 if _gap >= 0.016 else 0)
        IN = {1: [], -1: []}
        for k in range(N_IN):
            t = (k + 1) / float(N_IN + 1)
            yy = hip_yc + hip_d * (0.55 - 1.10 * t)
            IN[1].append(bm.verts.new((hip_xc + X_IN, yy, Z_CR)))
            IN[-1].append(bm.verts.new((hip_xc - X_IN, yy, Z_CR)))
        self.crease_cos.extend(v.co.copy() for v in IN[1] + IN[-1])
        HALF = NC // 2
        _R = [B[HALF]] + IN[1] + [B[0]]
        _L = [B[HALF]] + IN[-1] + [B[0]]
        for i in range(N_IN + 1):
            f = [_R[i], _R[i + 1], _L[i + 1], _L[i]]
            bm.faces.new([v for j, v in enumerate(f) if v is not f[j - 1]])
        first_built = [i for i, r in enumerate(rows) if r['z'] < Z_HIP - 0.008]
        fb = first_built[0] if first_built else 0
        for sd in (1, -1):
            loop = ([B[k] for k in range(0, HALF + 1)] + IN[1] if sd > 0
                    else [B[0]] + [B[k] for k in range(NC - 1, HALF - 1, -1)] + IN[-1])
            rings = [loop]
            tgt = self.leg_ring_co(rows[fb], sd)
            z0 = sum(v.co.z for v in loop) / len(loop)
            z1 = rows[fb]['z']
            for i in range(1, n_blend + 1):
                f = i / float(n_blend + 1)
                t = 1.0 - (1.0 - f) ** 1.6
                co = [(a.co.x + (b[0] - a.co.x) * t, a.co.y + (b[1] - a.co.y) * t, z0 + (z1 - z0) * f) for a, b in zip(loop, tgt)]
                rings.append(self.ring(co))
            for i in range(fb, len(rows)):
                r = rows[i]
                # a knee: patella proud, back of the knee pulled forward
                near = abs(r['z'] - knee_z) < 0.03
                ff = 0.002 if near else 0.0
                bf = 0.002 if near else 0.0
                rings.append(self.ring(self.leg_ring_co(r, sd, ff, bf)))
            for a, b in zip(rings, rings[1:]):
                self.bridge(a, b)
            side = 'L' if sd > 0 else 'R'
            self.map['leg'][side] = rings
            self.leg_rows = rows
            log('[body] leg %s: %d rings (%d blend + %d drawn), z %.3f .. %.3f' % (side, len(rings), n_blend, len(rows) - fb, Z_HIP, rows[-1]['z']))
            self.build_foot(rings, sd)
        return

    def build_foot(self, leg_rings, sd):
        C, bm, log = self.C, self.bm, self.log
        side = 'L' if sd > 0 else 'R'
        ank = leg_rings[-1]
        cz = sum(v.co.z for v in ank) / len(ank)
        cx = sum(v.co.x for v in ank) / len(ank)
        cy = sum(v.co.y for v in ank) / len(ank)
        sole_top = C['landmarks'].get('sole_top_' + side, {}).get('z', 0.02)
        z_sole = max(0.01, sole_top + 0.004)
        # the foot is a stub the shoe covers: the ankle section carried down to the sole
        rowB, rowC = [], []
        for v in ank:
            dv = v.co - Vector((cx, cy, cz))
            for row, f, zz in ((rowB, 0.96, cz - 0.5 * (cz - z_sole)), (rowC, 0.90, z_sole)):
                row.append(bm.verts.new((cx + dv.x * f, cy + dv.y * f, zz)))
        self.bridge(ank, rowB)
        self.bridge(rowB, rowC)
        c = Vector((cx, cy, z_sole))
        kt = min(range(len(rowC)), key=lambda i: (rowC[i].co - c).y)
        n = len(rowC)
        chainA = [rowC[(kt + i) % n] for i in range(n // 2 + 1)]
        chainB = [rowC[(kt - i) % n] for i in range(n // 2 + 1)]
        for i in range(n // 2):
            f = [chainA[i], chainA[i + 1], chainB[i + 1], chainB[i]]
            bm.faces.new([v for j, v in enumerate(f) if v is not f[j - 1]])
        self.map['foot'][side] = [rowB, rowC]
        log('[body] foot %s: stub (ankle section x0.90) down to z %.3f' % (side, z_sole))

    def shoulder_line_z(self, st):
        """The first trunk station whose width is the drawn outline, else the 2 x neck rule."""
        lm = self.C['landmarks']
        for q in st:
            if q['z'] <= lm['shoulder']['z'] + 1e-6 and str(q.get('w_src', '')).startswith('outline'):
                return q['z']
        w_neck = [q for q in st if 'neck' in q['tags']]
        w_neck = w_neck[0]['w_body_mm'] if w_neck else st[0]['w_body_mm']
        for q in st:
            if q['z'] <= lm['shoulder']['z'] + 1e-6 and q['w_body_mm'] >= 2 * w_neck:
                return q['z']
        return lm['shoulder']['z']

    def deltoid_radius(self, sleeve=False):
        """Widest arm section near the root, 4 % proud, with the cloth when `sleeve`."""
        rows = self.arm_rows(1)
        if not rows:
            return None
        s_end = max(1e-6, rows[-1]['s'])
        key = 'w_all' if sleeve else 'w'
        near = [r_.get(key, r_['w']) for r_ in rows if r_['s'] <= max(0.06, 0.15 * s_end)] or [rows[0].get(key, rows[0]['w'])]
        return 0.5 * max(near) * 1.04

    def arm_root(self):
        zs, xs = [], []
        for side in ('R', 'L'):
            A = self.C['stations'].get('arm_' + side) or {}
            r_ = (A.get('landmarks') or {}).get('root')
            if r_ and r_.get('z') is not None:
                zs.append(r_['z'])
                xs.append(abs(r_.get('x') or 0.0))
        if not zs:
            return None
        return (sum(zs) / len(zs), sum(xs) / len(xs))

    def arm_root_z(self):
        r_ = self.arm_root()
        return r_[0] if r_ else None

    def arm_hole_rows(self, st):
        """z of the three trunk rows framing the arm hole, at least 45 mm tall."""
        lm = self.C['landmarks']
        r_del = self.deltoid_radius(sleeve=True)
        z_root = self.arm_root_z()
        if r_del is None or z_root is None:
            return None
        z_ap = lm['armpit']['z']
        z_top = z_root
        span = z_top - z_ap
        if span < 0.03:
            return None
        h = min(max(2.0 * r_del, 0.04), span)
        z_bot = z_top - h
        if z_bot < z_ap + RING_MIN_GAP:
            z_bot = z_ap
        if z_top - z_bot < 0.03:
            return None
        return (z_top, 0.5 * (z_top + z_bot), z_bot)

    def build_arms(self, trunk_rings):
        C, bm, log = self.C, self.bm, self.log
        lm = C['landmarks']
        z_line = None
        for s in self.trunk_st:
            if s['z'] <= lm['shoulder']['z'] + 1e-6 and str(s.get('w_src', '')).startswith('outline'):
                z_line = s['z']
                break
        if z_line is None:
            w_neck = [s for s in self.trunk_st if 'neck' in s['tags']]
            w_neck = w_neck[0]['w_body_mm'] if w_neck else self.trunk_st[0]['w_body_mm']
            for s in self.trunk_st:
                if s['z'] <= lm['shoulder']['z'] + 1e-6 and s['w_body_mm'] >= 2 * w_neck:
                    z_line = s['z']
                    break
        z_line = z_line if z_line is not None else lm['shoulder']['z']
        z_ap = lm['armpit']['z']
        zc = z_line - 0.45 * (z_line - z_ap)
        zs = [sum(v.co.z for v in r) / len(r) for r in trunk_rings]
        if getattr(self, 'arm_hole', None):
            i = min(range(len(zs) - 2), key=lambda k: abs(zs[k] - self.arm_hole[0]))
        else:
            i = min(range(len(zs) - 2), key=lambda k: abs(0.5 * (zs[k] + zs[k + 2]) - zc))
        # arm room: the trunk garment plus the sleeve plus 2 mm
        ap_st = min(C['stations']['trunk'], key=lambda q: abs(q['z'] - z_ap))
        room_trunk = max(0.0, 0.5 * ((ap_st.get('w_mm') or 0) - (ap_st.get('w_body_mm') or 0)) / 1000.0)
        rows_all = self.arm_rows(1)
        s_end_all = max(1e-6, rows_all[-1]['s']) if rows_all else 1.0
        room_sleeve = max([0.0] + [0.5 * max(0.0, r_.get('w_all', r_['w']) - r_['w']) for r_ in rows_all if r_['s'] <= 0.4 * s_end_all])
        arm_gap = max(0.004, 0.002 + room_trunk + room_sleeve)
        log('[body] arms: room beside the trunk %.1f mm (trunk garment %.1f, sleeve %.1f)' % (arm_gap * 1000, room_trunk * 1000, room_sleeve * 1000))
        scrap = []
        ref_loop = None
        for sd in (1, -1):
            side = 'L' if sd > 0 else 'R'
            k_side = NC // 4 if sd > 0 else 3 * NC // 4
            cols = [k_side - 1, k_side]
            loop = open_patch(bm, trunk_rings, [i, i + 1], cols, NC, scrap=scrap)
            self.crease_cos.extend(v.co.copy() for v in loop)
            if ref_loop is None:
                ref_loop = loop
            else:
                mirrored = [Vector((-v.co.x, v.co.y, v.co.z)) for v in ref_loop]
                order = []
                used = set()
                for m in mirrored:
                    j = min((jj for jj in range(len(loop)) if jj not in used), key=lambda jj: (loop[jj].co - m).length)
                    used.add(j)
                    order.append(loop[j])
                loop = order
            A = C['stations']['arm_' + side]
            c0 = sum((v.co for v in loop), Vector()) / len(loop)
            y_arm = c0.y - 0.008
            y_hand = A.get('y_hand_m')
            pts, radii, flat = [c0], [None], [None]
            keep = self.arm_rows(sd)
            s_end = max(1e-6, keep[-1]['s']) if keep else 1.0
            trunk_polys = []
            for r_ in self.map['trunk']:
                cos_ = [v_.co for v_ in r_]
                trunk_polys.append((sum(c.z for c in cos_) / len(cos_), cos_))
            trunk_polys.sort(key=lambda q: q[0])
            zs_ = [q[0] for q in trunk_polys]

            def hw_poly(poly, y_):
                best = 0.0
                n_ = len(poly)
                for k_ in range(n_):
                    a_, b_ = poly[k_], poly[(k_ + 1) % n_]
                    if (a_.y <= y_ < b_.y) or (b_.y <= y_ < a_.y):
                        t_ = (y_ - a_.y) / (b_.y - a_.y)
                        best = max(best, sd * (a_.x + (b_.x - a_.x) * t_))
                return best

            def hw_at(z_, y_=None):
                if not zs_:
                    return 0.0
                if y_ is None:
                    f_ = lambda poly: max(sd * c.x for c in poly)
                else:
                    f_ = lambda poly: hw_poly(poly, y_)
                if z_ >= zs_[-1]:
                    return f_(trunk_polys[-1][1])
                if z_ <= zs_[0]:
                    return f_(trunk_polys[0][1])
                for (za, pa), (zb, pb) in zip(trunk_polys, trunk_polys[1:]):
                    if za <= z_ <= zb:
                        return max(f_(pa), f_(pb))
                return 0.0
            r0 = sum((v.co - c0).length for v in loop) / len(loop)
            z_hole = (min(v.co.z for v in loop) - 0.002, max(v.co.z for v in loop) + 0.002)
            y_hole = (min(v.co.y for v in loop) - 0.002, max(v.co.y for v in loop) + 0.002)
            if keep:
                a_dir = Vector((sd * keep[-1]['x'], 0.0, keep[-1]['z'])) - Vector((sd * keep[0]['x'], 0.0, keep[0]['z']))
                a_dir = a_dir.normalized() if a_dir.length > 1e-6 else Vector((0, 0, -1))
            else:
                a_dir = Vector((0, 0, -1))
            # the root patch normal (Newell), in the xz plane, away from the trunk
            n0 = Vector((0.0, 0.0, 0.0))
            for k_ in range(len(loop)):
                n0 += loop[k_].co.cross(loop[(k_ + 1) % len(loop)].co)
            n0.y = 0.0
            if n0.length < 1e-9:
                n0 = Vector((sd, 0.0, 0.0))
            if n0.x * sd < 0:
                n0 = -n0
            n0.normalize()
            if n0.x * sd < 0.5:
                n0 = (n0 + Vector((sd, 0.0, 0.0))).normalized()
            # the arm never leaves the trunk rising, or the deltoid pokes above the shoulder
            t0 = Vector((n0.x, 0.0, min(0.0, n0.z))).normalized()

            r_del_ = max(0.015, self.deltoid_radius() or 0.02)

            def ring_clear(c_, r_, t_):
                """How far a ring must move along +x to stand `arm_gap` off the trunk."""
                u_ = Vector((0, 1, 0)).cross(t_)
                u_ = u_.normalized() if u_.length > 1e-9 else Vector((sd, 0, 0))
                v_ = t_.cross(u_).normalized()
                need = 0.0
                for a_ in range(12):
                    al = 2 * math.pi * a_ / 12
                    q_ = c_ + (u_ * math.cos(al) + v_ * math.sin(al)) * r_
                    d_box = max(0.0, z_hole[0] - q_.z, q_.z - z_hole[1], y_hole[0] - q_.y, q_.y - y_hole[1])
                    if d_box <= 0.0:
                        back = -(q_ - c0).dot(n0)
                        if back > 0:
                            need = max(need, back / max(0.3, abs(n0.x)))
                        continue
                    gap_ = 0.002 + (arm_gap - 0.002) * min(1.0, d_box / max(1e-6, 0.6 * r_del_))
                    need = max(need, hw_at(q_.z, q_.y) + gap_ - sd * q_.x)
                return need

            def stations_from(base, d0):
                """Drawn stations past `base` along the arm, 0.6 radius apart, clear of the trunk."""
                P_, R_, F_ = [], [], []
                d_prev = d0
                n_skip_ = n_out_ = 0
                for s in keep:
                    y_ = y_arm if y_hand is None else (c0.y + (y_hand - c0.y) * min(1.0, s['s'] / s_end))
                    r_ = 0.5 * s['w']
                    p_ = Vector((sd * s['x'], y_, s['z']))
                    push = ring_clear(p_, r_, a_dir)
                    if push > 0:
                        p_ = p_ + Vector((sd * push, 0.0, 0.0))
                        n_out_ += 1
                    d_ = (p_ - base).dot(a_dir)
                    if d_ <= d_prev + max(0.004, 0.6 * r_):
                        if 'wrist' not in s['tags']:
                            n_skip_ += 1
                            continue
                        if P_:
                            P_.pop(); R_.pop(); F_.pop()
                    d_prev = d_
                    P_.append(p_)
                    R_.append(r_)
                    F_.append(min(1.0, s['d'] / s['w']) if s['d'] else 1.0)
                return P_, R_, F_, n_skip_, n_out_

            # the shoulder is a torus bend: rings of radius r <= R never fold on the armpit side
            P1, R1, F1, _, _ = stations_from(c0, r0)
            r_up = R1[0] if R1 else r0
            r_del = max(r_up, self.deltoid_radius() or 0.0)
            theta = math.acos(max(-1.0, min(1.0, t0.dot(a_dir))))
            arc_pts, arc_r = [], []
            c_end = c0
            if theta > math.radians(12):
                m = a_dir - t0 * a_dir.dot(t0)
                m = m.normalized() if m.length > 1e-9 else Vector((0, 0, -1))
                R_min = max(r_del, r_up) + room_sleeve + 0.003
                u_ = t0 * math.sin(theta) + m * (1.0 - math.cos(theta))
                R_ = R_min
                R_star = None
                if P1:
                    u_perp = u_ - a_dir * u_.dot(a_dir)
                    w_ = c0 - P1[0]
                    w_ = w_ - a_dir * w_.dot(a_dir)
                    if u_perp.length_squared > 1e-9:
                        R_star = -w_.dot(u_perp) / u_perp.length_squared
                        R_ = min(max(R_min, R_star), R_min + 0.06)
                N_arc = max(1 if theta < math.radians(40) else 2, int(math.ceil(R_ * theta / (1.2 * r_del))))
                # the first ring clears the trunk garment rim by 3 mm, or the sleeve's first band folds
                r_sleeve_ = max(r_del, self.deltoid_radius(sleeve=True) or 0.0)
                need_ = room_trunk + 0.003
                if R_ - r_sleeve_ < need_ / math.sin(theta):
                    R_ = r_sleeve_ + need_ / math.sin(theta)
                phi_min = math.asin(min(1.0, need_ / max(1e-6, R_ - r_sleeve_)))
                while N_arc > 1 and theta / N_arc < phi_min:
                    N_arc -= 1
                arc_r = [r_del + (r_up - r_del) * (k - 1) / max(1, N_arc - 1) for k in range(1, N_arc + 1)] if N_arc > 1 else [r_del]

                r_sleeve = max(r_del, self.deltoid_radius(sleeve=True) or 0.0)
                z_edge_bot = min(v.co.z for v in loop)

                def arc_rings(R):
                    out = []
                    for k in range(1, N_arc + 1):
                        phi = theta * k / N_arc
                        c_ = c0 + (t0 * math.sin(phi) + m * (1.0 - math.cos(phi))) * R
                        t_ = t0 * math.cos(phi) + m * math.sin(phi)
                        if k == 1:
                            low = c_.z - r_sleeve * math.sqrt(max(0.0, 1.0 - t_.z * t_.z))
                            if low > z_edge_bot - 0.003:
                                c_ = Vector((c_.x, c_.y, c_.z - (low - (z_edge_bot - 0.003))))
                        out.append((c_, arc_r[k - 1], t_))
                    return out
                n_push = 0
                R_0 = R_
                for _ in range(12):
                    step = 0.0
                    for k, (c_, r_, t_) in enumerate(arc_rings(R_), 1):
                        nd = ring_clear(c_, r_, t_)
                        if nd > 0:
                            step = max(step, nd / max(0.25, math.sin(theta * k / N_arc)))
                    if step <= 0:
                        break
                    R_ += step + 0.0005
                    n_push += 1
                arc_pts = [c_ for (c_, r_, t_) in arc_rings(R_)]
                c_end = arc_pts[-1]
                log('[body] arm %s: shoulder bend %.0f deg, radius %.1f mm (%d ring%s, deltoid r %.1f mm; drawn axis %s%s)' % (
                    side, math.degrees(theta), R_ * 1000, N_arc, 's' if N_arc > 1 else '', r_del * 1000,
                    ('R %.0f mm' % (R_star * 1000)) if R_star is not None else 'none',
                    ', pushed %.0f mm clear of the trunk' % ((R_ - R_0) * 1000) if n_push else ''))
            else:
                arc_pts = [c0 + t0 * max(0.018, 0.6 * r_del)]
                arc_r = [r_del]
                c_end = arc_pts[0]
            P2, R2, F2, n_skip, n_out = stations_from(c_end, 0.0)
            if n_skip or n_out:
                log('[body] arm %s: %d stations inside the shoulder skipped, %d pushed clear of the trunk' % (side, n_skip, n_out))
            pts = [c0] + arc_pts + P2
            radii = [r0] + arc_r + R2
            flat = [1.0] + [1.0] * len(arc_pts) + F2
            rings = extrude_loop(bm, loop, pts, radii, round_by=1, flat=flat)
            self.map['arm'][side] = rings
            log('[body] arm %s: %d rings, wrist at (%.3f, %.3f, %.3f), radius %.1f mm' % (
                side, len(rings), pts[-1].x, pts[-1].y, pts[-1].z, radii[-1] * 1000))
            self.build_hand(rings, sd, A, pts, keep)
        for v in scrap:
            if v.is_valid:
                bm.verts.remove(v)

    def arm_rows(self, sd):
        """Arm stations of both sides merged on u = s / s_wrist, from 35 mm out to the wrist."""
        C = self.C
        if getattr(self, '_arm_rows', None) is not None:
            return self._arm_rows
        tabs = {}
        for side in ('R', 'L'):
            A = C['stations']['arm_' + side]
            sw = A['landmarks']['wrist']['s_px'] / C['calibration']['ppm']
            s0 = A['stations'][0]['s_m']
            rows = [s for s in A['stations'] if s['w_mm'] and s['w_mm'] > 8 and s['s_m'] <= sw + 1e-9]
            tabs[side] = dict(sw=sw, s0=s0, rows=rows)
        us = sorted(set(round(s['s_m'] / tabs[sd_]['sw'], 3) for sd_ in tabs for s in tabs[sd_]['rows']))

        def interp(side, key, u):
            rows = tabs[side]['rows']
            uu = [s['s_m'] / tabs[side]['sw'] for s in rows]
            vv = [abs(s[key]) if key == 'x_m' else s[key] for s in rows]
            if u <= uu[0]:
                return vv[0]
            if u >= uu[-1]:
                return vv[-1]
            for i in range(len(uu) - 1):
                if uu[i] <= u <= uu[i + 1]:
                    t = (u - uu[i]) / max(1e-9, uu[i + 1] - uu[i])
                    return vv[i] + (vv[i + 1] - vv[i]) * t
            return vv[-1]
        out = []
        sw = 0.5 * (tabs['R']['sw'] + tabs['L']['sw'])
        for u in us:
            s_m = u * sw
            if s_m < 0.035:
                continue
            tags = sorted(set(sum([s['tags'] for sd_ in tabs for s in tabs[sd_]['rows'] if abs(s['s_m'] / tabs[sd_]['sw'] - u) < 1e-3], [])))
            kw_ = 'w_body_mm' if all(s.get('w_body_mm') for sd_ in tabs for s in tabs[sd_]['rows']) else 'w_mm'
            kd_ = 'd_body_mm' if all(s.get('d_body_mm') for sd_ in tabs for s in tabs[sd_]['rows']) else 'd_mm'
            row = dict(u=u, s=s_m, x=0.5 * (interp('R', 'x_m', u) + interp('L', 'x_m', u)),
                       z=0.5 * (interp('R', 'z', u) + interp('L', 'z', u)),
                       w=0.5 * (interp('R', kw_, u) + interp('L', kw_, u)) / 1000.0,
                       w_all=0.5 * (interp('R', 'w_mm', u) + interp('L', 'w_mm', u)) / 1000.0,
                       d=0.5 * (interp('R', kd_, u) + interp('L', kd_, u)) / 1000.0, tags=tags)
            if out and (row['s'] - out[-1]['s']) < 0.018 and not tags:
                continue
            out.append(row)
        # a width climbing from the root is the garment hiding the deltoid: use the plateau
        top = [r_ for r_ in out if r_['s'] <= max(0.15, 0.45 * sw)]
        if len(top) >= 2:
            w_max = max(top[i].get('w_all', top[i]['w']) for i in range(len(top)))
            k_pk = next(i for i in range(len(top)) if top[i].get('w_all', top[i]['w']) >= 0.9 * w_max)
            w_pk, wa_pk, d_pk = top[k_pk]['w'], top[k_pk].get('w_all', top[k_pk]['w']), top[k_pk]['d']
            n_occ = 0
            for r_ in top[:k_pk]:
                if r_.get('w_all', r_['w']) < 0.85 * wa_pk:
                    r_['w'], r_['w_all'], r_['d'] = max(r_['w'], w_pk), max(r_.get('w_all', r_['w']), wa_pk), max(r_['d'], d_pk)
                    r_['occluded_root'] = True
                    n_occ += 1
            if n_occ:
                self.log('[body] arm: %d root ring(s) hidden by the trunk take the deltoid width %.0f mm (a limb never grows from a point)' % (n_occ, 1000 * wa_pk))
        # the arm under a sleeve is the sleeve's narrowest local width less the cloth
        for r_ in out:
            r_['covered'] = (r_.get('w_all', r_['w']) - r_['w']) > 0.001
        smooth_w = []
        for r_ in out:
            if r_['covered']:
                win = [q['w'] for q in out if q['covered'] and abs(q['s'] - r_['s']) <= 0.05]
                smooth_w.append(min(win) if win else r_['w'])
            else:
                smooth_w.append(r_['w'])
        for r_, w_ in zip(out, smooth_w):
            if r_['covered'] and w_ < r_['w']:
                r_['d'] = r_['d'] * (w_ / max(1e-6, r_['w']))
                r_['w'] = w_
        if out[-1]['u'] < 0.999:
            u = 1.0
            out.append(dict(u=u, s=sw, x=0.5 * (interp('R', 'x_m', u) + interp('L', 'x_m', u)), z=0.5 * (interp('R', 'z', u) + interp('L', 'z', u)),
                            w=0.5 * (interp('R', 'w_mm', u) + interp('L', 'w_mm', u)) / 1000.0, w_all=0.5 * (interp('R', 'w_mm', u) + interp('L', 'w_mm', u)) / 1000.0,
                            d=0.5 * (interp('R', 'd_mm', u) + interp('L', 'd_mm', u)) / 1000.0, tags=['wrist']))
        self.log('[body] arm profile (s mm: skin w / drawn w / d): ' + ', '.join('%.0f: %.0f/%.0f/%.0f%s' % (
            1000 * r_['s'], 1000 * r_['w'], 1000 * r_.get('w_all', r_['w']), 1000 * r_['d'], '*' if r_.get('occluded_root') else '') for r_ in out))
        self._arm_rows = out
        return out

    def build_hand(self, arm_rings, sd, A, pts, keep):
        C, bm, log = self.C, self.bm, self.log
        side = 'L' if sd > 0 else 'R'
        lens, wids = [], []
        for sd_ in ('R', 'L'):
            lmA = C['stations']['arm_' + sd_]['landmarks']
            lens.append((lmA['tip']['s_px'] - lmA['wrist']['s_px']) / C['calibration']['ppm'])
            wids.append(lmA['hand_widest']['w_mm'] / 1000.0)
        hand_len = max(0.05, sum(lens) / 2)
        hand_w = max(0.03, sum(wids) / 2)
        P = HP.P
        HW = hand_len / P['length']
        HP.HW = HW
        w_ref = (P['y_front'][-1][1] + P['y_back'][-1][1]) * HW
        kw = min(1.3, max(0.5, hand_w / w_ref))
        over = dict(cap_wrist=False,
                    y_front=tuple((t, v * kw) for t, v in P['y_front']),
                    y_back=tuple((t, v * kw) for t, v in P['y_back']),
                    x_half=tuple((t, v * min(1.0, kw)) for t, v in P['x_half']))
        hbm, hwr = HP.build_hand(over)
        wr = arm_rings[-1]
        cw = sum((v.co for v in wr), Vector()) / len(wr)
        d = (pts[-1] - pts[-2]).normalized()
        # hand module: fingers along -Z, width along Y, palm normal along X
        fwd = Vector((0, -1, 0))
        c1 = -fwd
        c3 = -d
        c2 = c3.cross(c1).normalized()
        M = Matrix((c1, c2, c3)).transposed().to_4x4()
        M.translation = cw
        idx = {}
        hbm.verts.ensure_lookup_table()
        for v in hbm.verts:
            co = v.co.copy()
            if sd < 0:
                co.y = -co.y
            idx[v.index] = bm.verts.new(M @ co)
        for f in hbm.faces:
            try:
                bm.faces.new([idx[v.index] for v in f.verts])
            except ValueError:
                pass
        wring = [idx[v.index] for v in hwr]
        hbm.free()
        # squeeze the hand's wrist ring to 0.9 of the arm radius so the bridge does not fold
        r_arm = sum((v.co - cw).length for v in wr) / len(wr)
        r_hand = max((v.co - cw).length for v in wring)
        if r_hand > 0.85 * r_arm:
            k = 0.85 * r_arm / max(1e-6, r_hand)
            for v in wring:
                v.co = cw + (v.co - cw) * k
        for v in idx.values():
            v.co = v.co + d * 0.006
        for v in idx.values():
            t_ = (v.co - cw).dot(d)
            if t_ < 0.004:
                v.co = v.co + d * (0.004 - t_)
        bridge_8_10(bm, wr, wring)
        self.map['hand'][side] = wring
        log('[body] hand %s: length %.0f mm, width %.0f mm (HW %.3f, kw %.2f)' % (side, hand_len * 1000, hand_w * 1000, HW, kw))

    def finish(self, out_blend, report=None):
        bm, log = self.bm, self.log
        from mathutils.kdtree import KDTree
        mapped = set()

        def collect(x):
            if isinstance(x, bmesh.types.BMVert):
                mapped.add(x)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    collect(y)
            elif isinstance(x, dict):
                for y in x.values():
                    collect(y)
        collect({k: v for k, v in self.map.items() if k not in ('z', 'landmarks')})
        kd = KDTree(len(bm.verts))
        bm.verts.ensure_lookup_table()
        for i, v in enumerate(bm.verts):
            kd.insert(v.co, i)
        kd.balance()
        target = {}
        for v in bm.verts:
            if v in target:
                continue
            for co, i, dist in kd.find_range(v.co, 0.0004):
                u = bm.verts[i]
                if u is v or u in target or v in target:
                    continue
                if u in mapped and v not in mapped:
                    target[v] = u
                elif v in mapped and u not in mapped:
                    target[u] = v
                else:
                    target[u] = v
        if target:
            bmesh.ops.weld_verts(bm, targetmap=target)
            log('[body] %d near-double vert(s) merged' % len(target))

            def prune(x):
                if isinstance(x, list):
                    if x and isinstance(x[0], bmesh.types.BMVert):
                        x[:] = [v for v in x if v.is_valid]
                    else:
                        for y in x:
                            prune(y)
                elif isinstance(x, dict):
                    for y in x.values():
                        prune(y)
            prune({k: v for k, v in self.map.items() if k not in ('z', 'landmarks')})
        bnd = [e for e in bm.edges if e.is_boundary]
        if bnd:
            res = bmesh.ops.holes_fill(bm, edges=bnd, sides=6)
            n_new = len(res.get('faces', []))
            bnd2 = [e for e in bm.edges if e.is_boundary]
            log('[body] %d boundary edge(s): %d hole face(s) filled, %d boundary edge(s) left' % (len(bnd), n_new, len(bnd2)))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        vol = 0.0
        for f in bm.faces:
            vs = [v.co for v in f.verts]
            for i in range(1, len(vs) - 1):
                vol += vs[0].dot(vs[i].cross(vs[i + 1]))
        if vol < 0:
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        # designed creases for the judge's dent check
        cl = bm.verts.layers.int.get('s3d_crease')
        n_cr = 0
        kd_c = KDTree(len(self.crease_cos))
        for i_, co_ in enumerate(self.crease_cos):
            kd_c.insert(co_, i_)
        kd_c.balance()
        for v in bm.verts:
            if self.crease_cos and kd_c.find(v.co)[2] < 0.0005:
                v[cl] = 1
                n_cr += 1
        log('[body] %d vertices marked as designed creases (arm holes, crotch, nape)' % n_cr)
        me = bpy.data.meshes.new('Body')
        bm.to_mesh(me)
        ob = bpy.data.objects.new('Body', me)
        bpy.context.collection.objects.link(ob)
        for p in me.polygons:
            p.use_smooth = False
        ob['s3d_part'] = 'body'
        ob['s3d_class'] = 'skin'
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        for part_ in ('leg', 'foot', 'arm'):
            for s_, rr in self.map[part_].items():
                for k_, r in enumerate(rr):
                    bad = [j for j, v in enumerate(r) if not v.is_valid]
                    if bad:
                        log('[body] WARNING: %s %s ring %d has %d removed vertex(es) at columns %s' % (part_, s_, k_, len(bad), bad))
        MAP = dict(trunk=[[v.index for v in r if v.is_valid] for r in self.map['trunk']],
                   head=[[v.index for v in r if v.is_valid] for r in self.map['head']],
                   leg={s: [[v.index for v in r if v.is_valid] for r in rr] for s, rr in self.map['leg'].items()},
                   foot={s: [[v.index for v in r if v.is_valid] for r in rr] for s, rr in self.map['foot'].items()},
                   arm={s: [[v.index for v in r if v.is_valid] for r in rr] for s, rr in self.map['arm'].items()},
                   hand={s: [v.index for v in r if v.is_valid] for s, r in self.map['hand'].items()},
                   z=self.map['z'], dome=self.dome, nc=NC, nl=NL, ang=ANG,
                   counts=dict(verts=len(bm.verts), faces=len(bm.faces),
                               tris=sum(len(f.verts) - 2 for f in bm.faces)))
        if report:
            json.dump(MAP, open(report, 'w'), indent=1)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        log('[body] %s: %d verts %d faces %d tris -> %s' % (ob.name, len(bm.verts), len(bm.faces), MAP['counts']['tris'], out_blend))
        bm.free()


def _basis(ring):
    c = sum((v.co for v in ring), Vector()) / len(ring)
    n = Vector((0.0, 0.0, 0.0))
    for k in range(len(ring)):
        a, b = ring[k].co, ring[(k + 1) % len(ring)].co
        n += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
    n.normalize()
    e1 = ring[0].co - c
    e1 -= n * e1.dot(n)
    e1.normalize()
    return c, n, e1, n.cross(e1)


def bridge_8_10(bm, arm_ring, hand_ring, extra=(2, 7)):
    """Bridge an 8-column ring to a 10-column one: 8 quads and 2 triangles."""
    n = len(arm_ring)
    c8, nrm, e1, e2 = _basis(arm_ring)

    def ang(v, c):
        d = v.co - c
        return math.atan2(d.dot(e2), d.dot(e1))
    c10 = sum((v.co for v in hand_ring), Vector()) / len(hand_ring)
    HL, ids = list(hand_ring), list(range(len(hand_ring)))

    def turn(L, c):
        s, prev = 0.0, ang(L[0], c)
        for v in L[1:] + [L[0]]:
            a = ang(v, c)
            s += (a - prev + math.pi) % (2 * math.pi) - math.pi
            prev = a
        return s
    if turn(HL, c10) * turn(list(arm_ring), c8) < 0:
        HL, ids = HL[::-1], ids[::-1]
    a0 = ang(arm_ring[0], c8)
    best = min(range(len(HL)), key=lambda i: abs((ang(HL[i], c10) - a0 + math.pi) % (2 * math.pi) - math.pi))
    HL, ids = HL[best:] + HL[:best], ids[best:] + ids[:best]
    while ids[0] in extra:
        HL, ids = HL[1:] + HL[:1], ids[1:] + ids[:1]
    m, j = [], 0
    for i, vid in enumerate(ids):
        if i and vid not in extra:
            j = (j + 1) % n
        m.append(j)
    faces = []
    for i in range(len(HL)):
        i2 = (i + 1) % len(HL)
        a, b = m[i], m[i2]
        try:
            if a == b:
                faces.append(bm.faces.new([HL[i], HL[i2], arm_ring[a]]))
            else:
                faces.append(bm.faces.new([HL[i], HL[i2], arm_ring[b], arm_ring[a]]))
        except ValueError:
            pass
    return faces


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    char_path, out_blend = argv[0], argv[1]
    report = argv[argv.index('--report') + 1] if '--report' in argv else None
    C = json.load(open(char_path))
    B = Body(C)
    head = B.build_head()
    trunk = B.build_trunk(head)
    B.build_legs(trunk)
    B.build_arms(trunk)
    B.finish(out_blend, report)


if __name__ == '__main__':
    main()
