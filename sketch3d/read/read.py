"""Read the part map into char.json: calibration, landmarks, stations, garments.

    py -m sketch3d.read.read <parts.png> <sheet.json> <outdir> [--brief brief.json]
"""
import sys, os, json, time, re
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sketch3d.read import masks, geometry as G, landmarks as LM, arms as AR, side as SD, stations as ST, hair as HR, face as FC
from sketch3d.lexicon import is_leg_garment


TOL_MM, GAP_MM = 4.0, 16.0
THICK_MM = {'jacket': 8.0, 'coat': 8.0, 'shirt': 4.0, 'blouse': 4.0, 'pinafore': 4.0, 'dress': 4.0, 'vest': 4.0, 'glove': 2.0, 'sock': 2.0, 'cuff': 3.0, 'default': 4.0}


def thick_mm(cname):
    """Cloth thickness for a class, the longest key found in the name wins."""
    hits = [k for k in THICK_MM if k != 'default' and k in cname]
    return THICK_MM[max(hits, key=len)] if hits else THICK_MM['default']


def merge_tags(tags):
    """Merge tags of rows within 4 px into the first row."""
    out = {}
    last = None
    for r in sorted(tags):
        if last is not None and r - last < 4:
            out[last] = sorted(set(out[last] + tags[r]))
        else:
            out[r] = list(tags[r])
            last = r
    return out


def trunk_body(st, lm, garments, roles, F, log=print):
    """Fit the trunk under the clothes from the outline and an anatomical taper."""
    r_sh, r_ap, r_wa, r_hip = lm['shoulder']['row'], lm['armpit']['row'], lm['waist']['row'], lm['hip']['row']
    skirt = [g for g, e in garments.items() if e['kind'] == 'skirt']
    # cloth thickness at a row: all torso garments over it, stacked
    def thick_at(r):
        t = 0.0
        for g, e in garments.items():
            if e['kind'] in ('torso', 'trousers', 'skirt') and e['rows'][0] <= r <= e['rows'][1]:
                t += thick_mm(g)
        return t
    def val(r, key):
        rows = [q['row'] for q in st if q.get(key) is not None]
        vals = [q[key] for q in st if q.get(key) is not None]
        return float(np.interp(r, rows, vals)) if rows else None
    w_wa, d_wa = val(r_wa, 'w_mm'), val(r_wa, 'd_mm')
    if skirt:
        w_hip, d_hip, hip_src = w_wa * 1.25, d_wa * 1.15, 'CONV: hip = 1.25 x waist under a skirt'
    else:
        w_hip, d_hip, hip_src = val(r_hip, 'w_mm') - 2 * thick_at(r_hip), val(r_hip, 'd_mm') - 2 * thick_at(r_hip), 'outline - 2 x thickness'
    r_neck = lm['neck']['row']
    w_neck = val(r_neck, 'w_mm')
    d_neck = val(r_neck, 'd_mm')
    line = [q for q in st if q['row'] >= r_sh and q['w_mm'] is not None and q['w_mm'] >= 2 * w_neck]
    r_line = line[0]['row'] if line else r_sh
    w_line = (line[0]['w_mm'] - 2 * thick_at(r_line)) if line else w_neck
    d_line = (line[0]['d_mm'] - 2 * thick_at(r_line)) if line and line[0]['d_mm'] is not None else d_neck
    keep = []
    for q in st:
        r = q['row']
        t = thick_at(r)
        if q['w_mm'] is None:
            continue
        if r >= r_hip and q['w_mm'] < 0.5 * (val(r_hip, 'w_mm') or 1e9):
            continue  # the hem point at the crotch, not a trunk row
        keep.append(q)
        if r <= r_neck or (r < r_line and 'neck' in q['tags']):
            q['w_body_mm'], q['w_src'] = q['w_mm'], 'skin/neck'
            q['d_body_mm'], q['d_src'] = q['d_mm'], q['side_src']
            continue
        if r < r_line:
            u = (r - r_neck) / max(1.0, r_line - r_neck)
            q['w_body_mm'], q['w_src'] = round(w_neck + (w_line - w_neck) * u, 1), 'CONV: trapezius slope neck -> shoulder line (row %d)' % r_line
            q['d_body_mm'], q['d_src'] = round(d_neck + (d_line - d_neck) * u, 1) if d_neck is not None and d_line is not None else q['d_mm'], 'CONV: trapezius slope'
            continue
        if r < r_ap:
            q['w_body_mm'], q['w_src'] = round(q['w_mm'] - 2 * t, 1), 'outline - 2 x %.0f mm (deltoid)' % t
            q['d_body_mm'], q['d_src'] = round(q['d_mm'] - 2 * t, 1) if q['d_mm'] is not None else None, 'outline - 2 x thickness'
            continue
        if r <= r_wa:
            u = (r - r_ap) / max(1.0, r_wa - r_ap)
            tw, td = w_wa * (1.15 - 0.15 * u), d_wa * (1.10 - 0.10 * u)
        else:
            u = (r - r_wa) / max(1.0, r_hip - r_wa)
            u = min(1.0, max(0.0, u))
            tw, td = w_wa + (w_hip - w_wa) * u, d_wa + (d_hip - d_wa) * u
        ow = q['w_mm'] - 2 * t
        q['w_body_mm'] = round(min(ow, tw), 1)
        q['w_src'] = 'outline - 2 x %.0f mm' % t if ow <= tw else ('taper (waist %.0f, hip %.0f: %s)' % (w_wa, w_hip, hip_src))
        if q['d_mm'] is not None and d_wa is not None:
            od = q['d_mm'] - 2 * t
            q['d_body_mm'] = round(min(od, td), 1)
            q['d_src'] = ('outline - 2 x thickness' if od <= td else 'taper') + ' [' + q['side_src'] + ']'
        else:
            q['d_body_mm'], q['d_src'] = None, 'missing'
    # the ribcage only narrows going down: cap a station wider than the one above
    r_sh = lm['shoulder']['row'] if 'shoulder' in lm else None
    if r_sh is not None:
        cand_sh = [q for q in keep if r_sh <= q['row'] <= r_ap and q.get('w_body_mm') and (q.get('w_src') or '').startswith('outline')]
        if cand_sh:
            r_sh = max(cand_sh, key=lambda q: q['w_body_mm'])['row']
        prev = None
        for q in sorted(keep, key=lambda q: q['row']):
            if q['row'] <= r_sh or q['row'] > r_wa:
                prev = q if q['row'] > r_sh else prev
                continue
            if prev is not None and q.get('w_body_mm') and prev.get('w_body_mm') and q['w_body_mm'] > prev['w_body_mm']:
                q['w_body_mm'] = prev['w_body_mm']
                q['w_src'] += '; capped: no wider than the ring above (ribcage narrows to the waist)'
                if q.get('d_body_mm') and prev.get('d_body_mm') and q['d_body_mm'] > prev['d_body_mm']:
                    q['d_body_mm'] = prev['d_body_mm']
            prev = q
    return keep


def default_roles(names, brief):
    roles = {k: list(v) for k, v in (brief or {}).items() if isinstance(v, list)}
    roles.setdefault('skin', [n for n in names if 'skin' in n])
    roles.setdefault('hair', [n for n in names if 'hair' in n])
    roles.setdefault('shoe', [n for n in names if n in ('boot', 'sneaker', 'shoe', 'boots', 'sneakers')])
    roles.setdefault('sole', [n for n in names if 'sole' in n])
    roles.setdefault('accessory', [n for n in names if n in ('phones', 'bow', 'choker', 'collar', 'headphones')])
    return roles


def main(argv):
    png, js, out = argv[0], argv[1], argv[2]
    brief = json.load(open(argv[argv.index('--brief') + 1])) if '--brief' in argv else {}
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'read.log'), 'w', encoding='utf-8')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        logf.write(s + '\n')
    S = json.load(open(js))
    img = np.asarray(Image.open(png).convert('RGB')).astype(np.float32)
    t0 = time.time()
    L = masks.classify(img, S)
    islands = masks.absorb_islands(L)
    log('[read] classes %s, %d small islands absorbed (%.1fs)' % (list(L.ids), len(islands), time.time() - t0))
    roles = default_roles(list(L.ids), brief)
    log('[read] roles', json.dumps(roles))
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    mid = float(pan['front']['midline'])
    side_sign = 1.0 if pan['side'].get('sign', 1) == 1 else -1.0
    log('[read] landmarks (front panel rules):')
    # stations only where the contour turns more than TOL_MM off the chord
    global TOL_MM, GAP_MM
    lm0 = LM.front_landmarks(L, S, roles, log=lambda *a, **k: None)
    height_m = F.z(lm0['crown']['row'])
    TOL_MM = max(4.0, 3.5 * height_m)
    GAP_MM = max(16.0, 10.0 * height_m)
    log('[read] vertex budget: contour tolerance %.1f mm, ring gap %.0f mm' % (TOL_MM, GAP_MM))
    lm = LM.front_landmarks(L, S, roles, log=log)
    prof = lm['_profile']
    T, X0, X1 = prof['T'], prof['X0'], prof['X1']
    arms = {}
    arm_regions = {}
    for sd in ('R', 'L'):
        a, reg = AR.arm_chain(L, S, lm, roles, sd, log=log)
        arms[sd] = a
        arm_regions[sd] = reg
    log('[read] side panel:')
    face = SD.face_profile(L, S, lm, roles, log=log)
    dcol, dwhy = SD.datum(L, S, lm, roles, log=log)
    log('  datum col %.1f (%s)' % (dcol, dwhy))
    c0s, c1s = L.panels['side']
    c0f, c1f = L.panels['front']
    trunk_classes = [n for n in L.ids if n not in roles['hair'] + roles['accessory'] + roles['shoe'] + roles['sole']]
    r_sh, r_ap, r_cr = lm['shoulder']['row'], lm['armpit']['row'], lm['crotch']['row']
    tip_row = int(max(arms[s]['landmarks']['tip']['row'] for s in arms))
    wrist_row = int(min(arms[s]['landmarks']['wrist']['row'] for s in arms))
    occl_trunk = set(range(r_ap, tip_row + 1))
    skin_front = L.mask(roles['skin'], 'front')
    Tskin = G.runs_table(skin_front, c0f, c1f, r0=lm['chin']['row'], r1=lm['shoulder']['row'] + 1, min_w=3)
    front_trunk_neck = {}
    for r, runs in Tskin.items():
        cs = G.central(runs, mid)
        if cs is not None and abs(0.5 * (cs[0] + cs[1]) - mid) < 0.05 * (cal['floor_row'] - lm['crown']['row']):
            front_trunk_neck[r] = (float(cs[0]), float(cs[1]))
    if front_trunk_neck:
        wn = front_trunk_neck.get(lm['neck']['row'], None)
        wn = (wn[1] - wn[0]) if wn else float(np.median([b - a for a, b in front_trunk_neck.values()]))
        front_trunk_neck = {r: v for r, v in front_trunk_neck.items() if v[1] - v[0] >= 0.8 * wn}
    side_trunk = SD.depth_profile(L, S, trunk_classes, lm['neck']['row'], r_cr, occl_trunk, log=log)
    # keep the back edge in rows the arm hides
    raw_trunk = SD.depth_profile(L, S, trunk_classes, lm['neck']['row'], r_cr, set(), log=log)
    for r in side_trunk:
        if side_trunk[r][2] == 'CONV' and r in raw_trunk:
            side_trunk[r] = (side_trunk[r][0], raw_trunk[r][1], 'CONV-front')
    side_neck = SD.depth_profile(L, S, roles['skin'], lm['chin']['row'], lm['shoulder']['row'] + 1, set(), log=log)
    hair_side = L.mask(roles['hair'], 'side')
    neck_rows = [r for r in side_trunk if r <= lm['shoulder']['row']]
    for r in neck_rows:
        if r in side_neck and (side_neck[r][1] - side_neck[r][0]) < 0.5 * (side_trunk[r][1] - side_trunk[r][0]) + 40                 and not hair_side[r, int(side_neck[r][0]):int(side_neck[r][1])].any():
            side_trunk[r] = (side_neck[r][0], side_neck[r][1], 'measured-skin')
        elif r in front_trunk_neck:
            # covered neck: round, d = w
            xl, xr = front_trunk_neck[r]
            w = xr - xl
            side_trunk[r] = (dcol - 0.5 * w, dcol + 0.5 * w, 'CONV-neck: d = w')
    rows_trunk = [r for r in sorted(T) if lm['neck']['row'] <= r < r_cr]
    body = L.figure('front') & ~L.mask(roles['hair'] + roles['accessory'], 'front')
    Tb = G.runs_table(body, c0f, c1f, r0=lm['neck']['row'], r1=r_cr, min_w=3)
    front_trunk = {}
    x_sh = (X0[r_sh], X1[r_sh])
    x_ap = (X0[r_ap], X1[r_ap])
    for r in rows_trunk:
        if r not in Tb:
            continue
        c = G.central(Tb[r], mid)
        if c is None:
            continue
        xl, xr = float(c[0]), float(c[1])
        if r <= r_sh and r in front_trunk_neck:
            xl, xr = front_trunk_neck[r]
        if r_sh < r < r_ap:
            t = (r - r_sh) / max(1, r_ap - r_sh)
            xl = x_sh[0] + (x_ap[0] - x_sh[0]) * t
            xr = x_sh[1] + (x_ap[1] - x_sh[1]) * t
        front_trunk[r] = (xl, xr)
    # smooth the trunk over 15 mm, pockets and lapels are cloth
    ksm = max(2, int(0.015 * cal['ppm']))
    front_trunk = {r: (G.smooth1({q: v[0] for q, v in front_trunk.items()}, ksm)[r], G.smooth1({q: v[1] for q, v in front_trunk.items()}, ksm)[r]) for r in front_trunk}
    side_trunk = {r: (G.smooth1({q: v[0] for q, v in side_trunk.items()}, ksm)[r], G.smooth1({q: v[1] for q, v in side_trunk.items()}, ksm)[r], side_trunk[r][2]) for r in side_trunk}
    curves = [{r: v[0] for r, v in front_trunk.items()}, {r: v[1] for r, v in front_trunk.items()},
              {r: v[0] for r, v in side_trunk.items()}, {r: v[1] for r, v in side_trunk.items()}]
    tags = {}
    forced = []
    for k in ('neck', 'shoulder', 'armpit', 'waist', 'hip', 'crotch'):
        forced.append(lm[k]['row']); tags.setdefault(lm[k]['row'], []).append(k)
    cb = ST.class_boundaries(L, 'front', rows_trunk, lambda r: mid)
    for r, a, b in cb:
        forced.append(r); tags.setdefault(r, []).append('%s|%s' % (a, b))
    rows_all = [r for r in rows_trunk if all(r in c for c in curves)]
    picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=TOL_MM, min_gap_mm=GAP_MM)
    st_trunk = ST.table(picked, F, front_trunk, side_trunk, dcol, side_sign, mid, tags=merge_tags(tags))
    log('[read] trunk: %d stations (%d forced, %d from contour turns)' % (len(st_trunk), len(set(forced)), len(dp)))
    legs = prof['legs']
    leg_classes = [n for n in L.ids if n not in roles['hair'] + roles['accessory']]
    hand_rows = set()
    skin_side = L.mask(roles['skin'], 'side')
    for r in range(wrist_row, tip_row + 1):
        if skin_side[r, c0s:c1s].any():
            hand_rows.add(r)
    r_ank = int(round(np.mean([lm['ankle_R']['row'], lm['ankle_L']['row']])))
    side_leg = SD.depth_profile(L, S, leg_classes, r_cr, r_ank + 1, hand_rows, log=log)
    raw_leg = SD.depth_profile(L, S, leg_classes, r_cr, r_ank + 1, set(), log=log)
    for r in side_leg:
        if side_leg[r][2] == 'CONV' and r in raw_leg:
            side_leg[r] = (side_leg[r][0], raw_leg[r][1], 'CONV-front')
    st_legs = {}
    for sd in ('R', 'L'):
        LR = legs[sd]
        rows_leg = [r for r in sorted(LR) if r_cr <= r <= lm['ankle_' + sd]['row']]
        fr = {}
        for r in rows_leg:
            a, b = LR[r]
            fr[r] = (float(a), float(b))
        # R is image-left: outer = a, inner = b
        curves = [G.smooth1({r: v[0] for r, v in fr.items()}, 2), G.smooth1({r: v[1] for r, v in fr.items()}, 2),
                  G.smooth1({r: v[0] for r, v in side_leg.items()}, 2), G.smooth1({r: v[1] for r, v in side_leg.items()}, 2)]
        tags = {}
        forced = []
        for k in ('crotch', 'knee_' + sd, 'ankle_' + sd):
            forced.append(lm[k]['row']); tags.setdefault(lm[k]['row'], []).append(k.split('_')[0])
        cb = ST.class_boundaries(L, 'front', rows_leg, lambda r: 0.5 * (LR[r][0] + LR[r][1]))
        for r, a, b in cb:
            forced.append(r); tags.setdefault(r, []).append('%s|%s' % (a, b))
        rows_all = [r for r in rows_leg if all(r in c for c in curves)]
        picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=TOL_MM, min_gap_mm=GAP_MM)
        st_legs[sd] = ST.table(picked, F, fr, side_leg, dcol, side_sign, mid, tags=tags)
        log('[read] leg %s: %d stations (%d forced, %d turns)' % (sd, len(picked), len(set(forced)), len(dp)))
    st_shoes = {}
    for sd in ('R', 'L'):
        LR = legs[sd]
        rows_sh = [r for r in sorted(LR) if lm['ankle_' + sd]['row'] <= r < int(cal['floor_row'])]
        fr = {r: (float(LR[r][0]), float(LR[r][1])) for r in rows_sh}
        side_sh = SD.depth_profile(L, S, roles['shoe'] + roles['sole'] + roles['skin'], rows_sh[0], rows_sh[-1] + 1, set(), log=log) if rows_sh else {}
        curves = [G.smooth1({r: v[0] for r, v in fr.items()}, 2), G.smooth1({r: v[1] for r, v in fr.items()}, 2),
                  G.smooth1({r: v[0] for r, v in side_sh.items()}, 2), G.smooth1({r: v[1] for r, v in side_sh.items()}, 2)]
        tags = {}
        forced = [rows_sh[0], rows_sh[-1]] if rows_sh else []
        if 'sole_top_' + sd in lm:
            forced.append(lm['sole_top_' + sd]['row']); tags.setdefault(lm['sole_top_' + sd]['row'], []).append('sole_top')
        rows_all = [r for r in rows_sh if all(r in c for c in curves)]
        picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=0.6 * TOL_MM, min_gap_mm=0.6 * GAP_MM) if rows_all else ([], [])
        st_shoes[sd] = ST.table(picked, F, fr, side_sh, dcol, side_sign, mid, tags=tags)
    rows_head = [r for r in sorted(T) if lm['crown']['row'] <= r <= lm['chin']['row']]
    from scipy import ndimage as ndi
    cap_r = max(3, int(0.03 * (cal['floor_row'] - lm['crown']['row'])))
    hair_f = L.mask(roles['hair'], 'front')
    cap_f = ndi.binary_opening(hair_f, structure=np.ones((3, 3), bool), iterations=cap_r)
    headm = L.mask(roles['skin'] + [n for n in L.ids if n in ('ear',)], 'front') | cap_f
    Th = G.runs_table(headm, c0f, c1f, r0=rows_head[0], r1=rows_head[-1] + 1, min_w=3)
    fr_head, fr_face = {}, {}
    skin_f = L.mask(roles['skin'], 'front')
    Tsk = G.runs_table(skin_f, c0f, c1f, r0=rows_head[0], r1=rows_head[-1] + 1, min_w=3)
    for r in rows_head:
        if r in Th:
            near = [ab for ab in Th[r] if abs(0.5 * (ab[0] + ab[1]) - mid) < 0.25 * (cal['floor_row'] - lm['crown']['row'])]
            if near:
                fr_head[r] = (float(min(ab[0] for ab in near)), float(max(ab[1] for ab in near)))
        if r in Tsk:
            near = [ab for ab in Tsk[r] if abs(0.5 * (ab[0] + ab[1]) - mid) < 0.15 * (cal['floor_row'] - lm['crown']['row'])]
            if near:
                fr_face[r] = (float(min(ab[0] for ab in near)), float(max(ab[1] for ab in near)))
    hair_s = L.mask(roles['hair'], 'side')
    cap_s = ndi.binary_opening(hair_s, structure=np.ones((3, 3), bool), iterations=cap_r)
    c0s_, c1s_ = L.panels['side']
    capm_s = L.mask(roles['skin'], 'side') | cap_s
    Tcs = G.runs_table(capm_s, c0s_, c1s_, r0=rows_head[0], r1=rows_head[-1] + 1, min_w=3)
    side_head = {}
    for r, runs in Tcs.items():
        side_head[r] = (float(min(q[0] for q in runs)), float(max(q[1] for q in runs)), 'measured')
    lm['hair_cap_top'] = dict(row=int(min(side_head) if side_head else rows_head[0]), z=round(F.z(min(side_head) if side_head else rows_head[0]), 5),
                              rule='top of the hair CAP (hair mask opened by %d px: spikes and locks removed)' % cap_r, source='measured')
    side_face = SD.depth_profile(L, S, roles['skin'], rows_head[0], rows_head[-1] + 21, set(), log=log)
    curves = [G.smooth1({r: v[0] for r, v in fr_head.items()}, 2), G.smooth1({r: v[1] for r, v in fr_head.items()}, 2),
              G.smooth1({r: v[0] for r, v in side_head.items()}, 2), G.smooth1({r: v[1] for r, v in side_head.items()}, 2)]
    tags = {}
    forced = [lm['crown']['row'], lm['head_widest']['row'], lm['chin']['row']]
    tags.setdefault(lm['chin']['row'], []).append('chin')
    for k in ('brow', 'nose_tip', 'lip'):
        if k in face:
            forced.append(face[k]['row']); tags.setdefault(face[k]['row'], []).append(k)
    rows_all = [r for r in rows_head if all(r in c for c in curves)]
    picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=0.6 * TOL_MM, min_gap_mm=0.6 * GAP_MM)
    st_head = ST.table(picked, F, fr_head, side_head, dcol, side_sign, mid, tags=tags)
    face_w = {r: v[1] - v[0] for r, v in fr_face.items()}
    r_temple = max(face_w, key=face_w.get) if face_w else lm['head_widest']['row']
    hairline = min(fr_face) if fr_face else lm['crown']['row']
    lm['hairline'] = dict(row=int(hairline), z=round(F.z(hairline), 5), rule='first face-skin row at the midline', source='measured')
    lm['temple'] = dict(row=int(r_temple), z=round(F.z(r_temple), 5), rule='widest face-skin row', source='measured')
    face_d = {r: (v[1] - v[0]) for r, v in side_face.items() if r in fr_face}
    skull = dict(w_mm=round(F.mm(face_w[r_temple]), 1) if face_w else None,
                 d_mm=round(F.mm(max(face_d.values())), 1) if face_d else None,
                 d_src='side skin extent through the face rows (hair may hide the back: check against 1.05 x w)',
                 hairline_z=round(F.z(hairline), 4), cap_top_z=lm['hair_cap_top']['z'], crown_z=round(F.z(lm['crown']['row']), 4),
                 chin_z=lm['chin']['z'])
    log('[read] skull: w %s d %s hairline z %.3f cap top z %.3f crown z %.3f' % (skull['w_mm'], skull['d_mm'], skull['hairline_z'], skull['cap_top_z'], skull['crown_z']))
    st_face = ST.table([r for r in picked if r in fr_face], F, fr_face, side_face, dcol, side_sign, mid, tags=tags)
    log('[read] head: %d stations; face rows %d' % (len(st_head), len(st_face)))
    st_arms = {}
    for sd, a in arms.items():
        P = np.array(a['axis_px']); s = np.array(a['s_px']); w = np.array(a['width_px'])
        idx = list(range(len(P)))
        curve = {i: float(w[i]) for i in idx}
        forced = [a['landmarks'][k]['i'] for k in ('root', 'elbow', 'wrist', 'hand_widest', 'tip')]
        tagsA = {a['landmarks'][k]['i']: [k] for k in ('root', 'elbow', 'wrist', 'hand_widest', 'tip')}
        cls = a['cls']
        for i in range(1, len(cls)):
            if cls[i] != cls[i - 1]:
                forced.append(i); tagsA.setdefault(i, []).append('%s|%s' % (cls[i - 1], cls[i]))
        step_px = s[1] - s[0] if len(s) > 1 else 6.0
        dp = G.dp_rows(idx, [curve], tol_px=0.9 / 1000.0 * cal['ppm'], min_gap=2, corner_deg=10.0)
        gap = max(1, int(10.0 / 1000.0 * cal['ppm'] / step_px))
        picked = sorted(set(int(f) for f in forced))
        for i in dp:
            if all(abs(i - f) >= gap for f in picked):
                picked.append(i)
        picked = sorted(set(picked))
        rowsA = []
        for i in picked:
            r, c = P[i]
            rowsA.append(dict(i=int(i), s_m=round(float(F.m(s[i])), 4), row=float(r), col=float(c), z=round(F.z(r), 5),
                              x_m=round(float((c - mid) / cal['ppm']), 5), w_mm=round(F.mm(w[i]), 1),
                              d_mm=round(F.mm(w[i]), 1), d_src='CONV: d = w (arm)', cls=cls[i], tags=tagsA.get(i, [])))
        st_arms[sd] = dict(angle_deg=a['angle_deg'], length_m=a['length_m'], landmarks=a['landmarks'], stations=rowsA)
        y_hand = None
        try:
            r_w = int(a['landmarks']['wrist']['row']); r_t = int(a['landmarks']['tip']['row'])
            skin_s = L.mask(roles['skin'], 'side')
            c0s, c1s = L.panels['side']
            cents = []
            for r in range(min(r_w, r_t), max(r_w, r_t) + 1):
                runs = [q for q in G.row_runs(skin_s[r, c0s:c1s]) if q[1] - q[0] >= 3]
                if not runs:
                    continue
                runs = [(a_ + c0s, b_ + c0s) for a_, b_ in runs]
                if len(runs) > 1:
                    widest = max(runs, key=lambda q: q[1] - q[0])
                    if widest[1] - widest[0] > 0.06 * cal['ppm']:
                        runs = [q for q in runs if q is not widest]
                cents += [0.5 * (q[0] + q[1]) for q in runs]
            if len(cents) >= 0.3 * abs(r_t - r_w):
                y_hand = float(np.median(cents))
                st_arms[sd]['y_hand_m'] = round((y_hand - dcol) / cal['ppm'] * side_sign, 4)
                st_arms[sd]['y_hand_src'] = 'side panel: median centre of the hand skin runs, rows %d..%d' % (min(r_w, r_t), max(r_w, r_t))
        except Exception as e_:
            log('[read] arm %s: no side-panel hand position (%s)' % (sd, e_))
        log('[read] arm %s: %d stations%s' % (sd, len(rowsA), ('; hand y %.3f (side panel)' % st_arms[sd]['y_hand_m']) if y_hand is not None else ''))
    garments = {}
    Hpx = cal['floor_row'] - lm['crown']['row']
    for cname in L.ids:
        if cname in roles['skin'] + roles['hair']:
            continue
        m = L.mask(cname, 'front')
        comp, ncomp = ndi.label(m, structure=np.ones((3, 3), int))
        if ncomp == 0 and cname in roles['accessory'] and 'back' in L.panels:
            # an accessory hidden in front (a backpack) is read from the back view
            mb = L.mask(cname, 'back')
            if mb.any():
                rb = np.nonzero(mb.any(1))[0]
                garments[cname] = dict(kind='accessory', rows=[int(rb.min()), int(rb.max())], z=[round(F.z(rb.max()), 4), round(F.z(rb.min()), 4)], back_only=True)
                log('[read] garment %-10s %-9s rows %d..%d (back view only)' % (cname, 'accessory', rb.min(), rb.max()))
            continue
        if ncomp == 0:
            continue
        sizes = ndi.sum(np.ones_like(m), comp, index=np.arange(1, ncomp + 1))
        big = [i + 1 for i in range(ncomp) if sizes[i] >= 0.03 * sizes.sum()]
        m = np.isin(comp, big)
        Tc = G.runs_table(m, c0f, c1f, min_w=3)
        if not Tc:
            continue
        rows_c = sorted(Tc)
        r0c, r1c = rows_c[0], rows_c[-1]
        # a class mostly on the arm is an arm covering (glove, cuff), not a torso shell
        on_arm = sum(int(np.logical_and(m, arm_regions[sd_]).sum()) for sd_ in arm_regions if arm_regions.get(sd_) is not None)
        if cname in roles['accessory']:
            kind = 'accessory'
        elif m.sum() > 0 and on_arm >= 0.6 * m.sum():
            kind = 'arms'
        elif r0c >= r_cr - 0.02 * Hpx:
            kind = 'legs'
        elif r1c > r_cr + 0.05 * Hpx or (is_leg_garment(cname) and r1c > r_cr + 0.01 * Hpx):
            kind = 'trousers'
        else:
            kind = 'torso'
            top_w = np.median([Tc[r][-1][1] - Tc[r][0][0] for r in rows_c[:max(1, len(rows_c) // 10)]])
            bot_w = np.median([Tc[r][-1][1] - Tc[r][0][0] for r in rows_c[-max(1, len(rows_c) // 10):]])
            if r1c >= r_cr - 0.02 * Hpx and bot_w > 1.3 * top_w:
                kind = 'skirt'
        side_c = SD.depth_profile(L, S, [cname], r0c, r1c + 1, set(), log=log)
        entry = dict(kind=kind, rows=[int(r0c), int(r1c)], z=[round(F.z(r1c), 4), round(F.z(r0c), 4)])
        if kind == 'torso' and r1c < lm['neck']['row']:
            entry['painted'] = True
            entry['decal'] = True
            log('[read] garment %s: drawn on the head only - painted on it, no shell' % cname)
        n_arm = sum(len(st_arms[sd_]['stations']) for sd_ in st_arms)
        n_cls = sum(1 for sd_ in st_arms for q_ in st_arms[sd_]['stations'] if q_.get('cls') == cname)
        sleeved = n_arm > 0 and n_cls >= 0.25 * n_arm
        if kind == 'torso':
            entry['sleeved'] = bool(sleeved)
        if kind in ('torso', 'accessory', 'trousers', 'skirt'):
            fr = {}
            for r in [q for q in rows_c if kind != 'trousers' or q < r_cr]:
                cen = X0.get(r), X1.get(r)
                near = [ab for ab in Tc[r] if (cen[0] is not None and ab[1] > cen[0] and ab[0] < cen[1]) or abs(0.5 * (ab[0] + ab[1]) - mid) < 0.12 * Hpx]
                if near:
                    xl_, xr_ = float(min(ab[0] for ab in near)), float(max(ab[1] for ab in near))
                    if kind == 'torso' and sleeved and r_sh < r < r_ap:
                        t_ = (r - r_sh) / max(1, r_ap - r_sh)
                        tmm = thick_mm(cname) / 1000.0 * cal['ppm']
                        xl_ = max(xl_, x_sh[0] + (x_ap[0] - x_sh[0]) * t_ - tmm)
                        xr_ = min(xr_, x_sh[1] + (x_ap[1] - x_sh[1]) * t_ + tmm)
                    fr[r] = (xl_, xr_)
            # the back view adds width the front hides (hair, arms, a cape)
            if kind in ('torso', 'skirt') and 'back' in L.panels:
                c0b, c1b = L.panels['back']
                midb = float(pan['back']['midline'])
                mb = L.mask(cname, 'back')
                mb[:, :c0b] = False; mb[:, c1b:] = False
                compb, nb_ = ndi.label(mb, structure=np.ones((3, 3), int))
                if nb_:
                    sizes_b = ndi.sum(np.ones_like(mb), compb, index=np.arange(1, nb_ + 1))
                    mb = np.isin(compb, [i + 1 for i in range(nb_) if sizes_b[i] >= 0.03 * sizes_b.sum()])
                Tb = G.runs_table(mb, c0b, c1b, min_w=3)
                n_wid, n_new = 0, 0
                for r in sorted(Tb):
                    if kind == 'skirt' and r >= r_cr:
                        continue
                    near_b = [ab for ab in Tb[r] if abs(0.5 * (ab[0] + ab[1]) - midb) < 0.12 * Hpx]
                    if not near_b:
                        continue
                    bl, br = float(min(ab[0] for ab in near_b)), float(max(ab[1] for ab in near_b))
                    if sleeved and r_sh < r < r_ap:
                        t_ = (r - r_sh) / max(1, r_ap - r_sh)
                        tmm = thick_mm(cname) / 1000.0 * cal['ppm']
                        half = 0.5 * ((x_sh[1] - x_sh[0]) + ((x_ap[1] - x_ap[0]) - (x_sh[1] - x_sh[0])) * t_) + tmm
                        bl = max(bl, midb - half); br = min(br, midb + half)
                    wb_ = br - bl
                    if r in fr:
                        wf_ = fr[r][1] - fr[r][0]
                        if 1.05 * wf_ < wb_ <= 1.6 * wf_:
                            cx_ = 0.5 * (fr[r][0] + fr[r][1])
                            fr[r] = (cx_ - 0.5 * wb_, cx_ + 0.5 * wb_)
                            n_wid += 1
                    elif r0c - 0.02 * Hpx <= r <= r1c + 0.15 * Hpx and wb_ >= 12 / 1000.0 * cal['ppm']:
                        fr[r] = (mid - 0.5 * wb_, mid + 0.5 * wb_)
                        n_new += 1
                if n_wid or n_new:
                    log('[read] garment %s: back panel widened %d row(s), added %d row(s) the front does not show' % (cname, n_wid, n_new))
            if fr:
                if kind in ('torso', 'accessory'):
                    entry['rows'] = [int(min(fr)), int(max(fr))]
                    entry['z'] = [round(F.z(max(fr)), 4), round(F.z(min(fr)), 4)]
                ks = max(2, int(0.012 * cal['ppm']))
                fr = {r: (G.smooth1({q: v[0] for q, v in fr.items()}, ks)[r], G.smooth1({q: v[1] for q, v in fr.items()}, ks)[r]) for r in fr}
                side_c = {r: (G.smooth1({q: v[0] for q, v in side_c.items()}, ks)[r], G.smooth1({q: v[1] for q, v in side_c.items()}, ks)[r], side_c[r][2]) for r in side_c}
                curves = [{r: v[0] for r, v in fr.items()}, {r: v[1] for r, v in fr.items()},
                          {r: v[0] for r, v in side_c.items()}, {r: v[1] for r, v in side_c.items()}]
                rows_all = [r for r in sorted(fr) if all(r in c for c in curves)]
                side_missing = len(rows_all) < 3
                if side_missing:
                    # not visible on the side: depth = width
                    curves = curves[:2]
                    rows_all = sorted(fr)
                forced = [rows_all[0], rows_all[-1]] if rows_all else []
                picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=TOL_MM, min_gap_mm=GAP_MM) if len(rows_all) > 3 else (rows_all, [])
                entry['stations'] = ST.table(picked, F, fr, side_c, dcol, side_sign, mid)
                if side_missing:
                    tr_ = [q for q in st_trunk if q.get('w_mm') and q.get('d_mm')]
                    for q in entry['stations']:
                        if q.get('d_mm') is None and q.get('w_mm'):
                            ratio = 1.0
                            if tr_:
                                near_t = min(tr_, key=lambda t_: abs(t_['z'] - q['z']))
                                ratio = min(1.6, max(0.6, near_t['d_mm'] / max(1e-6, near_t['w_mm'])))
                            q['d_mm'] = round(q['w_mm'] * ratio, 1); q['y_mid_mm'] = 0.0; q['side_src'] = 'CONV: no side reading, d = w x body d/w'
                    log("[read] garment %s: no side reading over its rows - depth = width x the body's d/w (CONV)" % cname)
            # open front: two runs either side of the midline
            gap = {}
            for r in [q for q in rows_c if q < r_cr]:
                lf = [ab for ab in Tc[r] if ab[1] <= mid and mid - ab[1] < 0.12 * Hpx]
                rt = [ab for ab in Tc[r] if ab[0] >= mid and ab[0] - mid < 0.12 * Hpx]
                if lf and rt and not any(ab[0] <= mid < ab[1] for ab in Tc[r]):
                    gap[r] = (float(max(ab[1] for ab in lf)), float(min(ab[0] for ab in rt)))
            if len(gap) > 0.3 * len([q for q in rows_c if q < r_cr]):
                gr = sorted(gap)
                entry['open_front'] = dict(rows=[int(gr[0]), int(gr[-1])], gap_mm={int(r): [round(F.mm(gap[r][0] - mid), 1), round(F.mm(gap[r][1] - mid), 1)] for r in gr[::max(1, len(gr) // 12)]})
        if kind in ('legs', 'trousers'):
            entry['sides'] = {}
            for sd in ('R', 'L'):
                fr = {}
                for r in [q for q in rows_c if q >= r_cr]:
                    runs = [ab for ab in Tc[r] if (ab[1] <= mid + 3 if sd == 'R' else ab[0] >= mid - 3) and abs(0.5 * (ab[0] + ab[1]) - mid) < 0.2 * Hpx]
                    if runs:
                        fr[r] = (float(min(ab[0] for ab in runs)), float(max(ab[1] for ab in runs)))
                if not fr:
                    continue
                curves = [G.smooth1({r: v[0] for r, v in fr.items()}, 2), G.smooth1({r: v[1] for r, v in fr.items()}, 2),
                          G.smooth1({r: v[0] for r, v in side_c.items()}, 2), G.smooth1({r: v[1] for r, v in side_c.items()}, 2)]
                rows_all = [r for r in sorted(fr) if all(r in c for c in curves)]
                forced = [rows_all[0], rows_all[-1]] if rows_all else []
                picked, dp = ST.pick_rows(rows_all, curves, forced, F, tol_mm=TOL_MM, min_gap_mm=GAP_MM) if len(rows_all) > 3 else (rows_all, [])
                tab = ST.table(picked, F, fr, side_c, dcol, side_sign, mid)
                tab = [q for q in tab if (q.get('w_mm') or 0) >= 12.0 and (q.get('d_mm') is None or q['d_mm'] >= 12.0)]
                entry['sides'][sd] = tab
        if kind == 'arms':
            entry['sides'] = {}
            for sd in ('R', 'L'):
                tab = [dict(q) for q in st_arms[sd]['stations'] if q.get('cls') == cname and (q.get('w_mm') or 0) >= 8.0]
                if tab:
                    entry['sides'][sd] = tab
                    s_w = st_arms[sd]['landmarks']['wrist']['s_px'] / cal['ppm']
                    entry.setdefault('reaches_hand', {})[sd] = bool(max(q['s_m'] for q in tab) >= s_w + 0.02)
            # a glove is painted on the hand, only its cuff is a tube
            if any(entry.get('reaches_hand', {}).values()) or re.search(r'glove|mitt|gauntlet', cname):
                entry['painted'] = True
        garments[cname] = entry
        log('[read] garment %-10s %-9s rows %d..%d %s' % (cname, kind, r0c, r1c, 'open front' if 'open_front' in entry else ''))
    log('[read] hair:')
    hair = HR.hair_read(L, S, lm, roles, dcol, side_sign, log=log, skull=skull)
    if hair.get('plates') or hair.get('painted'):
        tops = [r['z'] for pn in ('front', 'side') for r in hair.get('cap', {}).get(pn, [])]
        for cname in hair.get('painted', []):
            for pn in ('front', 'side'):
                if pn in L.panels:
                    mp = L.mask([cname], pn)
                    rr = np.nonzero(mp.any(1))[0]
                    if len(rr):
                        tops.append(F.z(rr.min()))
        if tops:
            z_cap = max(tops)
            skull['cap_top_z'] = round(z_cap, 4)
            skull['cap_top_src'] = 'top of the hair on the skull (plates excluded)'
            lm['hair_cap_top'] = dict(row=int(F.row(z_cap)) if hasattr(F, 'row') else lm['hair_cap_top']['row'], z=round(z_cap, 4),
                                      rule='top of the cap classes (the plate above is not the cap)', source='measured')
            log('[read] skull: cap top z %.3f (the plate above is not the cap)' % z_cap)
    log('[read] face features:')
    # the face comes from the art, the part map has none
    art_png = S.get('art')
    if art_png and os.path.exists(art_png):
        img_art = np.asarray(Image.open(art_png).convert('RGB')).astype(np.float32)
        if img_art.shape == img.shape:
            log('[read] face from the art %s' % os.path.basename(art_png))
            face_vec = FC.face_read(img_art, L, S, lm, roles, log=log)
        else:
            face_vec = FC.face_read(img, L, S, lm, roles, log=log)
    else:
        face_vec = FC.face_read(img, L, S, lm, roles, log=log)
    st_trunk = trunk_body(st_trunk, lm, garments, roles, F, log=log)
    # the limb under clothes is the outline minus the stacked cloth
    def cover_t(z, kinds=('trousers', 'legs', 'skirt')):
        t = 0.0
        for g, e in garments.items():
            if e['kind'] in kinds and min(e['z']) - 0.004 <= z <= max(e['z']) + 0.004:
                t += thick_mm(g)
        return t
    legs_g = [(g, e) for g, e in garments.items() if e['kind'] == 'legs']

    def despike(st, keys=(('d_mm', 'y_mid_mm'), ('w_mm', 'x_c_mm')), ratio=1.35, log=log, what='limb'):
        """Replace a station wider than both neighbours by `ratio` with their mean."""
        st = sorted(st, key=lambda q: -q['z'])
        n = 0
        for i in range(1, len(st) - 1):
            a, q, b = st[i - 1], st[i], st[i + 1]
            for k, kc in keys:
                if not (a.get(k) and q.get(k) and b.get(k)):
                    continue
                if q[k] > ratio * max(a[k], b[k]):
                    q[k] = round(0.5 * (a[k] + b[k]), 1)
                    if a.get(kc) is not None and b.get(kc) is not None:
                        q[kc] = round(0.5 * (a[kc] + b[kc]), 1)
                    q.setdefault('tags', []).append('despiked_' + k[0])
                    n += 1
        if n:
            log("[read] %s: %d station value(s) were an occluder's run (spike vs both neighbours), took the neighbours' mean" % (what, n))
        return n

    for side in ('R', 'L'):
        despike([q for q in st_legs[side] if q.get('w_mm') and q.get('d_mm')], what='leg_' + side)
        despike([q for q in st_arms[side]['stations'] if q.get('w_mm') and q.get('d_mm')], what='arm_' + side)
    for side in ('R', 'L'):
        st = [q for q in st_legs[side] if q.get('w_mm') and q.get('d_mm')]
        st.sort(key=lambda q: -q['z'])
        for q in st:
            t = cover_t(q['z'])
            q['w_body_mm'] = round(q['w_mm'] - 2 * t, 1)
            q['d_body_mm'] = round(q['d_mm'] - 2 * t, 1) if q['d_mm'] else None
            q['body_src'] = 'outline - 2 x %.0f mm' % t
        # under a legs garment: no wider than the last station above it
        for g, e in legs_g:
            z_top = max(e['z'])
            above = [q for q in st if q['z'] > z_top + 0.004]
            if not above:
                continue
            ref = above[-1]
            for q in st:
                if q['z'] <= z_top + 0.004 and q['z'] >= min(e['z']) - 0.004:
                    if q['w_body_mm'] > ref['w_body_mm']:
                        q['w_body_mm'] = ref['w_body_mm']
                        q['body_src'] = 'CONV: no wider than above the %s' % g
                    if q.get('d_body_mm') and ref.get('d_body_mm') and q['d_body_mm'] > ref['d_body_mm']:
                        q['d_body_mm'] = ref['d_body_mm']
        # under trousers use a leg template hung from the thigh width
        z_knee = lm['knee_' + side]['z'] if ('knee_' + side) in lm else None
        z_ank = lm['ankle_' + side]['z'] if ('ankle_' + side) in lm else None
        covered = [q for q in st if cover_t(q['z'], kinds=('trousers', 'legs')) > 0]
        if covered and z_knee is not None and z_ank is not None:
            top = max(covered, key=lambda q: q['z'])
            w_top = top['w_body_mm']; d_top = top.get('d_body_mm') or w_top
            z_top = top['z']

            def tmpl(z):
                if z >= z_knee:
                    t = (z_top - z) / max(1e-6, z_top - z_knee)
                    return 1.0 + (0.70 - 1.0) * min(1.0, max(0.0, t))
                t = (z_knee - z) / max(1e-6, z_knee - z_ank)
                t = min(1.0, max(0.0, t))
                if t < 0.30:
                    return 0.70 + (0.84 - 0.70) * (t / 0.30)
                if t < 0.55:
                    return 0.84 + (0.72 - 0.84) * ((t - 0.30) / 0.25)
                return 0.72 + (0.50 - 0.72) * ((t - 0.55) / 0.45)
            for q in st:
                if q['z'] > z_top or cover_t(q['z'], kinds=('trousers', 'legs')) <= 0:
                    continue
                cap_w = w_top * tmpl(q['z']); cap_d = d_top * tmpl(q['z'])
                if q['w_body_mm'] > cap_w:
                    q['w_body_mm'] = round(cap_w, 1); q['body_src'] = 'CONV: leg template under trousers'
                if q.get('d_body_mm') and q['d_body_mm'] > cap_d:
                    q['d_body_mm'] = round(cap_d, 1)
            # give the calf peak its own station so interpolation keeps it
            z_peak = z_knee - 0.30 * (z_knee - z_ank)
            if cover_t(z_peak, kinds=('trousers', 'legs')) > 0 and not any(abs(q['z'] - z_peak) < 0.004 for q in st):
                near = min(st, key=lambda q: abs(q['z'] - z_peak))
                peak = dict(near, z=round(z_peak, 5), row=int(round(F.floor_row - z_peak * F.ppm)) if hasattr(F, 'floor_row') else near['row'],
                            w_body_mm=round(min(near['w_body_mm'] / max(1e-6, tmpl(near['z'])) * 0.84, w_top * 0.84), 1),
                            body_src='CONV: calf peak (leg template)', tags=list(near.get('tags', [])) + ['calf'])
                if near.get('d_body_mm'):
                    peak['d_body_mm'] = round(min(near['d_body_mm'] / max(1e-6, tmpl(near['z'])) * 0.84, d_top * 0.84), 1)
                st_legs[side].append(peak)
                st_legs[side].sort(key=lambda q: -q['z'])
                st.append(peak); st.sort(key=lambda q: -q['z'])
        for g, e in legs_g:
            z_top = max(e['z'])
            above = [q for q in st if q['z'] > z_top + 0.004]
            if not above:
                continue
            ref = above[-1]
            for q in st:
                if min(e['z']) - 0.004 <= q['z'] <= z_top + 0.004:
                    if q['w_body_mm'] > ref['w_body_mm']:
                        q['w_body_mm'] = ref['w_body_mm']; q['body_src'] = 'CONV: no wider than above the %s' % g
                    if q.get('d_body_mm') and ref.get('d_body_mm') and q['d_body_mm'] > ref['d_body_mm']:
                        q['d_body_mm'] = ref['d_body_mm']
        bad = [q for q in st if q['w_body_mm'] < 15.0 or (q.get('d_body_mm') is not None and q['d_body_mm'] < 15.0)]
        if bad:
            st_legs[side] = [q for q in st_legs[side] if q not in bad]
            st = [q for q in st if q not in bad]
            log('[read] leg %s: %d station(s) too narrow for a limb dropped (z %s)' % (side, len(bad), ', '.join('%.3f' % q['z'] for q in bad)))
        log('[read] leg %s body: %s' % (side, ', '.join('%.2f:%.0f' % (q['z'], q['w_body_mm']) for q in st[::3])))
    for side in ('R', 'L'):
        A = st_arms[side]
        for q in A['stations']:
            t = thick_mm(q['cls']) if q['cls'] not in roles['skin'] else 0.0
            q['w_body_mm'] = round(q['w_mm'] - 2 * t, 1)
            q['d_body_mm'] = round(q['d_mm'] - 2 * t, 1)
        # wrist: the narrowest bare sample before the hand
        try:
            s_hw = A['landmarks']['hand_widest']['s_px'] / cal['ppm']
            covered_s = [q['s_m'] for q in A['stations'] if q['cls'] not in roles['skin'] and q['s_m'] < s_hw - 0.005]
            s_from = max(covered_s) if covered_s else 0.75 * s_hw
            bare = [q for q in A['stations'] if q['cls'] in roles['skin'] and s_from <= q['s_m'] <= s_hw - 0.005 and q['w_body_mm'] > 8]
            if bare and (bare[-1]['s_m'] - bare[0]['s_m']) > 0.12:
                bare = [q for q in bare if q['s_m'] >= bare[-1]['s_m'] - 0.12]
            if len(bare) >= 2:
                w_wr = min(q['w_body_mm'] for q in bare)
                for q in bare:
                    if q['w_body_mm'] > w_wr:
                        q['w_body_mm'] = w_wr; q['d_body_mm'] = min(q['d_body_mm'], w_wr); q['w_src'] = 'CONV: wrist, no wider than the narrowest bare sample above the hand'
                # under a sleeve use an arm template hung from the wrist width
                s_wr = min(q['s_m'] for q in bare if q['w_body_mm'] <= w_wr + 0.5)
                s_el = A['landmarks']['elbow']['s_px'] / cal['ppm'] if 'elbow' in A['landmarks'] else 0.5 * s_wr
                for q in A['stations']:
                    if q['cls'] in roles['skin'] or q['s_m'] > s_wr:
                        continue
                    if q['s_m'] <= s_el:
                        k = 1.7 + (1.4 - 1.7) * (q['s_m'] / max(1e-6, s_el))
                    else:
                        k = 1.4 + (1.0 - 1.4) * ((q['s_m'] - s_el) / max(1e-6, s_wr - s_el))
                    cap = w_wr * k
                    if q['w_body_mm'] > cap:
                        q['w_body_mm'] = round(cap, 1); q['d_body_mm'] = round(min(q['d_body_mm'], cap), 1)
                        q['w_src'] = 'CONV: arm template under the sleeve (%.2f x wrist)' % k
        except KeyError:
            pass

    for q in st_trunk:
        log('  trunk row %4d z %.3f outline w %6s d %6s -> body w %6s (%s) d %6s' % (q['row'], q['z'], q['w_mm'], q['d_mm'], q['w_body_mm'], q['w_src'], q['d_body_mm']))
    # cloth within 12 mm of the body is painted, not built
    for cname, entry in garments.items():
        kind = entry['kind']
        if cname in roles.get('shoe', []) + roles.get('sole', []):
            continue  # a shoe has its own volume around the foot stub
        if kind in ('torso', 'legs', 'arms') and 'open_front' not in entry and not entry.get('painted'):
            offs = []
            if kind == 'torso':
                for q in entry.get('stations', []):
                    if not q.get('w_mm'):
                        continue
                    b = min(st_trunk, key=lambda t: abs(t['z'] - q['z']))
                    if abs(b['z'] - q['z']) < 0.03 and b.get('w_body_mm'):
                        offs.append(0.5 * (q['w_mm'] - b['w_body_mm']))
            elif kind == 'legs':
                for sd_, tab in entry.get('sides', {}).items():
                    for q in tab:
                        if not q.get('w_mm'):
                            continue
                        cand = [t for t in st_legs.get(sd_, []) if t.get('w_body_mm')]
                        if not cand:
                            continue
                        b = min(cand, key=lambda t: abs(t['z'] - q['z']))
                        if abs(b['z'] - q['z']) < 0.03:
                            offs.append(0.5 * (q['w_mm'] - b['w_body_mm']))
            else:
                for sd_, tab in entry.get('sides', {}).items():
                    for q in tab:
                        if q.get('w_mm') and q.get('w_body_mm'):
                            offs.append(0.5 * (q['w_mm'] - q['w_body_mm']))
            if len(offs) >= 2 and float(np.percentile(offs, 90)) < 12.0:
                entry['painted'] = True
                entry['tight_mm'] = round(float(np.percentile(offs, 90)), 1)
                log('[read] garment %s: tight (%.0f mm off the body at most) - painted on the body, no shell' % (cname, entry['tight_mm']))
    char = dict(
        sheet=os.path.abspath(png), sheet_json=os.path.abspath(js), calibration=cal, palette=S['palette'],
        art=(os.path.abspath(S['art']) if S.get('art') else None), art_colours=S.get('art_colours', {}), art_bands=S.get('art_bands', {}), art_details=S.get('art_details', []), top=S.get('top', []),
        parts=S['parts'], roles=roles, height_m=round(F.z(lm['crown']['row']), 4),
        datum=dict(side_col=dcol, why=dwhy, side_sign=side_sign, front_mid=mid),
        landmarks={k: v for k, v in lm.items() if not k.startswith('_')}, face_profile={k: v for k, v in face.items() if k not in ('rows', 'front_px')},
        stations=dict(trunk=st_trunk, head=st_head, face=st_face, leg_R=st_legs['R'], leg_L=st_legs['L'],
                      shoe_R=st_shoes['R'], shoe_L=st_shoes['L'], arm_R=st_arms['R'], arm_L=st_arms['L']),
        islands=islands, garments=garments, skull=skull, hair=hair, face=face_vec,
    )
    np.save(os.path.join(out, 'labels.npy'), L.lab.astype(np.int16))
    np.save(os.path.join(out, 'labels_ink.npy'), L.ink.astype(np.uint8))
    # shaved temple: dark pixels in the hair over the head rows, painted
    try:
        hair_m = L.mask(roles['hair']) if roles.get('hair') else np.zeros_like(L.ink)
        lum = img.mean(2)
        buzz = hair_m & L.ink & (lum > 22) & (lum < 170)
        r_top, r_bot = lm['crown']['row'], lm['chin']['row'] + 20
        buzz[:r_top] = False; buzz[r_bot:] = False
        buzz = ndi.binary_opening(buzz, structure=np.ones((3, 3), bool), iterations=2)
        n_buzz = int(buzz.sum())
        if n_buzz >= 150:
            rgb = [int(v) for v in np.median(img[buzz], axis=0)]
            char['buzz'] = dict(px=n_buzz, rgb=rgb, rule='hair-labelled unmatched dark pixels over the head, opened 2 px: skin paint')
            np.save(os.path.join(out, 'labels_buzz.npy'), buzz.astype(np.uint8))
            log('[read] buzz (shaved) regions: %d px, colour %s -> skin paint' % (n_buzz, rgb))
    except Exception as e_:
        log('[read] buzz: skipped (%s)' % e_)
    json.dump(char, open(os.path.join(out, 'char.json'), 'w'), indent=1)
    json.dump(dict(ids=L.ids, panels={k: list(v) for k, v in L.panels.items()}), open(os.path.join(out, 'labels.json'), 'w'))
    base = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    ov = Image.new('RGBA', base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for k, v in lm.items():
        if k.startswith('_'):
            continue
        r = v['row']
        col = (0, 150, 0, 160) if v.get('source') != 'CONV' else (200, 120, 0, 160)
        dr.line([(c0f, r), (c1f, r)], fill=col, width=2)
        dr.text((c0f + 4, r - 12), k, fill=(0, 90, 0, 255))
    for sd, a in arms.items():
        P = np.array(a['axis_px'])
        dr.line([(c, r) for r, c in P], fill=(220, 0, 0, 200), width=3)
        for k, v in a['landmarks'].items():
            dr.ellipse([v['col'] - 7, v['row'] - 7, v['col'] + 7, v['row'] + 7], outline=(0, 0, 255, 255), width=2)
    for name, st in (('trunk', st_trunk), ('leg_R', st_legs['R']), ('leg_L', st_legs['L']), ('head', st_head), ('shoe_R', st_shoes['R']), ('shoe_L', st_shoes['L'])):
        src = {'trunk': front_trunk, 'leg_R': {r: (float(legs['R'][r][0]), float(legs['R'][r][1])) for r in legs['R']},
               'leg_L': {r: (float(legs['L'][r][0]), float(legs['L'][r][1])) for r in legs['L']}, 'head': fr_head,
               'shoe_R': {r: (float(legs['R'][r][0]), float(legs['R'][r][1])) for r in legs['R']},
               'shoe_L': {r: (float(legs['L'][r][0]), float(legs['L'][r][1])) for r in legs['L']}}[name]
        for s_ in st:
            r = s_['row']
            if r in src:
                a, b = src[r]
                dr.line([(a, r), (b, r)], fill=(0, 0, 255, 220), width=2)
    for r, (fc, bc, srcs) in side_trunk.items():
        if r % 4 == 0:
            dr.point((fc, r), fill=(255, 0, 255, 255) if srcs == 'measured' else (255, 160, 0, 255))
            dr.point((bc, r), fill=(255, 0, 255, 255))
    for r, (fc, bc, srcs) in side_leg.items():
        if r % 4 == 0:
            dr.point((fc, r), fill=(255, 0, 255, 255) if srcs == 'measured' else (255, 160, 0, 255))
            dr.point((bc, r), fill=(255, 0, 255, 255))
    for s_ in st_head + st_trunk + st_legs['R']:
        r = s_['row']
        if r in side_trunk or r in side_leg or r in side_head:
            fc, bc, _ = (side_trunk.get(r) or side_leg.get(r) or side_head.get(r))
            dr.line([(fc, r), (bc, r)], fill=(0, 0, 255, 200), width=2)
    for k in ('brow', 'nose_tip', 'lip'):
        if k in face:
            v = face[k]
            x = v['x_px'] if side_sign > 0 else -v['x_px']
            dr.ellipse([x - 6, v['row'] - 6, x + 6, v['row'] + 6], outline=(255, 0, 0, 255), width=2)
            dr.text((x + 8, v['row'] - 6), k, fill=(150, 0, 0, 255))
    dr.line([(dcol, 0), (dcol, base.size[1])], fill=(0, 160, 160, 200), width=2)
    for f in face_vec['features']:
        pts = [(mid + u * cal['ppm'], cal['floor_row'] - z * cal['ppm']) for u, z in f['poly']]
        if len(pts) >= 3:
            dr.polygon(pts, outline=(255, 0, 0, 255))
    base = Image.alpha_composite(base.convert('RGBA'), ov).convert('RGB')
    base.save(os.path.join(out, 'reading.png'))
    log('[read] wrote %s/char.json and reading.png in %.1fs' % (out, time.time() - t0))
    logf.close()
    return char


if __name__ == '__main__':
    main(sys.argv[1:])
