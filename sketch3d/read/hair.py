"""Read the drawn hair: a cap outline per row and the spurs (spikes, locks, tails).

Front u = x (+X is the character's left), side u = y (+Y is back), z up from the floor, metres.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import measure
from . import geometry as G


def poly_of(mask, to_u, F, tol_px=3.0):
    cs = measure.find_contours(np.pad(mask.astype(float), 1), 0.5)
    if not cs:
        return []
    c = max(cs, key=len) - 1.0
    c = measure.approximate_polygon(c, tol_px)
    return [[round(float(to_u(q[1])), 4), round(float(F.z(q[0])), 4)] for q in c]
try:
    import cv2
except ImportError:
    cv2 = None


def contour_spikes(mask, min_depth_px, min_len_px):
    """Return spikes as convex arcs between deep contour notches: [(base_a, base_b, tip, depth_px)]."""
    if cv2 is None:
        return []
    m8 = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    cnt = max(cs, key=cv2.contourArea)
    if len(cnt) < 20:
        return []
    hull = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return []
    if defects is None:
        return []
    notches = sorted([(int(d[0][2]), float(d[0][3]) / 256.0) for d in defects if d[0][3] / 256.0 >= min_depth_px], key=lambda q: q[0])
    if len(notches) < 2:
        return []
    pts = cnt[:, 0, :]
    n = len(pts)
    out = []
    idxs = [q[0] for q in notches]
    for a, b in zip(idxs, idxs[1:] + [idxs[0] + n]):
        arc = [pts[i % n] for i in range(a, b + 1)]
        if len(arc) < 4:
            continue
        pa, pb = pts[a % n].astype(float), pts[b % n].astype(float)
        chord = pb - pa
        L = np.linalg.norm(chord)
        if L < 1e-6:
            continue
        nrm = np.array([-chord[1], chord[0]]) / L
        dists = [float((np.array(q, float) - pa).dot(nrm)) for q in arc]
        j = int(np.argmax(np.abs(dists)))
        depth = abs(dists[j])
        if depth < min_len_px:
            continue
        tip = arc[j]
        arc_rc = [(float(q[1]), float(q[0])) for q in arc[::max(1, len(arc) // 24)]]
        out.append(((float(pa[1]), float(pa[0])), (float(pb[1]), float(pb[0])), (float(tip[1]), float(tip[0])), depth, arc_rc))
    return out


def split_branches(pm, touch, min_len_px):
    """Split a piece by its skeleton's leaf branches into [(pixels, base, tip)]."""
    sk = skeletonize(pm)
    if not sk.any():
        rr, cc = np.nonzero(pm)
        tr, tc = np.nonzero(touch)
        base = (float(tr.mean()), float(tc.mean()))
        d2 = (rr - base[0]) ** 2 + (cc - base[1]) ** 2
        j = int(np.argmax(d2))
        return [(pm, base, (float(rr[j]), float(cc[j])))]
    paths, deg = trace_paths(sk)
    near_base = ndi.binary_dilation(touch, iterations=6)
    tips = []
    for path in paths:
        ends = [path[0], path[-1]]
        for e in ends:
            if deg.get(e, 0) == 1 and not near_base[e]:
                br = path if path[-1] == e else path[::-1]
                br = br[::-1]
                L = sum(np.hypot(br[i][0] - br[i + 1][0], br[i][1] - br[i + 1][1]) for i in range(len(br) - 1))
                tips.append((e, br, L))
    # a leaf under 35 % of the main reach is a decoration, not a spur
    pts_sk = set(zip(*np.nonzero(sk)))
    tr, tc = np.nonzero(touch)
    b0 = (float(tr.mean()), float(tc.mean()))
    start = min(pts_sk, key=lambda q: (q[0] - b0[0]) ** 2 + (q[1] - b0[1]) ** 2)
    dist = {start: 0.0}
    frontier = [start]
    while frontier:
        nxt = []
        for q in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    w = (q[0] + dr, q[1] + dc)
                    if (dr or dc) and w in pts_sk and w not in dist:
                        dist[w] = dist[q] + (1.4142 if dr and dc else 1.0)
                        nxt.append(w)
        frontier = nxt
    diameter = max(dist.values()) if dist else 0.0
    tips = [t for t in tips if t[2] >= max(min_len_px, 0.35 * diameter)]
    if len(tips) <= 1:
        rr, cc = np.nonzero(pm)
        tr, tc = np.nonzero(touch)
        base = (float(tr.mean()), float(tc.mean()))
        if tips:
            tip = (float(tips[0][0][0]), float(tips[0][0][1]))
        else:
            d2 = (rr - base[0]) ** 2 + (cc - base[1]) ** 2
            j = int(np.argmax(d2))
            tip = (float(rr[j]), float(cc[j]))
        return [(pm, base, tip)]
    lab_sk = np.zeros(pm.shape, np.int32)
    for i, (e, br, L) in enumerate(tips):
        for q in br:
            lab_sk[q] = i + 1
    if not (lab_sk > 0).any():
        return []
    _, inds = ndi.distance_transform_edt(lab_sk == 0, return_indices=True)
    owner = lab_sk[inds[0], inds[1]]
    out = []
    for i, (e, br, L) in enumerate(tips):
        pix = pm & (owner == i + 1)
        if not pix.any():
            continue
        t_ = pix & touch
        if t_.any():
            tr, tc = np.nonzero(t_)
            base = (float(tr.mean()), float(tc.mean()))
        else:
            base = (float(br[-1][0]), float(br[-1][1]))
        out.append((pix, base, (float(e[0]), float(e[1]))))
    return out


def trace_paths(sk):
    pts = set(zip(*np.nonzero(sk)))

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in pts]
    deg = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    if not nodes and pts:
        nodes = {next(iter(pts))}
    visited = set()
    paths = []
    for start in nodes:
        for nb in nbrs(start):
            e = (start, nb) if start < nb else (nb, start)
            if e in visited:
                continue
            path = [start, nb]
            visited.add(e)
            prev, cur = start, nb
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                e = (prev, cur) if prev < cur else (cur, prev)
                if e in visited:
                    break
                visited.add(e)
                path.append(cur)
            paths.append(path)
    return paths, deg


CAP_MARGIN = 0.035  # metres, hair farther off the skull is a spur


def head_region(panel, shape, lm, skull, cal, pan, dcol, side_sign, L, roles):
    """Return the skull ellipse in `panel` grown by CAP_MARGIN, as a bool mask."""
    if skull is None or not skull.get('w_mm'):
        return None
    ppm = cal['ppm']
    r_top, r_bot = lm['crown']['row'], lm['chin']['row']
    # the dome above the hairline is 0.55 x the skull width
    r_hair = lm['hairline']['row'] if 'hairline' in lm else r_top
    r_skull_top = min(r_top, r_hair - 0.55 * skull['w_mm'] / 1000.0 * ppm)
    rc = 0.5 * (r_skull_top + r_bot)
    b = 0.5 * (r_bot - r_skull_top) + CAP_MARGIN * ppm
    if panel in ('front', 'back'):
        cc = float(pan[panel]['midline'])
        a = 0.5 * skull['w_mm'] / 1000.0 * ppm + CAP_MARGIN * ppm
    else:
        # the side skull is at least 1.05 x its width deep, the hair hides its back
        skin = L.mask(roles['skin'], 'side')
        c0, c1 = L.panels['side']
        fronts = []
        for r in range(int(r_hair), int(r_bot)):
            cs = np.nonzero(skin[r, c0:c1])[0]
            if len(cs):
                us = (c0 + cs - dcol) / ppm * side_sign
                fronts.append(float(us.min()))
        d = max(skull.get('d_mm') or 0.0, skull['w_mm'] * 1.05)
        if fronts:
            front_u = float(np.median(sorted(fronts)[:max(1, len(fronts) // 4)]))
            cc = dcol + (front_u + 0.5 * d / 1000.0) * ppm * side_sign
        else:
            cc = float(dcol)
        a = 0.5 * d / 1000.0 * ppm + CAP_MARGIN * ppm
    rr, cc_ = np.ogrid[:shape[0], :shape[1]]
    ell = ((cc_ - cc) / a) ** 2 + ((rr - rc) / b) ** 2 <= 1.0
    return ell, int(round(rc))


def hair_read(L, S, lm, roles, dcol, side_sign, log=print, skull=None):
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    Hpx = cal['floor_row'] - lm['crown']['row']
    cap_r = max(3, int(0.03 * Hpx))
    out = dict(cap={}, spurs={}, mask_rows={}, plates=[])
    # plates (mohawk, crest, fin) come from the side silhouette, other hair from cap and spurs
    cap_classes = list(roles['hair'])
    if skull and skull.get('w_mm') and 'side' in L.panels and 'front' in L.panels:
        w_sk = skull['w_mm'] / 1000.0
        z_sk_top = lm['chin']['z'] + 0.9 * w_sk
        z_mid_head = lm['chin']['z'] + 0.45 * w_sk
        for cname in list(roles['hair']):
            mf = L.mask([cname], 'front'); ms = L.mask([cname], 'side')
            cf0, cf1 = L.panels['front']; cs0, cs1 = L.panels['side']
            Tf = G.runs_table(mf, cf0, cf1, min_w=3); Ts = G.runs_table(ms, cs0, cs1, min_w=3)
            if not Tf or not Ts:
                continue
            z_top = F.z(min(Ts))
            ws = [max(q[1] for q in Tf[r]) - min(q[0] for q in Tf[r]) for r in Tf if F.z(r) > z_mid_head]
            w_front = (np.percentile(ws, 90) / cal['ppm']) if ws else 0.0
            n_side = sum(1 for r in Ts if F.z(r) > z_sk_top)
            if z_top - z_sk_top >= 0.03 and w_front <= 0.45 * w_sk and n_side >= 0.02 * Hpx:
                to_us = lambda c: (c - dcol) / cal['ppm'] * side_sign
                ms_p = ms.copy(); ms_p[:, :cs0] = False; ms_p[:, cs1:] = False
                poly = poly_of(ms_p, to_us, F, tol_px=max(2.0, 0.004 * cal['ppm']))
                thick = []
                rows_f = sorted(Tf)
                for r in rows_f[::max(1, len(rows_f) // 24)]:
                    w_ = (max(q[1] for q in Tf[r]) - min(q[0] for q in Tf[r])) / cal['ppm']
                    xc_ = (0.5 * (max(q[1] for q in Tf[r]) + min(q[0] for q in Tf[r])) - float(pan['front']['midline'])) / cal['ppm']
                    thick.append(dict(z=round(F.z(r), 4), w=round(w_, 4), x=round(xc_, 4)))
                out['plates'].append(dict(cls=cname, poly=poly, thick=thick, z_top=round(z_top, 4), w_front=round(w_front, 4)))
                cap_classes.remove(cname)
                log('  hair %s: a PLATE (top %.0f mm above the skull, %.0f mm wide in front) - its profile silhouette extruded' % (cname, 1000 * (z_top - z_sk_top), 1000 * w_front))
    # hair under 14 mm off the skin (buzz cut, stubble) is painted
    out['painted'] = []
    head_side = head_region('side', L.lab.shape, lm, skull, cal, pan, dcol, side_sign, L, roles) if skull else None
    if 'side' in L.panels and head_side is not None:
        cs0, cs1 = L.panels['side']
        ell = ndi.binary_erosion(head_side[0], iterations=max(1, int(round(CAP_MARGIN * cal['ppm']))))
        Tk = G.runs_table(ell, cs0, cs1, min_w=3)
        r_skull_top = min(Tk) if Tk else None
        for cname in list(cap_classes):
            ms = L.mask([cname], 'side')
            Th = G.runs_table(ms, cs0, cs1, min_w=2)
            if not Th or r_skull_top is None:
                continue
            beyond = []
            for r, runs in Th.items():
                if r not in Tk:
                    continue
                a = min(q[0] for q in runs); b = max(q[1] for q in runs)
                sa = min(q[0] for q in Tk[r]); sb = max(q[1] for q in Tk[r])
                beyond.append(max(0, sa - a) + max(0, b - sb))
            above = max(0, r_skull_top - min(Th))
            thick_mm = 1000.0 * max(float(np.percentile(beyond, 75)) if beyond else 0.0, above) / cal['ppm']
            log('  hair %s: %.0f mm beyond the skull in profile' % (cname, thick_mm))
            if thick_mm < 14.0:
                cap_classes.remove(cname)
                out['painted'].append(cname)
                log('  hair %s: %.0f mm thick in profile - a buzz, painted on the skull (no cap)' % (cname, thick_mm))
    if not cap_classes:
        return out
    for panel in ('front', 'side', 'back'):
        if panel not in L.panels:
            continue
        c0, c1 = L.panels[panel]
        m = L.mask(cap_classes, panel)
        if not m[:, c0:c1].any():
            continue
        acc_cls = list(roles.get('accessory', []))
        if acc_cls:
            acc = L.mask(acc_cls, panel)
            acc[:, :c0] = False; acc[:, c1:] = False
            lab_h, n_h = ndi.label(m, structure=np.ones((3, 3), int))
            lab_a, n_a = ndi.label(acc, structure=np.ones((3, 3), int))
            n_bridge = 0
            for k in range(1, n_a + 1):
                am = lab_a == k
                ring = ndi.binary_dilation(am, iterations=3) & ~am
                touched = set(np.unique(lab_h[ring]).tolist()) - {0}
                if len(touched) >= 2:
                    m = m | am
                    n_bridge += 1
            if n_bridge:
                log('  hair %s: %d accessory blob(s) tied around the hair read as hair (the hair continues under them)' % (panel, n_bridge))
        if panel == 'front':
            mid = float(pan['front']['midline'])
            to_u = lambda c: (c - mid) / cal['ppm']
        elif panel == 'back':
            mid = float(pan['back']['midline'])
            to_u = lambda c: (mid - c) / cal['ppm']
        else:
            to_u = lambda c: (c - dcol) / cal['ppm'] * side_sign
        head_reg = head_region(panel, m.shape, lm, skull, cal, pan, dcol, side_sign, L, roles)
        # the cap: the thick mass above the skull centre plus thin sheets on the skull below it
        cap_big = ndi.binary_opening(m, structure=np.ones((3, 3), bool), iterations=cap_r)
        if head_reg is not None:
            ell, rc = head_reg
            cap_small = ndi.binary_opening(m & ell, structure=np.ones((3, 3), bool), iterations=max(2, cap_r // 4))
            rows = np.arange(m.shape[0])[:, None]
            cap = (cap_big & (rows < rc)) | cap_small
        else:
            cap = cap_big
        T = G.runs_table(cap, c0, c1, min_w=3)
        rows = sorted(T)
        cap_rows = []
        for r in rows[::max(1, len(rows) // 40)]:
            a = min(q[0] for q in T[r]); b = max(q[1] for q in T[r])
            cap_rows.append(dict(row=int(r), z=round(F.z(r), 4), u0=round(to_u(a), 4), u1=round(to_u(b), 4)))
        out['cap'][panel] = cap_rows
        Tm = G.runs_table(m, c0, c1, min_w=3)
        rows_m = sorted(Tm)
        out['mask_rows'][panel] = [dict(row=int(r), z=round(F.z(r), 4), runs=[[round(to_u(a), 4), round(to_u(b), 4)] for a, b in Tm[r]])
                                   for r in rows_m[::max(1, len(rows_m) // 60)]]
        dist = ndi.distance_transform_edt(m)
        capd = ndi.binary_dilation(cap, iterations=3)
        prot = m & ~capd
        lab_p, n_p = ndi.label(prot, structure=np.ones((3, 3), int))
        spurs = []
        min_px = max(40, 0.0004 * Hpx * Hpx * 0.05)
        for k in range(1, n_p + 1):
            pm = lab_p == k
            if int(pm.sum()) < min_px:
                continue
            touch = pm & ndi.binary_dilation(capd, iterations=1)
            if not touch.any():
                continue
            for pix, base, tip in split_branches(pm, touch, min_len_px=0.02 * cal['ppm']):
                px = int(pix.sum())
                if px < min_px:
                    continue
                length = float(np.hypot(tip[0] - base[0], tip[1] - base[1])) / cal['ppm']
                if length < 0.02:
                    continue
                n_ = 5
                samples = []
                for t in np.linspace(0.0, 1.0, n_):
                    r_ = int(round(base[0] + (tip[0] - base[0]) * t)); c_ = int(round(base[1] + (tip[1] - base[1]) * t))
                    r_ = min(max(r_, 0), m.shape[0] - 1); c_ = min(max(c_, 0), m.shape[1] - 1)
                    samples.append(dict(u=round(to_u(c_), 4), z=round(F.z(r_), 4), r=round(float(dist[r_, c_] / cal['ppm']), 4)))
                d = np.array([tip[1] - base[1], -(tip[0] - base[0])], float)
                ang = float(np.degrees(np.arctan2(d[0] * (1 if panel != 'back' else -1), d[1])))  # 0 = up, +90 = toward +u
                spurs.append(dict(base=dict(u=round(to_u(base[1]), 4), z=round(F.z(base[0]), 4)),
                                  tip=dict(u=round(to_u(tip[1]), 4), z=round(F.z(tip[0]), 4)),
                                  length_m=round(length, 4), angle_deg=round(ang, 1), px=px, samples=samples,
                                  poly=poly_of(pix | (capd & ndi.binary_dilation(pix, iterations=6)), to_u, F)))
        # locks over the face (a bang, a sideburn) below the hairline
        r_hair = int(lm['hairline']['row']) if 'hairline' in lm else int(lm['crown']['row'])
        r_chin = int(lm['chin']['row'])
        skin = L.mask(roles['skin'], panel)
        lockm = np.zeros_like(m)
        if panel in ('front', 'back'):
            for r in range(max(0, r_hair - 8), min(m.shape[0], r_chin + 1)):
                cs = np.nonzero(skin[r, c0:c1])[0]
                if len(cs) == 0:
                    continue
                a0, a1 = c0 + int(cs.min()) - 12, c0 + int(cs.max()) + 12
                lockm[r, max(c0, a0):min(c1, a1 + 1)] = m[r, max(c0, a0):min(c1, a1 + 1)]
        else:
            rows_t = [r for r in range(r_hair, r_chin) if skin[r, c0:c1].any()][:20]
            if rows_t:
                cen = int(np.mean([0.5 * (np.nonzero(skin[r, c0:c1])[0].min() + np.nonzero(skin[r, c0:c1])[0].max()) for r in rows_t])) + c0
                face_side = -side_sign
                for r in range(max(0, r_hair - 8), min(m.shape[0], r_chin + 1)):
                    if face_side < 0:
                        lockm[r, c0:cen] = m[r, c0:cen]
                    else:
                        lockm[r, cen:c1] = m[r, cen:c1]
        lockm &= ~ndi.binary_dilation(prot, iterations=2)
        lab_l, n_l = ndi.label(lockm, structure=np.ones((3, 3), int))
        for k in range(1, n_l + 1):
            lm_ = lab_l == k
            px = int(lm_.sum())
            if px < max(60, 0.0004 * Hpx * Hpx * 0.05):
                continue
            rr, cc = np.nonzero(lm_)
            r_top = int(rr.min())
            top = rr <= r_top + 3
            base = (float(rr[top].mean()), float(cc[top].mean()))
            d2 = (rr - base[0]) ** 2 + (cc - base[1]) ** 2
            j = int(np.argmax(d2))
            tip = (float(rr[j]), float(cc[j]))
            length = float(np.sqrt(d2[j])) / cal['ppm']
            if length < 0.015:
                continue
            n_ = 5
            samples = []
            for t in np.linspace(0.0, 1.0, n_):
                r_ = int(round(base[0] + (tip[0] - base[0]) * t)); c_ = int(round(base[1] + (tip[1] - base[1]) * t))
                r_ = min(max(r_, 0), m.shape[0] - 1); c_ = min(max(c_, 0), m.shape[1] - 1)
                samples.append(dict(u=round(to_u(c_), 4), z=round(F.z(r_), 4), r=round(float(dist[r_, c_] / cal['ppm']), 4)))
            d = np.array([tip[1] - base[1], -(tip[0] - base[0])], float)
            ang = float(np.degrees(np.arctan2(d[0] * (1 if panel != 'back' else -1), d[1])))
            spurs.append(dict(base=dict(u=round(to_u(base[1]), 4), z=round(F.z(base[0]), 4)),
                              tip=dict(u=round(to_u(tip[1]), 4), z=round(F.z(tip[0]), 4)),
                              length_m=round(length, 4), angle_deg=round(ang, 1), px=px, samples=samples, via='lock',
                              poly=poly_of(lm_, to_u, F)))
        spurs.sort(key=lambda s: s['base']['u'])
        out['spurs'][panel] = spurs
        log('  hair %-5s: cap rows %d..%d, %d spurs (lengths %s)' % (panel, rows[0] if rows else -1, rows[-1] if rows else -1, len(spurs),
                                                                   ' '.join('%.0f' % (s['length_m'] * 1000) for s in spurs)))
    return out
