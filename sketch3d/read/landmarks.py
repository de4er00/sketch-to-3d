"""Find landmark rows (neck, shoulder, armpit, waist, crotch, knee, ankle) on the front panel."""
import numpy as np
from . import geometry as G

CHIN_K = 1.15


def _width_profile(T, mid):
    W, X0, X1, N = {}, {}, {}, {}
    for r, runs in T.items():
        c = G.central(runs, mid)
        if c is None:
            continue
        W[r] = float(c[1] - c[0])
        X0[r], X1[r] = c
        N[r] = len(runs)
    return W, X0, X1, N


def _first(rows, pred):
    for r in rows:
        if pred(r):
            return r
    return None


def lies_between(body, over):
    """Pixels of `over` with `body` on both sides in their row."""
    out = np.zeros_like(over)
    for r in np.nonzero(over.any(1))[0]:
        row = over[r]
        cols = np.nonzero(row)[0]
        starts = cols[np.r_[True, np.diff(cols) > 1]]
        ends = cols[np.r_[np.diff(cols) > 1, True]]
        for a, b in zip(starts, ends):
            if a > 0 and b + 1 < row.shape[0] and body[r, a - 1] and body[r, b + 1]:
                out[r, a:b + 1] = True
    return out


def front_landmarks(L, S, roles, log=print):
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    c0, c1 = L.panels['front']
    mid = float(pan['front']['midline'])
    fig = L.figure('front')
    T = G.runs_table(fig, c0, c1, min_w=3)
    rows = sorted(T)
    crown = rows[0]
    floor = int(cal['floor_row'])
    Hpx = floor - crown
    W, X0, X1, N = _width_profile(T, mid)
    Ws = G.smooth1(W, 3)
    skin = L.mask(roles.get('skin', []), 'front')
    lm = {}

    def put(name, row, rule, source='measured'):
        lm[name] = dict(row=int(row), z=round(F.z(row), 5), rule=rule, source=source)
        log('  %-10s row %4d  z %.4f  %s' % (name, row, F.z(row), rule))

    put('crown', crown, 'first figure row')
    # read the head and neck without hair and accessories
    hair_acc = L.mask(list(roles.get('hair', [])) + list(roles.get('accessory', [])), 'front')
    fig_body = fig & ~hair_acc
    fig_body |= fig & hair_acc & ~L.mask(list(roles.get('hair', [])), 'front') & lies_between(fig_body, fig & hair_acc)
    Tb = G.runs_table(fig_body, c0, c1, min_w=3)
    Wb_ = {}
    for r, runs in Tb.items():
        cen = [ab for ab in runs if ab[0] <= mid <= ab[1]]
        if not cen:
            cen = [min(runs, key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - mid))]
            if abs(0.5 * (cen[0][0] + cen[0][1]) - mid) > 0.08 * Hpx:
                continue
        Wb_[r] = float(cen[0][1] - cen[0][0])
    Wb = G.smooth1(Wb_, 3)
    body_rows = set(Wb_)
    Ws_fig = Ws
    Wb_only = {r: Wb[r] for r in rows if r in body_rows and r in Wb}
    if len(Wb_only) < 0.05 * Hpx:
        Wb_only = dict(Ws)
    Ws = Wb_only
    Tsk0 = G.runs_table(skin, c0, c1, r0=crown, r1=crown + int(0.30 * Hpx), min_w=3)
    Wsk0 = {}
    for r, runs in Tsk0.items():
        cen = [ab for ab in runs if ab[0] <= mid <= ab[1]] or [min(runs, key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - mid))]
        if abs(0.5 * (cen[0][0] + cen[0][1]) - mid) <= 0.08 * Hpx:
            Wsk0[r] = float(cen[0][1] - cen[0][0])
    Wsk0 = G.smooth1(Wsk0, 3) if len(Wsk0) >= 6 else Wsk0
    r_hmax = G.argmax_in(Wsk0, crown, crown + int(0.30 * Hpx)) if Wsk0 else None
    if r_hmax is None:
        r_hmax = G.argmax_in(Ws, crown, crown + int(0.30 * Hpx))
    put('head_widest', r_hmax, 'widest central skin run (the face) in the top 30 %')
    r_neck = G.argmin_in(Ws, r_hmax + 1, r_hmax + int(0.25 * Hpx))
    if r_neck is None:
        raise RuntimeError('no neck: the figure never narrows below its head')
    wneck0 = Ws[r_neck]
    r_chin = None
    for r in [q for q in rows if r_hmax <= q <= r_neck and q in Ws]:
        if Ws[r] <= 1.25 * wneck0:
            r_chin = r
            break
    if r_chin is None:
        r_chin = r_neck
    plat = [q for q in rows if r_chin <= q <= r_neck + int(0.08 * Hpx) and q in Ws and Ws[q] <= wneck0 * 1.06]
    if len(plat) >= 6:
        r_neck = int(round(np.mean(plat)))
        put('chin', r_chin, 'first row below the head whose width is within 1.25 x the neck (the jaw corner)')
        put('neck', r_neck, 'centre of the neck plateau (rows %d..%d)' % (plat[0], plat[-1]))
    else:
        put('chin', r_chin, 'first row below the head whose width is within 1.25 x the neck (the jaw corner)')
        put('neck', r_neck, 'narrowest central run below the head')
    Ws = Ws_fig
    Tsk = G.runs_table(skin, c0, c1, r0=r_chin, r1=r_neck + 4, min_w=3)
    wsk = [max(ab[1] for ab in runs) - min(ab[0] for ab in runs) for r, runs in Tsk.items() if any(ab[0] <= mid <= ab[1] for ab in runs)]
    if wsk:
        lm['neck']['skin_width_px'] = float(np.median(wsk))
    # armpit: where an arm merged with the torso splits off (a tail never merges)
    fig_nohair = fig & ~L.mask(list(roles.get('hair', [])), 'front')
    Ta = G.runs_table(fig_nohair, c0, c1, min_w=3)
    _Wa, Xa0, Xa1, _Na = _width_profile(Ta, mid)

    def flank(r, side):
        if r not in Ta or r not in Xa0:
            return None
        runs = Ta[r]
        c = (Xa0[r], Xa1[r])
        if side < 0:
            cand = [ab for ab in runs if ab[1] <= c[0] and ab[1] - ab[0] >= 0.03 * Hpx and c[0] - ab[1] < 0.2 * Hpx]
        else:
            cand = [ab for ab in runs if ab[0] >= c[1] and ab[1] - ab[0] >= 0.03 * Hpx and ab[0] - c[1] < 0.2 * Hpx]
        return cand[0] if cand else None
    r_armpit_side = {}
    for side in (-1, 1):
        prev_merged = True
        for r in [q for q in rows if q > r_neck and q in Ta]:
            f = flank(r, side)
            if f is not None and prev_merged and all(flank(q, side) is not None for q in range(r, min(r + 8, rows[-1])) if q in Ta):
                r_armpit_side[side] = r
                break
            prev_merged = f is None
    if not r_armpit_side:
        raise RuntimeError('no armpit: the figure never splits into arm | torso | arm (A-pose required)')
    r_armpit = max(r_armpit_side.values())
    put('armpit', r_armpit, 'first row split into arm | torso | arm')
    # shoulder: the outline turns outward more than 35 deg from vertical for 4 % of the height
    body = fig & ~L.mask(roles.get('accessory', []) + roles.get('hair', []), 'front')
    Tb = G.runs_table(body, c0, c1, r0=r_neck, r1=r_armpit + 1, min_w=3)
    Wb, Xb0, Xb1, _ = _width_profile(Tb, mid)
    seg = [r for r in rows if r_neck <= r < r_armpit and r in Xb0]
    xl = G.smooth1({r: float(Xb0[r]) for r in seg}, 3)
    xr = G.smooth1({r: float(Xb1[r]) for r in seg}, 3)
    corners = G.dp_rows(seg, [xl, xr], tol_px=1.5, min_gap=4, corner_deg=8.0)
    cand = []
    for j in range(1, len(corners) - 1):
        a_, b_, c_ = corners[j - 1], corners[j], corners[j + 1]
        angs = []
        for cv, sgn in ((xl, -1.0), (xr, 1.0)):
            v1 = np.array([sgn * (cv[b_] - cv[a_]), b_ - a_]); v2 = np.array([sgn * (cv[c_] - cv[b_]), c_ - b_])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            angs.append(np.degrees(np.arccos(np.clip(v1.dot(v2) / (n1 * n2), -1, 1))))
        if angs:
            cand.append((b_, max(angs)))
    log('    shoulder corners: ' + ', '.join('%d:%.0f' % c for c in cand))
    good = [c for c in cand if c[1] >= 25.0 and c[0] >= r_neck + 0.01 * Hpx]
    if good:
        r_sh = good[0][0]
        put('shoulder', r_sh, 'highest body-contour corner >= 25 deg between neck and armpit (%.0f deg)' % good[0][1])
    else:
        put('shoulder', r_neck + int(0.4 * (r_armpit - r_neck)), 'CONV: 40 % of the way from neck to armpit', 'CONV')
    def split(r):
        runs = T[r]
        if any(a <= mid < b for a, b in runs):
            return False
        lf = [ab for ab in runs if ab[1] <= mid and mid - ab[1] < 0.15 * Hpx and ab[1] - ab[0] > 0.04 * Hpx]
        rt = [ab for ab in runs if ab[0] > mid and ab[0] - mid < 0.15 * Hpx and ab[1] - ab[0] > 0.04 * Hpx]
        return bool(lf) and bool(rt)
    r_crotch = _first([r for r in rows if r > r_armpit], split)
    if r_crotch is None:
        raise RuntimeError('no crotch/legs split found below the armpit')
    put('crotch', r_crotch, 'first row where the central run splits into two legs')
    r_waist = G.argmin_in(Ws, r_armpit + int(0.04 * Hpx), r_crotch - int(0.03 * Hpx))
    put('waist', r_waist, 'narrowest central run between armpit and crotch')
    r_hip = G.argmax_in(Ws, r_waist, r_crotch - 1)
    put('hip', r_hip, 'widest central run between waist and crotch')
    legs = {'R': {}, 'L': {}}  # image-left = character's right on the front panel
    for r in [r for r in rows if r >= r_crotch]:
        runs = T[r]
        lf = [ab for ab in runs if ab[1] <= mid and mid - ab[1] < 0.15 * Hpx]
        rt = [ab for ab in runs if ab[0] >= mid and ab[0] - mid < 0.15 * Hpx]
        if lf:
            legs['R'][r] = max(lf, key=lambda ab: ab[1])
        if rt:
            legs['L'][r] = min(rt, key=lambda ab: ab[0])
    shoe = L.mask(roles.get('shoe', []), 'front')
    sole = L.mask(roles.get('sole', []), 'front')
    for side in ('R', 'L'):
        LR = legs[side]
        lrows = sorted(LR)
        wl = {r: float(LR[r][1] - LR[r][0]) for r in lrows}
        r_ank = None
        for r in lrows:
            a, b = LR[r]
            if shoe[r, a:b].mean() > 0.5:
                r_ank = r
                break
        if r_ank is None:
            r_ank = G.argmin_in(G.smooth1(wl, 3), lrows[0] + int(0.6 * (lrows[-1] - lrows[0])), lrows[-1] - 5)
            put('ankle_' + side, r_ank, 'narrowest leg row in its lower third (no shoe class)')
        else:
            put('ankle_' + side, r_ank, 'top row of the shoe class on this leg')
        bare = [r for r in lrows if r_crotch + 0.15 * (r_ank - r_crotch) < r < r_ank - 0.1 * (r_ank - r_crotch)
                and skin[r, LR[r][0]:LR[r][1]].mean() > 0.6]
        if len(bare) > 0.15 * (r_ank - r_crotch):
            wb = G.smooth1({r: wl[r] for r in bare}, 3)
            wmin_ = min(wb.values())
            plateau = [r for r in bare if wb[r] <= wmin_ + 1.0]
            r_knee = int(round(np.mean(plateau)))
            put('knee_' + side, r_knee, 'narrowest BARE leg row between crotch and ankle')
        else:
            r_knee = int(round(r_crotch + 0.42 * (r_ank - r_crotch)))
            put('knee_' + side, r_knee, 'CONV: crotch + 0.42 x (ankle - crotch); leg covered', 'CONV')
        r_sole = None
        for r in reversed(lrows):
            a, b = LR[r]
            cov = sole[r, a:b].mean()
            if cov >= 0.7:
                r_sole = r
            elif r_sole is not None and r < r_sole - 3:
                break
        if r_sole is not None:
            put('sole_top_' + side, r_sole, 'top of the full-width sole slab (sole class >= 70 % of the leg run, scanned up from the floor)')
        else:
            put('sole_top_' + side, int(round(F.floor - 0.012 * F.ppm)), 'CONV: no sole class drawn; 12 mm slab from the floor', 'CONV')
    lm['_profile'] = dict(rows=rows, W=W, X0=X0, X1=X1, N=N, legs=legs, T=T)
    return lm
