"""Read the profile panel: face turning points, the fore-aft datum, depth per row.

y = 0 is the centre of a bare neck or shin section, +y is the character's back.
"""
import numpy as np
from . import geometry as G


def side_runs(L, S, classes, r0, r1):
    c0, c1 = L.panels['side']
    m = L.mask(classes, 'side')
    return G.runs_table(m, c0, c1, r0=r0, r1=r1, min_w=3)


def face_profile(L, S, lm, roles, log=print):
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    sp = pan['side']
    faces_left = sp.get('sign', 1) == 1  # faces image-left: the front is the low column
    c0, c1 = L.panels['side']
    skin = L.mask(roles.get('skin', []), 'side')
    T = G.runs_table(skin, c0, c1, r0=lm['crown']['row'], r1=lm['chin']['row'] + 40, min_w=3)
    front = {}
    for r, runs in T.items():
        ab = min(runs, key=lambda q: q[0]) if faces_left else max(runs, key=lambda q: q[1])
        front[r] = float(ab[0] if faces_left else -ab[1])
    if len(front) < 10:
        raise RuntimeError('side panel: no face skin profile found')
    rows = sorted(front)
    fs = G.smooth1(front, 2)
    corners = G.dp_rows(rows, [fs], tol_px=1.5, min_gap=3, corner_deg=10.0)
    pts = []
    for j in range(1, len(corners) - 1):
        a, b, c = corners[j - 1], corners[j], corners[j + 1]
        v1 = np.array([fs[b] - fs[a], b - a]); v2 = np.array([fs[c] - fs[b], c - b])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        ang = np.degrees(np.arccos(np.clip(v1.dot(v2) / (n1 * n2), -1, 1)))
        pts.append(dict(row=int(b), z=round(F.z(b), 5), x_px=float(fs[b]), angle=float(ang), convex=bool(cross < 0)))
    prof = dict(rows=rows, front_px={r: fs[r] for r in rows}, corners=pts)
    if pts:
        nose = min(pts, key=lambda p: p['x_px'])
        prof['nose_tip'] = nose
        above = [p for p in pts if p['row'] < nose['row'] - 5]
        below = [p for p in pts if p['row'] > nose['row'] + 5]
        if above:
            prof['brow'] = max(above, key=lambda p: p['angle'])
        if below:
            prof['lip'] = max(below, key=lambda p: p['angle'])
        prof['chin_side'] = dict(row=rows[-1], z=round(F.z(rows[-1]), 5))
    log('  face profile: %d corners; nose tip row %s; brow %s; lip %s' % (
        len(pts), prof.get('nose_tip', {}).get('row'), prof.get('brow', {}).get('row'), prof.get('lip', {}).get('row')))
    return prof


def datum(L, S, lm, roles, log=print):
    """Return (col, why): the mid of the bare neck run on the side, else of a bare shin."""
    c0, c1 = L.panels['side']
    skin = L.mask(roles.get('skin', []), 'side')
    hair = L.mask(roles.get('hair', []), 'side')
    r_chin, r_neck = lm['chin']['row'], lm['neck']['row']
    T = G.runs_table(skin, c0, c1, r0=r_chin, r1=r_neck + 30, min_w=6)
    cands = []
    for r, runs in T.items():
        if len(runs) == 1 and not hair[r, runs[0][0]:runs[0][1]].any():
            cands.append((r, 0.5 * (runs[0][0] + runs[0][1])))
    if len(cands) >= 8:
        col = float(np.median([c for r, c in cands]))
        return col, 'bare neck: median mid of %d single skin runs, rows %d..%d' % (len(cands), cands[0][0], cands[-1][0])
    kn = lm.get('knee_R', lm.get('knee_L'))['row']
    an = lm.get('ankle_R', lm.get('ankle_L'))['row']
    T = G.runs_table(skin, c0, c1, r0=kn, r1=an, min_w=6)
    cands = [(r, 0.5 * (runs[0][0] + runs[0][1])) for r, runs in T.items() if len(runs) == 1]
    if len(cands) >= 8:
        col = float(np.median([c for r, c in cands]))
        return col, 'bare shin: median mid of %d single skin runs, rows %d..%d' % (len(cands), cands[0][0], cands[-1][0])
    fig = L.figure('side')
    T = G.runs_table(fig, c0, c1, r0=lm['hip']['row'], r1=lm['hip']['row'] + 1, min_w=6)
    runs = T.get(lm['hip']['row'], [])
    if runs:
        a = min(q[0] for q in runs); b = max(q[1] for q in runs)
        return 0.5 * (a + b), 'CONV: figure mid at the hip row (no bare neck or shin on the side panel)'
    raise RuntimeError('side panel: no datum')


def depth_profile(L, S, classes, r0, r1, occluded_rows, log=print):
    """Per row (front_col, back_col) of `classes`, occluded rows interpolated."""
    c0, c1 = L.panels['side']
    T = side_runs(L, S, classes, r0, r1)
    clean = {}
    for r, runs in T.items():
        a = min(q[0] for q in runs); b = max(q[1] for q in runs)
        clean[r] = (float(a), float(b))
    out = {}
    rows = sorted(clean)
    if not rows:
        return out
    good = [r for r in rows if r not in occluded_rows]
    for r in rows:
        if r in occluded_rows and good:
            above = [q for q in good if q < r]
            below = [q for q in good if q > r]
            if above and below:
                qa, qb = above[-1], below[0]
                t = (r - qa) / float(qb - qa)
                fa, ba = clean[qa]; fb, bb = clean[qb]
                out[r] = (fa + (fb - fa) * t, ba + (bb - ba) * t, 'CONV')
            elif above:
                out[r] = (clean[above[-1]][0], clean[above[-1]][1], 'CONV')
            else:
                out[r] = (clean[below[0]][0], clean[below[0]][1], 'CONV')
        else:
            out[r] = (clean[r][0], clean[r][1], 'measured')
    return out
