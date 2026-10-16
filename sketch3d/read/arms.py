"""Trace the arm chains of an A-pose front panel: axis, widths, landmarks."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from . import geometry as G


def _trace(sk):
    pts = set(zip(*np.nonzero(sk)))

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in pts]
    deg = {p: len(nbrs(p)) for p in pts}
    return pts, nbrs, deg


def diameter_path(sk):
    import heapq
    pts, nbrs, deg = _trace(sk)
    if not pts:
        return []

    def far(src):
        dist = {src: 0}
        prev = {}
        pq = [(0, src)]
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist.get(x, 1e18):
                continue
            for y in nbrs(x):
                w = 1.0 if (y[0] == x[0] or y[1] == x[1]) else 1.4142
                if d + w < dist.get(y, 1e18):
                    dist[y] = d + w
                    prev[y] = x
                    heapq.heappush(pq, (d + w, y))
        end = max(dist, key=dist.get)
        return end, prev
    tips = [p for p in pts if deg[p] == 1] or list(pts)
    a, _ = far(tips[0])
    b, prev = far(a)
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def resample(path, step):
    P = np.array(path, float)
    if len(P) < 2:
        return P
    d = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(d)])
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1), t


def perp_width(region, p, tangent, max_r=400):
    """Return (width, centre offset) along the normal at p, in px."""
    n = np.array([-tangent[1], tangent[0]])
    H, W = region.shape
    ends = []
    for sgn in (1, -1):
        k = 0
        while k < max_r:
            q = p + sgn * n * k
            r, c = int(round(q[0])), int(round(q[1]))
            if r < 0 or r >= H or c < 0 or c >= W or not region[r, c]:
                break
            k += 1
        ends.append(sgn * (k - 1))
    return float(ends[0] - ends[1]), float(0.5 * (ends[0] + ends[1]))


def arm_chain(L, S, lm, roles, side, log=print):
    """side is 'R' (image-left on the front panel) or 'L'."""
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    c0, c1 = L.panels['front']
    mid = float(pan['front']['midline'])
    prof = lm['_profile']
    T, X0, X1 = prof['T'], prof['X0'], prof['X1']
    r_sh, r_ap = lm['shoulder']['row'], lm['armpit']['row']
    floor = int(cal['floor_row'])
    Hpx = floor - lm['crown']['row']
    fig = L.figure('front')
    skin = L.mask(roles.get('skin', []), 'front')
    acc = L.mask(roles.get('accessory', []) + roles.get('hair', []), 'front')
    r_crotch = lm['crotch']['row']
    # the arm is the figure outside the torso edge, which above the armpit runs from the shoulder corner
    cand = fig & ~acc
    cand[:r_sh, :] = False
    cand[int(r_crotch + 0.05 * Hpx):, :] = False
    x_sh = X0[r_sh] if side == 'R' else X1[r_sh]
    x_ap = X0[r_ap] if side == 'R' else X1[r_ap]
    for r in range(r_sh, int(r_crotch + 0.05 * Hpx)):
        if r < r_ap:
            t = (r - r_sh) / max(1, r_ap - r_sh)
            xe = int(x_sh + (x_ap - x_sh) * t)
        elif r in X0:
            xe = X0[r] if side == 'R' else X1[r]
        else:
            xe = int(mid)
        if side == 'R':
            cand[r, xe:] = False
        else:
            cand[r, :xe] = False
    seed = np.zeros_like(cand)
    for r in range(r_ap, min(r_ap + 12, cand.shape[0])):
        if r not in T:
            continue
        runs = [ab for ab in T[r] if (ab[1] <= X0[r] if side == 'R' else ab[0] >= X1[r])] if r in X0 else []
        if runs:
            ab = max(runs, key=lambda q: q[1]) if side == 'R' else min(runs, key=lambda q: q[0])
            seed[r, ab[0]:ab[1]] = True
    comp, n = ndi.label(cand, structure=np.ones((3, 3), int))
    ids = np.unique(comp[seed & (comp > 0)])
    if len(ids) == 0:
        raise RuntimeError('arm %s: no connected region at the armpit' % side)
    region = np.isin(comp, ids)
    region = ndi.binary_opening(region, iterations=2)
    comp, n = ndi.label(region, structure=np.ones((3, 3), int))
    if n == 0:
        raise RuntimeError('arm %s: empty region' % side)
    sizes = ndi.sum(np.ones_like(region), comp, index=np.arange(1, n + 1))
    region = comp == (int(np.argmax(sizes)) + 1)
    sk = skeletonize(ndi.binary_closing(region, iterations=3))
    path = diameter_path(sk)
    if len(path) < 10:
        raise RuntimeError('arm %s: skeleton too short' % side)
    if path[0][0] > path[-1][0]:
        path = path[::-1]
    P, s = resample(path, 6.0)
    widths, centres, cls = [], [], []
    for i in range(len(P)):
        j0, j1 = max(0, i - 2), min(len(P) - 1, i + 2)
        tg = P[j1] - P[j0]
        tg = tg / (np.linalg.norm(tg) + 1e-9)
        w, cofs = perp_width(region, P[i], tg)
        widths.append(w)
        centres.append(cofs)
        r, c = int(round(P[i][0])), int(round(P[i][1]))
        cls.append('skin' if skin[r, c] else L.name_of(int(L.lab[r, c])))
    widths = np.array(widths)
    ws = np.convolve(np.pad(widths, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    n = len(P)
    out = dict(side=side, axis_px=P.tolist(), s_px=s.tolist(), width_px=widths.tolist(), cls=cls)
    tip = n - 1
    skin_idx = [i for i in range(n) if cls[i] == 'skin']
    hand_start = None
    if skin_idx:
        blk = [skin_idx[-1]]
        for i in reversed(skin_idx[:-1]):
            if blk[-1] - i <= 2:
                blk.append(i)
            else:
                break
        hand_start = blk[-1]
    # wrist: where a glove starts, where skin starts after a sleeve, or the narrowest point before the palm
    import re as _re
    hand_cls = [i for i in range(int(0.4 * n), n) if _re.search(r'glove|mitt|gauntlet|paw', cls[i])]
    if hand_cls:
        wrist = hand_cls[0]
        wrule = 'first sample of the hand covering (%s)' % cls[wrist]
    elif hand_start is not None and hand_start > 0.5 * n and cls[max(0, hand_start - 3)] != 'skin':
        wrist = hand_start
        wrule = 'first skin sample after the sleeve'
    else:
        hw0 = int(0.6 * n) + int(np.argmax(ws[int(0.6 * n):]))
        lo = int(0.45 * n)
        wrist = lo + int(np.argmin(ws[lo:hw0 + 1]))
        wrule = 'narrowest sample before the hand widest (45 %% .. %d %% of the axis)' % int(100 * hw0 / n)
    # a hand is at most 12 % of the figure height
    h_fig = F.floor - lm['crown']['row'] if isinstance(lm, dict) and 'crown' in lm else None
    if h_fig:
        s_min = s[tip] - 0.12 * h_fig
        if s[wrist] < s_min:
            wrist = int(np.argmin(np.abs(s - s_min)))
            wrule += '; moved down: a hand is at most 12 % of the height'
    hand_widest = wrist + int(np.argmax(ws[wrist:tip + 1]))
    bare = [i for i in range(int(0.15 * n), wrist) if cls[i] == 'skin']
    if len(bare) > 0.3 * wrist:
        lo, hi = int(0.30 * wrist), int(0.80 * wrist)
        elbow = lo + int(np.argmin(ws[lo:hi]))
        erule = 'narrowest BARE sample between 30 % and 80 % of shoulder->wrist'
    else:
        elbow = int(round(0.47 * wrist))
        erule = 'CONV: 0.47 of shoulder->wrist (sleeve covers the elbow)'
    def rec(i, rule):
        r, c = P[i]
        return dict(i=int(i), s_px=float(s[i]), row=float(r), col=float(c), z=round(F.z(r), 5),
                    x=round((c - mid) / cal['ppm'], 5), w_mm=round(F.mm(widths[i]), 1), rule=rule)
    out['landmarks'] = dict(root=rec(0, 'axis start at the shoulder'), elbow=rec(elbow, erule), wrist=rec(wrist, wrule),
                            hand_widest=rec(hand_widest, 'widest sample between wrist and tip'), tip=rec(tip, 'axis end'))
    d = P[min(n - 1, int(0.6 * n))] - P[int(0.1 * n)]
    out['angle_deg'] = float(np.degrees(np.arctan2(abs(d[1]), d[0])))  # from vertical-down
    out['length_m'] = float(F.m(s[-1]))
    log('  arm %s: length %.3f m, angle %.0f deg from vertical, root row %d, elbow s %.0f (%s), wrist s %.0f (%s), tip row %d' % (
        side, out['length_m'], out['angle_deg'], P[0][0], s[elbow], erule.split(':')[0], s[wrist], wrule.split(' ')[0], P[-1][0]))
    return out, region
