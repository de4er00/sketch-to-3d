"""Trace the drawn face (eyes, brows, nose, mouth, blush) into polygons in face-plane metres.

Read from the art, not the label map: thin lines vanish into the skin class there.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from . import geometry as G


def _poly(mask, tol_px=1.2):
    cs = measure.find_contours(np.pad(mask.astype(float), 1), 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1.0
    return measure.approximate_polygon(c, tol_px)


def face_read(img, L, S, lm, roles, log=print):
    cal = S['calibration']
    F = G.Frame(cal['ppm'], cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    mid = float(pan['front']['midline'])
    c0, c1 = L.panels['front']
    r0 = lm['hairline']['row'] - 10
    r1 = lm['chin']['row'] + 8
    skin = L.mask(roles['skin'], 'front')
    rows = [r for r in range(r0, r1) if skin[r, c0:c1].any()]
    if not rows:
        raise RuntimeError('face: no skin rows')
    cols = np.nonzero(skin[r0:r1, c0:c1].any(0))[0] + c0
    cl, cr = int(cols.min()), int(cols.max()) + 1
    box = dict(r0=int(r0), r1=int(r1), c0=int(cl), c1=int(cr))
    sub = img[r0:r1, cl:cr].astype(np.float32)
    lab = L.lab[r0:r1, cl:cr]
    sk = skin[r0:r1, cl:cr]
    hair_mask = np.isin(lab, [L.ids[c] for c in roles.get('hair', []) if c in L.ids]) if roles.get('hair') else None
    # erode so the skin-labelled outline stays outside
    region = ndi.binary_erosion(sk, iterations=4)
    region = ndi.binary_fill_holes(region)
    lum = sub.mean(2)
    pal = S['palette']['colours']
    skin_rgb = np.array([pal[c] for c in S['parts'][[p['name'] for p in S['parts']].index(roles['skin'][0])]['colours']][0], np.float32)
    hair_rgb = np.array(pal[S['parts'][[p['name'] for p in S['parts']].index(roles['hair'][0])]['colours'][0]], np.float32) if roles.get('hair') else None
    ac = S.get('art_colours') or {}
    if ac.get(roles['skin'][0]):
        skin_rgb = np.array(ac[roles['skin'][0]], np.float32)
    if roles.get('hair') and ac.get(roles['hair'][0]):
        hair_rgb = np.array(ac[roles['hair'][0]], np.float32)
    dist_skin = np.sqrt(((sub - skin_rgb) ** 2).sum(2))
    dark = (lum < 90) & region
    white = (lum > 225) & (sub.min(2) > 205) & region
    coloured = (dist_skin > 60) & ~dark & ~white & region
    border = region & ~ndi.binary_erosion(region, iterations=2)
    dark &= ~ndi.binary_dilation(border, iterations=1)
    if hair_rgb is not None:
        hairish = np.sqrt(((sub - hair_rgb) ** 2).sum(2)) < 60
    else:
        hairish = np.zeros_like(dark)
    H, W = dark.shape
    cx = mid - cl
    feats = []

    def comps(mask, minpx):
        lab_, n = ndi.label(mask, structure=np.ones((3, 3), int))
        out = []
        for i in range(1, n + 1):
            m = lab_ == i
            px = int(m.sum())
            if px < minpx:
                continue
            rr, cc = np.nonzero(m)
            out.append(dict(mask=m, px=px, r=(rr.min(), rr.max() + 1), c=(cc.min(), cc.max() + 1), cen=(rr.mean(), cc.mean())))
        return out
    # thick dark fills are eyes, thin dark lines are brows, nose and mouth
    thick = ndi.binary_opening(dark & ~hairish, structure=np.ones((3, 3), bool), iterations=2)
    thick = ndi.binary_dilation(thick, iterations=2) & dark
    thin = dark & ~ndi.binary_dilation(thick, iterations=1)
    dk = comps(thick, 25)
    dk.sort(key=lambda d: -d['px'])
    eyes = []
    for d in dk:
        if len(eyes) == 2:
            break
        if eyes and abs((d['cen'][1] - cx) + (eyes[0]['cen'][1] - cx)) > 0.25 * W:
            continue
        if eyes and abs(d['cen'][0] - eyes[0]['cen'][0]) > 0.12 * H:
            continue
        eyes.append(d)
    eye_row = np.mean([e['cen'][0] for e in eyes]) if eyes else 0.4 * H
    used = set(id(e) for e in eyes)
    brows, mouth, nose, other = [], None, None, []
    for d in dk + comps(thin, 20):
        if id(d) in used:
            continue
        w_, h_ = d['c'][1] - d['c'][0], d['r'][1] - d['r'][0]
        if d['cen'][0] < eye_row - 0.03 * H and w_ > 1.5 * h_:
            brows.append(d)
        elif d['cen'][0] > eye_row + 0.12 * H and abs(d['cen'][1] - cx) < 0.25 * W and w_ >= h_:
            if mouth is None or d['px'] > mouth['px']:
                if mouth is not None:
                    other.append(mouth)
                mouth = d
        elif eye_row < d['cen'][0] and abs(d['cen'][1] - cx) < 0.12 * W and d['px'] < 400:
            if nose is None or d['px'] > nose['px']:
                nose = d
        else:
            other.append(d)
    hb = comps(ndi.binary_opening(hairish & region, iterations=1), 20)
    # brows are small fills just above the eyes, bigger ones are hair
    hb = [d for d in hb if eye_row - 0.12 * H < d['cen'][0] < eye_row - 0.02 * H
          and (d['c'][1] - d['c'][0]) > 1.3 * (d['r'][1] - d['r'][0])
          and (d['r'][1] - d['r'][0]) < 0.06 * H and (d['c'][1] - d['c'][0]) < 0.40 * W
          and d['r'][0] > 2]
    dark_near = [d for d in brows if d['cen'][0] > eye_row - 0.12 * H]
    hb = [d for d in hb if not (ndi.binary_dilation(d['mask'], iterations=3) & hair_mask).any()] if hair_mask is not None else []
    if hb and not dark_near:
        brows = hb
        for d in brows:
            d['mask'] = ndi.binary_dilation(d['mask'], iterations=2) & (d['mask'] | dark)
    scleras = []
    wc = comps(white, 12)
    for e in eyes:
        near = [w for w in wc if abs(w['cen'][0] - e['cen'][0]) < 0.08 * H and abs(w['cen'][1] - e['cen'][1]) < 0.12 * W]
        if near:
            m = np.zeros_like(dark)
            for w in near:
                m |= w['mask']
            rr, cc = np.nonzero(m)
            scleras.append(dict(mask=m, px=int(m.sum()), r=(rr.min(), rr.max() + 1), c=(cc.min(), cc.max() + 1), cen=(rr.mean(), cc.mean())))
    iris = None
    irises = []
    highlights = []
    for e, s_ in zip(eyes, scleras + [None] * 2):
        hull = ndi.binary_fill_holes(ndi.binary_closing(e['mask'] | (s_['mask'] if s_ else False), iterations=3))
        m = coloured & hull
        if m.sum() > 30:
            col = [int(v) for v in np.median(sub[m], 0)]
            iris = iris or col
            rr, cc = np.nonzero(m)
            irises.append(dict(mask=m, px=int(m.sum()), r=(rr.min(), rr.max() + 1), c=(cc.min(), cc.max() + 1), cen=(rr.mean(), cc.mean()), colour=col))
            hm = white & ndi.binary_dilation(m, iterations=2)
            for h in comps(hm, 6):
                if h['px'] < 0.5 * m.sum():
                    highlights.append(h)
    blush = [d for d in comps(coloured, 60) if d['cen'][0] > eye_row and abs(d['cen'][1] - cx) > 0.1 * W]

    def rec(d, kind, colour):
        p = _poly(d['mask'])
        if p is None:
            return None
        pts = [[round(float((cl + c + 0.5 - mid) / cal['ppm']), 4), round(float(F.z(r0 + r)), 4)] for r, c in p]
        return dict(kind=kind, colour=colour, px=d['px'], poly=pts,
                    centre=[round(float((cl + d['cen'][1] - mid) / cal['ppm']), 4), round(float(F.z(r0 + d['cen'][0])), 4)],
                    box=[round(float((cl + d['c'][0] - mid) / cal['ppm']), 4), round(float((cl + d['c'][1] - mid) / cal['ppm']), 4),
                         round(float(F.z(r0 + d['r'][1])), 4), round(float(F.z(r0 + d['r'][0])), 4)])
    for e in eyes:
        f = rec(e, 'eye', [20, 20, 24])
        if f:
            feats.append(f)
    for s_ in scleras:
        f = rec(s_, 'sclera', [255, 255, 255])
        if f:
            feats.append(f)
    for i_ in irises:
        f = rec(i_, 'iris', i_['colour'])
        if f:
            feats.append(f)
    for h in highlights:
        f = rec(h, 'highlight', [255, 255, 255])
        if f:
            feats.append(f)
    for b in brows:
        col = [int(v) for v in hair_rgb] if hair_rgb is not None and hairish[b['mask']].mean() > 0.5 else [20, 20, 24]
        f = rec(b, 'brow', col)
        if f:
            feats.append(f)
    if nose:
        f = rec(nose, 'nose', [20, 20, 24])
        if f:
            feats.append(f)
    if mouth:
        f = rec(mouth, 'mouth', [20, 20, 24])
        if f:
            feats.append(f)
    for b in blush:
        px = sub[b['mask']]
        f = rec(b, 'blush', [int(v) for v in np.median(px, 0)])
        if f:
            feats.append(f)
    out = dict(box_m=[round(float((cl - mid) / cal['ppm']), 4), round(float((cr - mid) / cal['ppm']), 4), round(float(F.z(r1)), 4), round(float(F.z(r0)), 4)],
               box_px=box, iris=iris, features=feats, skin_rgb=[int(v) for v in skin_rgb],
               eye_row_z=round(float(F.z(r0 + eye_row)), 4))
    log('  face: %d eyes, %d sclera, %d brows, nose %s, mouth %s, %d blush, iris %s' % (
        len(eyes), len(scleras), len(brows), 'yes' if nose else 'no', 'yes' if mouth else 'no', len(blush), iris))
    return out
