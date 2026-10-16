"""Turn a flat-colour part map and its prompt into the reader's sheet description.

    py sketch3d/read/protocol.py <parts.png> <turn.png> <outdir> --prompt prompt_parts.txt [--height 1.6] [--name punk]
"""
import os, sys, json, re
import numpy as np
from PIL import Image

BAND_MIN_M = 0.004
BAND_ROW_SHARE = 0.75
BAND_SAME_RGB = 60.0
DETAIL_RGB = 90.0  # RGB distance of a mark from its part colour
DETAIL_MIN_M = 0.008


def components(mask):
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    return lab, n


def read_legend(img, n_parts, log=print):
    """Legend swatches in the right 40 % of the image, as (swatches, swatch_mask)."""
    from scipy import ndimage as ndi
    H, W, _ = img.shape
    lum = img.mean(2)
    x0 = int(W * 0.6)
    # a swatch has a dark outline, so a white swatch is found too
    dark = (lum < 110)[:, x0:]
    lab, n = components(dark)
    sw = []
    sw_mask = np.zeros((H, W), bool)
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        h_, w_ = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if h_ < 14 or w_ < 14 or h_ > 0.08 * H or w_ > 0.08 * H:
            continue
        if not (0.8 < w_ / float(h_) < 1.25):
            continue
        ring_fill = len(ys) / float(h_ * w_)
        box = np.zeros((H, W), bool)
        box[ys.min():ys.max() + 1, xs.min() + x0:xs.max() + 1 + x0] = True
        inner = ndi.binary_erosion(box, iterations=max(4, int(0.12 * h_)))
        if inner.sum() < 30:
            continue
        iy, ix = np.nonzero(inner)
        px = img[iy, ix]
        if ring_fill > 0.85:
            solid = True
        else:
            solid = ring_fill < 0.6 and px.std(0).max() < 14
        if not solid:
            continue
        sw.append((float(ys.mean()), float(xs.mean() + x0), px.mean(0), int(len(ys))))
        sw_mask |= box
    sw.sort(key=lambda q: q[0])
    if sw:
        cx = float(np.median([q[1] for q in sw]))
        keep = [q for q in sw if abs(q[1] - cx) < 0.03 * W]
        sw = keep
    log('[protocol] legend: %d swatches found (%d parts expected)' % (len(sw), n_parts))
    return sw, sw_mask


def find_figures(img, legend_mask, legend_cx, log=print, swatch_rgb=None, floor_line=True):
    """Return the figure blobs and the floor line row."""
    from scipy import ndimage as ndi
    H, W, _ = img.shape
    lum = img.mean(2)
    nonwhite = (lum < 240) | (img.min(2) < 230)
    if swatch_rgb:
        # a near-white part is told from the background by the nearest colour
        cols = np.array(list(swatch_rgb) + [[255.0, 255.0, 255.0]], np.float32)
        best = np.full((H, W), np.inf, np.float32)
        near = np.zeros((H, W), np.int32)
        for k in range(len(cols)):
            d = np.sqrt(((img - cols[k]) ** 2).sum(2))
            upd = d < best
            best[upd] = d[upd]
            near[upd] = k
        is_part = (near < len(cols) - 1) & (best < 30)
        nonwhite = nonwhite | is_part
        nonwhite &= ~((near == len(cols) - 1) & (best < 6))
    region = nonwhite & ~ndi.binary_dilation(legend_mask, iterations=3)
    if legend_cx is not None:
        lab, n = components(region[:, int(legend_cx):int(min(W, legend_cx + 0.06 * W))])
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) < 1500:
                region[ys, xs + int(legend_cx)] = False
    rowfrac = region.mean(1)
    # a prop sheet has no floor line, a wide object would pass for one
    floor_rows = [r for r in range(6, H - 6) if rowfrac[r] > 0.55 and max(rowfrac[r - 6], rowfrac[r + 6]) < 0.5 * rowfrac[r]] if floor_line else []
    floor = None
    if floor_rows:
        floor = int(max(floor_rows))
        band = [r for r in floor_rows if floor - 8 <= r <= floor]
        floor = int(min(band))
    body = region.copy()
    if floor is not None:
        body[max(0, floor - 3):floor + 12, :] = False
    # closing by 6 px joins a detached spike or hand to its figure
    blob = ndi.binary_closing(body, structure=np.ones((3, 3), bool), iterations=6)
    lab, n = components(blob)
    sizes = ndi.sum(np.ones_like(blob), lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    figs = []
    for idx in order[:6]:
        i = idx + 1
        if sizes[idx] < 0.002 * H * W:
            continue
        ys, xs = np.nonzero(lab == i)
        figs.append(dict(c0=int(xs.min()), c1=int(xs.max()) + 1, r0=int(ys.min()), r1=int(ys.max()), cx=float(xs.mean()), area=int(sizes[idx]), mask=(lab == i)))
    figs = split_touching(figs, log)
    figs.sort(key=lambda f: f['cx'])
    h_max = max(f['r1'] - f['r0'] for f in figs) if figs else 1
    tall = [f for f in figs if (f['r1'] - f['r0']) >= 0.6 * h_max][:3]
    short = [f for f in figs if f not in tall and f['cx'] > tall[-1]['cx']] if tall else []
    figs = tall + short[:1]
    log('[protocol] figures: %d blobs %s, floor row %s' % (len(figs), [(f['c0'], f['c1'], f['r0'], f['r1']) for f in figs], floor))
    return figs, floor, body


def split_touching(figs, log=print):
    """Cut a blob of two figures touching hands at its thinnest column."""
    tall = [f for f in figs if (f['r1'] - f['r0']) >= 0.6 * max(g['r1'] - g['r0'] for g in figs)]
    if len(tall) < 2:
        return figs
    w_med = float(np.median([f['c1'] - f['c0'] for f in tall]))
    out = []
    for f in figs:
        w = f['c1'] - f['c0']
        if f not in tall or w < 1.4 * w_med:
            out.append(f)
            continue
        count = f['mask'][:, f['c0']:f['c1']].sum(0)
        lo, hi = int(0.25 * w), int(0.85 * w)
        cut = lo + int(np.argmin(count[lo:hi]))
        if count[cut] > 0.1 * count.max():
            out.append(f)
            continue
        for a, b in ((f['c0'], f['c0'] + cut), (f['c0'] + cut + 1, f['c1'])):
            m = np.zeros_like(f['mask'])
            m[:, a:b] = f['mask'][:, a:b]
            ys, xs = np.nonzero(m)
            if len(ys):
                out.append(dict(c0=int(xs.min()), c1=int(xs.max()) + 1, r0=int(ys.min()), r1=int(ys.max()), cx=float(xs.mean()), area=int(len(ys)), mask=m))
        log('[protocol] a blob %d px wide (median figure %d) cut at column %d: two figures touching' % (w, w_med, f['c0'] + cut))
    return out


def main(argv):
    parts_png, turn_png, out = argv[0], argv[1], argv[2]
    names = [q.strip() for q in argv[argv.index('--parts') + 1].split(',')] if '--parts' in argv else []
    height = float(argv[argv.index('--height') + 1]) if '--height' in argv else 1.6
    name = argv[argv.index('--name') + 1] if '--name' in argv else os.path.splitext(os.path.basename(parts_png))[0].replace('_parts', '')
    os.makedirs(out, exist_ok=True)
    log = print
    img = np.asarray(Image.open(parts_png).convert('RGB')).astype(np.float32)
    art = np.asarray(Image.open(turn_png).convert('RGB')).astype(np.float32)
    H, W, _ = img.shape
    # prompt colours win, each refined to the median of image pixels within 40
    given = []
    if '--prompt' in argv:
        for line in open(argv[argv.index('--prompt') + 1], encoding='utf-8'):
            m = re.match(r'\s*(\d+)\s+(.+?)\s*=\s*#([0-9A-Fa-f]{6})\s*$', line)
            if m:
                given.append((int(m.group(1)), m.group(2).strip(), [int(m.group(3)[k:k + 2], 16) for k in (0, 2, 4)]))
    if given:
        given.sort()
        names = [g[1] for g in given]
    sw, sw_mask = read_legend(img, len(names), log=log)
    if given:
        legend_cx0 = float(np.median([q[1] for q in sw])) if sw else 0.8 * W
        fig_cols = slice(0, int(legend_cx0 - 0.04 * W))
        sw2 = []
        for k, (num, pname, rgb) in enumerate(given):
            d = np.sqrt(((img[:, fig_cols] - np.array(rgb, np.float32)) ** 2).sum(2))
            m = d < 40
            if m.sum() >= 20:
                rgb_ref = [float(v) for v in np.median(img[:, fig_cols][m], axis=0)]
            else:
                rgb_ref = [float(v) for v in rgb]
                log('[protocol] part %d %s: colour #%02X%02X%02X not found in the figures (drawn elsewhere or missing)' % (num, pname, *rgb))
            row = sw[k][0] if k < len(sw) else int(H * (0.1 + 0.8 * k / max(1, len(given))))
            sw2.append((row, legend_cx0, rgb_ref))
        sw = sw2
        log('[protocol] %d part colours from the prompt (legend: %d swatches seen)' % (len(sw), len(sw_mask.nonzero()[0]) and len(sw2)))
    elif len(sw) != len(names):
        log('[protocol] WARNING: %d swatches vs %d names - the first %d are paired in order' % (len(sw), len(names), min(len(sw), len(names))))
    n = min(len(sw), len(names))
    legend_cx = float(np.median([q[1] for q in sw])) if sw else None
    int(legend_cx - 0.02 * W) if sw else int(0.8 * W)
    figs, floor, body = find_figures(img, sw_mask, legend_cx, log=log, swatch_rgb=[q[2] for q in sw])
    if floor is None:
        floor = int(max(np.nonzero(body.any(1))[0]))
    classes = []
    colours = {}
    cls_of_part = {}
    for i in range(n):
        pname = names[i]
        rgb = [int(round(c)) for c in sw[i][2]]
        # the class name is the part's head noun, without the explanation
        cname = re.split(r'\(| on | of | and | with ', pname.lower())[0].strip()
        cname = re.sub(r'^(left|right)\s+', '', cname).strip()
        cname = re.sub(r'[^a-z0-9]+', '_', cname).strip('_')
        col_name = 'p%d' % (i + 1)
        colours[col_name] = rgb
        cls_of_part[i] = cname
        for c in classes:
            if c[0] == cname:
                c[1].append(col_name)
                break
        else:
            classes.append([cname, [col_name]])
    colours['white'] = [255, 255, 255]
    panel_names = ['front', 'side', 'back', 'top']
    panels = []

    def side_facing(f):
        """Return which way the profile faces, read from the feet."""
        mk = f['mask']
        r0, r1 = f['r0'], f['r1']
        h = r1 - r0
        shin = mk[int(r1 - 0.25 * h):int(r1 - 0.15 * h)]
        feet = mk[int(r1 - 0.08 * h):r1 + 1]
        cs = np.nonzero(shin.any(0))[0]
        cf = np.nonzero(feet.any(0))[0]
        if len(cs) == 0 or len(cf) == 0:
            return 'left'
        c_shin = 0.5 * (cs.min() + cs.max())
        ext_r = cf.max() - c_shin
        ext_l = c_shin - cf.min()
        return 'right' if ext_r > ext_l * 1.15 else ('left' if ext_l > ext_r * 1.15 else 'left')
    for k, f in enumerate(figs[:4]):
        pn = panel_names[k]
        c0, c1 = f['c0'], f['c1']
        facing_side = side_facing(f) if pn == 'side' else None
        if pn == 'side':
            log('[protocol] the profile figure faces %s (its feet point that way)' % facing_side.upper())
        lo = figs[k - 1]['c1'] if k > 0 else 0
        hi = figs[k + 1]['c0'] if k + 1 < len(figs) else W
        c0p = int(max(lo, c0 - 0.01 * W)) if k > 0 else int(max(0, c0 - 0.02 * W))
        c1p = int(min(hi, c1 + 0.01 * W)) if k + 1 < len(figs) else int(min(W, c1 + 0.02 * W))
        panels.append(dict(name=pn, c0=c0p, c1=c1p, midline=0.5 * (c0 + c1),
                           sign=(1 if pn not in ('back', 'side') else (-1 if pn == 'back' else (1 if facing_side == 'left' else -1))),
                           facing={'front': 'the figure FACES THE VIEWER', 'side': 'the figure FACES %s' % (facing_side or 'left').upper(),
                                   'back': 'the figure FACES AWAY', 'top': 'seen from ABOVE, the face toward the bottom'}[pn],
                           axis={'front': 'lateral', 'side': 'depth', 'back': 'lateral', 'top': 'plan'}[pn],
                           r0=int(f['r0']), r1=int(f['r1'])))
    front = panels[0]
    ppm = (floor - front['r0']) / height
    log('[protocol] front figure rows %d..%d, floor %d -> %.1f px/m for %.2f m' % (front['r0'], floor, floor, ppm, height))
    cal = dict(ppm=ppm, floor_row=float(floor), panels=[p for p in panels if p['name'] != 'top'],
               axis_ends={'lateral': ["character's RIGHT", "character's LEFT"], 'depth': ["character's FRONT", "character's BACK"]},
               side_anchor_rows=[int(floor - 0.26 * (floor - front['r0'])), int(floor - 0.16 * (floor - front['r0']))],
               side_anchor_why='lower leg band (protocol default)')
    # register the art onto the part map so later steps read both at the same pixels
    art = register_art(art, figs[:3], img.shape, log)
    art_png = os.path.join(out, name + '_art.png')
    Image.fromarray(np.clip(art, 0, 255).astype(np.uint8)).save(art_png)
    # a part's art colour is the one most over-represented under its mask
    art_colours = {}
    if art.shape == img.shape:
        fig = (np.abs(art - 255).sum(2) > 40) & ~sw_mask
        fig[:, int(legend_cx - 0.03 * W):] = False
        from scipy import ndimage as ndi
        art_fill = ndi.binary_fill_holes(ndi.binary_closing(art.min(2) < 225, iterations=2))
        q = (art // 32).astype(np.int32)
        key = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        all_counts = np.bincount(key[fig], minlength=512).astype(np.float64)
        all_p = all_counts / max(1.0, all_counts.sum())
        for i in range(n):
            rgb = np.array(sw[i][2], np.float32)
            m = (np.sqrt(((img - rgb) ** 2).sum(2)) < 18)
            m &= ~sw_mask
            m = np.logical_and(m, ndi_erode(m, 3))
            if (m & art_fill).sum() >= 30:
                m &= art_fill
            if m.sum() < 30:
                continue
            if m.sum() < 200:
                px = art[m]
                px = px[(np.abs(px - 255).sum(1) > 40) & (px.mean(1) > 40)]
                if len(px):
                    art_colours[cls_of_part[i]] = [int(v) for v in np.median(px, axis=0)]
                continue
            km = key[m]
            cnt = np.bincount(km, minlength=512).astype(np.float64)
            pm = cnt / cnt.sum()
            cand = [b for b in range(512) if pm[b] >= 0.10]
            if not cand:
                cand = [int(np.argmax(pm))]
            best = max(cand, key=lambda b: pm[b] / (all_p[b] + 1e-4))
            px = art[m][km == best]
            art_colours[cls_of_part[i]] = [int(v) for v in np.median(px, axis=0)]
    else:
        log('[protocol] WARNING: turn and parts differ in size (%s vs %s): no art colours' % (art.shape, img.shape))
    art_bands = colour_bands(img, art, sw, cls_of_part, n, panels[0], floor, ppm, log) if art.shape == img.shape else {}
    skip = set(cls_of_part[i] for i in range(n) if i == 0 or re.search(r'skin|hair|mohawk|fringe|bang|ponytail|braid|stubble|beard', cls_of_part[i]))
    det_lab, art_details = detail_marks(img, art, sw, cls_of_part, n, panels, ppm, art_colours, skip, log)
    np.save(os.path.join(out, 'details.npy'), det_lab)
    parts = [dict(name=c[0], colours=c[1], bands=[[0, int(floor)]], panels=None, seed=None) for c in classes]
    sheet = dict(_='generated by sketch3d/read/protocol.py from the part map: no hand-typed number',
                 sheet=dict(name=os.path.basename(parts_png), width=int(W), height=int(H)),
                 calibration=cal,
                 palette=dict(colours=colours, tol=22, gate_check=[]),
                 parts=parts,
                 art=os.path.abspath(art_png), art_colours=art_colours, art_bands=art_bands, art_details=art_details,
                 top=[p for p in panels if p['name'] == 'top'],
                 legend=[dict(part=names[i], cls=cls_of_part[i], colour=[int(round(c)) for c in sw[i][2]], row=int(sw[i][0])) for i in range(n)])
    sp = os.path.join(out, name + '_sheet.json')
    json.dump(sheet, open(sp, 'w'), indent=1)
    roles = dict(skin=[], hair=[], shoe=[], sole=[], accessory=[])
    for c in classes:
        cn = c[0]
        if cn == 'skin':
            roles['skin'].append(cn)
        if re.search(r'hair|mohawk|fringe|bang|ponytail|braid|stubble|beard', cn):
            roles['hair'].append(cn)
        elif re.search(r'sole|toe_cap', cn):
            roles['sole'].append(cn)
        elif re.search(r'shoe|boot|sneaker|sandal', cn):
            roles['shoe'].append(cn)
        elif re.search(r'pack|strap|stud|earring|bow|choker|necklace|chain|phones|badge|pin|buckle|bag|pouch|holster', cn):
            roles['accessory'].append(cn)
    if not roles['skin']:
        roles['skin'] = [classes[0][0]]
    bp = os.path.join(out, 'brief.json')
    json.dump(roles, open(bp, 'w'), indent=1)
    json.dump(dict(sheet=sp, brief=bp, turn=os.path.abspath(turn_png), parts=os.path.abspath(parts_png), classes=[c[0] for c in classes],
                   roles=roles, art_colours=art_colours, top=sheet['top'], height_m=height), open(os.path.join(out, name + '_protocol.json'), 'w'), indent=1)
    log('[protocol] classes: %s' % ', '.join(c[0] for c in classes))
    log('[protocol] roles: %s' % json.dumps(roles))
    log('[protocol] -> %s, %s' % (sp, bp))


def register_art(art, figs, shape, log=print):
    """Scale and shift each art figure onto its part map figure."""
    H, W = shape[:2]
    if art.shape[:2] != (H, W):
        art = np.asarray(Image.fromarray(art.astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32)
    ink = art.min(2) < 225
    line = np.zeros(H, bool)
    for r in np.nonzero(ink.mean(1) > 0.4)[0]:
        cols = np.nonzero(ink[r])[0]
        breaks = np.nonzero(np.diff(cols) > 1)[0]
        runs = np.diff(np.r_[-1, breaks, len(cols) - 1])
        line[r] = runs.max() > 0.4 * W
    ink[np.convolve(line.astype(float), np.ones(7), mode='same') > 0] = False
    cols = np.nonzero(ink.sum(0) > 2)[0]
    if not len(cols):
        return art
    breaks = np.nonzero(np.diff(cols) > 0.02 * W)[0]
    groups = []
    for c0, c1 in zip(np.r_[cols[0], cols[breaks + 1]], np.r_[cols[breaks], cols[-1]] + 1):
        rows = np.nonzero(ink[:, c0:c1].any(1))[0]
        groups.append((int(c0), int(c1), int(rows[0]), int(rows[-1])))
    h_max = max(g[3] - g[2] for g in groups)
    tall = [g for g in groups if g[3] - g[2] >= 0.6 * h_max][:len(figs)]
    if len(tall) < len(figs):
        log('[protocol] art: %d separate figures, %d expected - each read in a window round its part map figure' % (len(tall), len(figs)))
        tall = []
        for f in figs:
            pad = int(0.1 * (f['c1'] - f['c0']))
            a0, a1 = max(0, f['c0'] - pad), min(W, f['c1'] + pad)
            rows = np.nonzero(ink[:, a0:a1].any(1))[0]
            if not len(rows):
                return art
            tall.append((a0, a1, int(rows[0]), int(rows[-1])))

    def hips_mid(mask, r0, r1):
        band = mask[int(r0 + 0.7 * (r1 - r0)):int(r0 + 0.95 * (r1 - r0))]
        return float(np.median(np.nonzero(band)[1]))
    from scipy import ndimage as ndi
    silhouette = ndi.binary_fill_holes(ndi.binary_closing(ink, iterations=2))

    def overlap(f, ar0, s, am, pm):
        rr = np.arange(f['r0'], f['r1'] + 1, 4)
        cc = np.arange(f['c0'], f['c1'] + 1, 4)
        rs = np.clip(np.round(ar0 + (rr - f['r0']) * s).astype(int), 0, H - 1)
        cs = np.round(am + (cc - pm) * s).astype(int)
        inside = (cs >= 0) & (cs < W)
        a = silhouette[np.ix_(rs, np.clip(cs, 0, W - 1))] & inside[None, :]
        b = f['mask'][np.ix_(rr, cc)]
        return (a & b).sum() / float(max(1, (a | b).sum()))
    line_rows = np.nonzero(line)[0]
    out = np.full_like(art, 255.0)
    for k, (f, (ac0, ac1, ar0, ar1)) in enumerate(zip(figs, tall)):
        under = line_rows[(line_rows > ar1) & (line_rows <= ar1 + 10)]
        if len(under):
            ar1 = int(line_rows[line_rows >= under[0]][np.r_[np.diff(line_rows[line_rows >= under[0]]) > 1, True]][0])
        s = (ar1 - ar0) / float(max(1, f['r1'] - f['r0']))
        pm = hips_mid(f['mask'], f['r0'], f['r1'])
        a_mask = np.zeros_like(ink)
        a_mask[:, ac0:ac1] = ink[:, ac0:ac1]
        am = hips_mid(a_mask, ar0, ar1)
        w = f['c1'] - f['c0']
        cands = set(range(int(am - 0.15 * w), int(am + 0.15 * w) + 1, 2)) | set(range(int(pm - 0.05 * w), int(pm + 0.05 * w) + 1, 2))
        am = max(sorted(cands), key=lambda a_: overlap(f, ar0, s, a_, pm))
        lo = 0 if k == 0 else (figs[k - 1]['c1'] + f['c0']) // 2
        hi = W if k == len(figs) - 1 else (f['c1'] + figs[k + 1]['c0']) // 2
        rr = np.arange(H)
        cc = np.arange(lo, hi)
        r_src = np.round(ar0 + (rr - f['r0']) * s).astype(int)
        c_src = np.round(am + (cc - pm) * s).astype(int)
        vr, vc = (r_src >= 0) & (r_src < H), (c_src >= 0) & (c_src < W)
        out[np.ix_(rr[vr], cc[vc])] = art[np.ix_(r_src[vr], c_src[vc])]
        log('[protocol] art figure %d: scale %.3f, shift %+d px, %+d px' % (k, s, int(round(am - pm)), int(ar0 - f['r0'])))
    return out


def detail_marks(img, art, sw, cls_of_part, n, panels, ppm, art_colours, skip, log=print):
    """Marks drawn inside a part (eyes on a visor, an emblem), as a label image and a list."""
    from scipy import ndimage as ndi
    H, W = img.shape[:2]
    lab = np.zeros((H, W), np.int16)
    out = []
    if art.shape != img.shape:
        return lab, out
    min_px = (DETAIL_MIN_M * ppm) ** 2
    r_open = max(1, int(round(0.25 * DETAIL_MIN_M * ppm)))
    for i in range(n):
        cname = cls_of_part[i]
        if cname in skip or cname not in art_colours:
            continue
        full = np.sqrt(((img - np.array(sw[i][2], np.float32)) ** 2).sum(2)) < 18
        m = ndi_erode(full, 3)
        d = np.sqrt(((art - np.array(art_colours[cname], np.float32)) ** 2).sum(2))
        det = ndi.binary_opening(m & (d > DETAIL_RGB), iterations=r_open)
        comp, k = ndi.label(det)
        for j in range(1, k + 1):
            b = comp == j
            if b.sum() < min_px or (ndi.binary_dilation(b, iterations=3) & ~full).any():
                continue
            ys, xs = np.nonzero(b)
            pad = max(8, int(0.5 * max(ys.max() - ys.min(), xs.max() - xs.min())))
            r0, r1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
            c0, c1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
            rgb = np.median(art[b], axis=0)
            same = np.sqrt(((art[r0:r1, c0:c1] - rgb) ** 2).sum(2)) < 0.5 * DETAIL_RGB
            patch, _ = ndi.label(same | b[r0:r1, c0:c1])
            ids = np.unique(patch[b[r0:r1, c0:c1]])
            if (np.isin(patch, ids[ids > 0]) & ~full[r0:r1, c0:c1]).sum() > 0.05 * b.sum():
                continue
            xs = np.nonzero(b)[1]
            view = next((p['name'] for p in panels if p['c0'] <= xs.mean() < p['c1']), None)
            if view not in ('front', 'side', 'back'):
                continue
            out.append(dict(id=len(out) + 1, cls=cname, view=view, rgb=[int(v) for v in np.median(art[b], axis=0)],
                            area_mm2=int(round(b.sum() / ppm / ppm * 1e6))))
            lab[b] = len(out)
    if out:
        log('[protocol] %d marks drawn inside parts: %s' % (len(out), ', '.join('%s %s %d mm2' % (q['cls'], q['view'], q['area_mm2']) for q in out)))
    return lab, out


def colour_bands(img, art, sw, cls_of_part, n, front, floor, ppm, log=print):
    """Return {class: [[z_top, z_bottom, rgb], ...]} for parts drawn in stripes."""
    c0, c1 = front['c0'], front['c1']
    sub_img, sub_art = img[:, c0:c1], art[:, c0:c1]
    q = (sub_art // 32).astype(np.int32)
    key = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    min_rows = max(2, int(round(BAND_MIN_M * ppm)))
    out = {}
    for i in range(n):
        cname = cls_of_part[i]
        if cname in out:
            continue
        m = ndi_erode(np.sqrt(((sub_img - np.array(sw[i][2], np.float32)) ** 2).sum(2)) < 18, 3)
        rows = [r for r in range(m.shape[0]) if m[r].sum() >= 6]
        if len(rows) < 3 * min_rows:
            continue
        per_row = []
        for r in rows:
            cnt = np.bincount(key[r][m[r]], minlength=512)
            b = int(cnt.argmax())
            per_row.append((r, b, cnt[b] / float(cnt.sum())))
        if np.mean([s for _, _, s in per_row]) < BAND_ROW_SHARE:
            continue
        runs = []
        for r, b, _ in per_row:
            if runs and runs[-1][0] == b and r - runs[-1][2] <= 2:
                runs[-1][2] = r
            else:
                runs.append([b, r, r])
        merged = []
        for b, ra, rb in runs:
            if merged and (rb - ra + 1 < min_rows or merged[-1][0] == b):
                merged[-1][2] = rb
            else:
                merged.append([b, ra, rb])
        if merged and merged[0][2] - merged[0][1] + 1 < min_rows and len(merged) > 1:
            merged[1][1] = merged[0][1]
            merged.pop(0)
        bands = []
        for b, ra, rb in merged:
            sel = m[ra:rb + 1] & (key[ra:rb + 1] == b)
            col = np.median(sub_art[ra:rb + 1][sel], axis=0)
            if bands and np.linalg.norm(col - bands[-1][2]) < BAND_SAME_RGB:
                bands[-1][1] = rb
            else:
                bands.append([ra, rb, col])
        clusters = []
        for _, _, col in bands:
            for cl in clusters:
                if np.linalg.norm(col - cl[0]) < BAND_SAME_RGB:
                    cl[1] += 1
                    break
            else:
                clusters.append([col, 1])
        if sum(1 for cl in clusters if cl[1] >= 2) < 2:
            continue
        out[cname] = [[round((floor - ra) / ppm, 4), round((floor - rb - 1) / ppm, 4), [int(v) for v in col]] for ra, rb, col in bands]
        bands = out[cname]
        log('[protocol] %s: %d colour bands by height (%s)' % (cname, len(bands), ', '.join('#%02X%02X%02X' % tuple(c) for _, _, c in bands)))
    return out


def ndi_erode(m, n):
    from scipy import ndimage as ndi
    return ndi.binary_erosion(m, iterations=n)


if __name__ == '__main__':
    main(sys.argv[1:])
