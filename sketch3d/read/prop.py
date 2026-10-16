"""Read a rigid prop sheet (front, side, top) into prop.json and labels.

    py sketch3d/read/prop.py <parts.png> <turn.png> <outdir> --prompt prompt_parts.txt --name chest --size 0.6
"""
import os, sys, json, re
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sketch3d.read.protocol import read_legend, find_figures, register_art   # noqa: E402


def parse_prompt(path):
    given = []
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\s*(\d+)\s+(.+?)\s*=\s*#([0-9A-Fa-f]{6})\s*$', line)
        if m:
            given.append((int(m.group(1)), m.group(2).strip(), [int(m.group(3)[k:k + 2], 16) for k in (0, 2, 4)]))
    given.sort()
    return given


def class_name(pname):
    c = re.split(r'\(| on | of | and | with ', pname.lower())[0].strip()
    c = re.sub(r'^(left|right)\s+', '', c).strip()
    return re.sub(r'[^a-z0-9]+', '_', c).strip('_')


def front_side_of(lab, classes, panels, log=print):
    """Return (side_sign, top_front_down): where the object's front is in the side and top views."""
    pan = {p['name']: p for p in panels}
    front = pan['front']
    fl = lab[:, front['c0']:front['c1']]
    on_face = []
    for c in classes:
        m = fl == c['id']
        if not m.any():
            continue
        ring = ndi.binary_dilation(m) & ~m
        others = fl[ring]
        if len(others) and (others == 0).mean() <= 0.25:  # other parts all round it
            on_face.append(c['id'])
    side_sign, top_down = 1, True
    for view in ('side', 'top'):
        p = pan.get(view)
        if p is None:
            continue
        votes = 0.0
        for cid in on_face:
            ys, xs = np.nonzero(lab[:, p['c0']:p['c1']] == cid)
            if not len(ys):
                continue
            if view == 'side':
                u = (xs.mean() + p['c0'] - p['cx']) / max(1.0, 0.5 * (p['c1'] - p['c0']))
            else:
                u = (ys.mean() - p['cy']) / max(1.0, 0.5 * (p['r1'] - p['r0']))
            if abs(u) > 0.6:
                votes += np.sign(u) * len(ys)
        if votes and view == 'side':
            side_sign = -1 if votes > 0 else 1
        elif votes:
            top_down = votes > 0
    log('[prop] the front faces %s in the side view and %s in the top view' % ('left' if side_sign > 0 else 'right', 'down' if top_down else 'up'))
    return int(side_sign), bool(top_down)


def main(argv):
    def arg(k, default=None):
        return argv[argv.index(k) + 1] if k in argv else default
    parts_png, turn_png, out = argv[0], argv[1], os.path.abspath(argv[2])
    prompt = arg('--prompt')
    name = arg('--name', 'prop')
    size = float(arg('--size', 1.0))
    views_want = arg('--views', 'front,side,top').split(',')
    os.makedirs(out, exist_ok=True)
    log = print
    img = np.asarray(Image.open(parts_png).convert('RGB')).astype(np.float32)
    art = np.asarray(Image.open(turn_png).convert('RGB')).astype(np.float32) if turn_png and os.path.exists(turn_png) else None
    H, W, _ = img.shape
    given = parse_prompt(prompt)
    names = [g[1] for g in given]
    sw, sw_mask = read_legend(img, len(names), log=log)
    legend_cx = float(np.median([q[1] for q in sw])) if sw else 0.8 * W
    fig_cols = slice(0, int(legend_cx - 0.04 * W))
    cols = []
    for num, pname, rgb in given:
        d = np.sqrt(((img[:, fig_cols] - np.array(rgb, np.float32)) ** 2).sum(2))
        m = d < 40
        cols.append([float(v) for v in np.median(img[:, fig_cols][m], axis=0)] if m.sum() >= 20 else [float(v) for v in rgb])
    figs, floor, body = find_figures(img, sw_mask, legend_cx, log=log, swatch_rgb=cols, floor_line=False)
    view_names = [v for v in views_want if v != 'top'][:len(figs)]
    panels = []
    for k, f in enumerate(figs):
        pn = view_names[k] if k < len(view_names) else 'top'
        panels.append(dict(name=pn, c0=int(f['c0']), c1=int(f['c1']), r0=int(f['r0']), r1=int(f['r1']),
                           cx=float(0.5 * (f['c0'] + f['c1'])), cy=float(0.5 * (f['r0'] + f['r1']))))
    front = panels[0]
    ppm = (front['r1'] - front['r0']) / size
    if art is not None:
        art = register_art(art, [f for f, p in zip(figs, panels) if p['name'] != 'top'], img.shape, log)
        art_png = os.path.join(out, name + '_art.png')
        Image.fromarray(np.clip(art, 0, 255).astype(np.uint8)).save(art_png)
    log('[prop] %d view(s): %s; front figure %d px tall -> %.1f px/m for %.3f m' % (len(panels), ', '.join(p['name'] for p in panels), front['r1'] - front['r0'], ppm, size))
    C = np.array(cols, np.float32)
    dist = np.sqrt(((img[:, :, None, :] - C[None, None, :, :]) ** 2).sum(3))
    near = dist.argmin(2)
    dmin = dist.min(2)
    # a tight 35 keeps anti-aliased edges of saturated parts from passing as pastel parts
    lab = np.where(dmin < 35, near + 1, 0).astype(np.int16)
    lab[sw_mask] = 0
    lab[:, int(legend_cx - 0.04 * W):] = 0
    for i in range(len(given)):
        mi = lab == i + 1
        if mi.any():
            keep = ndi.binary_dilation(ndi.binary_erosion(mi, iterations=1), iterations=1)
            lab[mi & ~keep] = 0
    fig = body & ~sw_mask
    fig[:, int(legend_cx - 0.04 * W):] = False
    hole = fig & (lab == 0)
    if hole.any():
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        filled = lab[idx[0], idx[1]]
        lab[hole] = filled[hole]
    lab[~fig] = 0
    art_colours = {}
    if art is not None and art.shape == img.shape:
        figa = (np.abs(art - 255).sum(2) > 40)
        figa[:, int(legend_cx - 0.04 * W):] = False
        q = (art // 32).astype(np.int32)
        key = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        all_p = np.bincount(key[figa], minlength=512).astype(np.float64)
        all_p /= max(1.0, all_p.sum())
        for i in range(len(given)):
            m = ndi.binary_erosion(lab == i + 1, iterations=3)
            if m.sum() < 30:
                continue
            km = key[m]
            if m.sum() < 200:
                px = art[m]
                px = px[(np.abs(px - 255).sum(1) > 40) & (px.mean(1) > 40)]
                if len(px):
                    art_colours[class_name(given[i][1])] = [int(v) for v in np.median(px, axis=0)]
                continue
            pm = np.bincount(km, minlength=512).astype(np.float64) / len(km)
            cand = [b for b in range(512) if pm[b] >= 0.10] or [int(np.argmax(pm))]
            best = max(cand, key=lambda b: pm[b] / (all_p[b] + 1e-4))
            art_colours[class_name(given[i][1])] = [int(v) for v in np.median(art[m][km == best], axis=0)]
    classes = [dict(id=i + 1, part=given[i][1], name=class_name(given[i][1]), colour=[int(round(c)) for c in cols[i]]) for i in range(len(given))]
    P = dict(name=name, size_m=size, ppm=ppm, panels=panels, classes=classes, art_colours=art_colours,
             parts_png=os.path.abspath(parts_png), turn_png=os.path.abspath(turn_png) if turn_png else None,
             labels=os.path.join(out, 'labels.npy'))
    np.save(P['labels'], lab)
    ov = np.full(img.shape, 255, np.uint8)
    for c in classes:
        ov[lab == c['id']] = c['colour']
    Image.fromarray(ov[:, :int(legend_cx - 0.04 * W)]).resize((int((legend_cx - 0.04 * W) / 2), H // 2)).save(os.path.join(out, 'reading.png'))
    log('[prop] classes: %s' % ', '.join('%s(%d px)' % (c['name'], int((lab == c['id']).sum())) for c in classes))
    log('[prop] art colours: %s' % json.dumps(art_colours))
    side_sign, top_front_down = front_side_of(lab, classes, panels, log)
    P.update(side_sign=side_sign, top_front_down=top_front_down)
    json.dump(P, open(os.path.join(out, 'prop.json'), 'w'), indent=1)
    views = [p for p in panels if p['name'] != 'top']
    json.dump(dict(ids={c['name']: c['id'] for c in classes}, panels={p['name']: [p['c0'], p['c1']] for p in views}),
              open(os.path.join(out, 'labels.json'), 'w'), indent=1)
    side = next((p for p in views if p['name'] == 'side'), None)
    char = dict(kind='prop', sheet=os.path.abspath(parts_png), art=art_png if art is not None else None,
                calibration=dict(ppm=ppm, floor_row=float(front['r1'] + 1),
                                 panels=[dict(name=p['name'], c0=p['c0'], c1=p['c1'], midline=p['cx']) for p in views]),
                datum=dict(side_col=side['cx'] if side else front['cx'], side_sign=float(side_sign)),
                roles={}, garments={}, art_colours=art_colours, classes=classes, size_m=size)
    json.dump(char, open(os.path.join(out, 'char.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
