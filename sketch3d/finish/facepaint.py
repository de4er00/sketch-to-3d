"""Paint the 3x3 face atlas of expressions from the traced face.

    python sketch3d/finish/facepaint.py <char.json> <face_cell.json> <face_atlas.png> [--preview preview.png]
"""
import sys
import json
import math
from PIL import Image, ImageDraw

SS = 4


def transform(pts, scale=(1.0, 1.0), about=None, rot_deg=0.0, shift=(0.0, 0.0)):
    if not pts:
        return pts
    ax = about if about is not None else (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    out = []
    for x, z in pts:
        dx, dz = (x - ax[0]) * scale[0], (z - ax[1]) * scale[1]
        out.append((ax[0] + dx * c - dz * s + shift[0], ax[1] + dx * s + dz * c + shift[1]))
    return out


def bbox(pts):
    xs = [p[0] for p in pts]; zs = [p[1] for p in pts]
    return min(xs), max(xs), min(zs), max(zs)


def expression(feats, name, iris):
    F = [dict(f) for f in feats]
    eyes = [f for f in F if f['kind'] in ('eye', 'sclera', 'iris', 'highlight')]
    side = lambda f: 1 if f['centre'][0] >= 0 else -1
    out = []

    def eye_group(sgn):
        return [f for f in eyes if side(f) == sgn]

    def eye_box(sgn):
        pts = sum([f['poly'] for f in eye_group(sgn)], [])
        return bbox(pts) if pts else None
    for f in F:
        k = f['kind']
        p = f['poly']
        if k in ('eye', 'sclera', 'iris', 'highlight'):
            eb = eye_box(side(f))
            ec = (0.5 * (eb[0] + eb[1]), 0.5 * (eb[2] + eb[3])) if eb else None
            if name == 'happy':
                p = transform(p, (1.0, 0.45), about=(ec[0], eb[3]))
            elif name == 'angry':
                p = transform(p, (1.0, 0.75), about=ec)
            elif name == 'surprised':
                p = transform(p, (1.2, 1.25), about=ec)
            elif name == 'blink' or (name == 'wink' and side(f) > 0):
                if k != 'eye':
                    continue
                p = transform(p, (1.0, 0.10), about=(ec[0], eb[2] + 0.35 * (eb[3] - eb[2])))
            elif name == 'shout':
                p = transform(p, (1.0, 0.9), about=ec)
        elif k == 'brow':
            sgn = side(f)
            c = f['centre']
            if name == 'angry':
                p = transform(p, rot_deg=-14 * sgn, about=(c[0], c[1]), shift=(0.0, -0.004))
            elif name == 'sad':
                p = transform(p, rot_deg=12 * sgn, about=(c[0], c[1]), shift=(0.0, 0.002))
            elif name == 'surprised':
                p = transform(p, shift=(0.0, 0.008))
            elif name == 'shout':
                p = transform(p, rot_deg=-8 * sgn, about=(c[0], c[1]), shift=(0.0, -0.003))
            elif name == 'wink' and sgn > 0:
                p = transform(p, shift=(0.0, -0.003))
        elif k == 'mouth':
            mb = bbox(p)
            mc = (0.5 * (mb[0] + mb[1]), 0.5 * (mb[2] + mb[3]))
            w = mb[1] - mb[0]
            if name in ('happy', 'wink'):
                p = arc_mouth(mc, w * 1.25, 0.010, up=True)
            elif name == 'sad':
                p = arc_mouth(mc, w * 1.1, 0.008, up=False)
            elif name == 'surprised':
                p = ellipse(mc, w * 0.22, w * 0.30)
                f = dict(f, fill=(40, 20, 24))
            elif name == 'grin':
                p = arc_mouth(mc, w * 1.5, 0.014, up=True, thick=True)
            elif name == 'shout':
                p = ellipse((mc[0], mc[1] - 0.006), w * 0.36, w * 0.42)
                f = dict(f, fill=(40, 20, 24))
            elif name == 'angry':
                p = arc_mouth(mc, w * 1.1, 0.004, up=False)
        f['poly'] = p
        out.append(f)
    return out


def area(poly):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])))


def inside(pt, poly):
    x, y = pt
    hit = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def stack(F, order):
    """Drawing order by kind, a feature inside a larger one after it."""
    def centre(f):
        xs, zs = zip(*f['poly'])
        return 0.5 * (min(xs) + max(xs)), 0.5 * (min(zs) + max(zs))
    depth = {}
    for f in F:
        c = centre(f)
        depth[id(f)] = sum(1 for g in F if g is not f and len(g['poly']) >= 3 and area(g['poly']) > area(f['poly']) and inside(c, g['poly']))
    return [(f, depth[id(f)] > 0) for f in sorted(F, key=lambda f: (depth[id(f)], order.get(f['kind'], 9)))]


def arc_mouth(c, w, h, up=True, thick=False, n=14):
    """A mouth as a curved band, corners lifted when `up`."""
    top, bot = [], []
    t_ = 0.0022 if not thick else 0.0035
    for i in range(n + 1):
        t = i / n
        x = c[0] - 0.5 * w + w * t
        y = c[1] + (h if up else -h) * (4 * (t - 0.5) ** 2 - 1) * (1 if up else 1) * (-1 if up else 1) * -1
        top.append((x, y + 0.5 * t_))
        bot.append((x, y - 0.5 * t_))
    return top + bot[::-1]


def ellipse(c, rx, ry, n=24):
    return [(c[0] + rx * math.cos(2 * math.pi * i / n), c[1] + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def main(argv):
    char_path, cell_path, out_png = argv[:3]
    preview = argv[argv.index('--preview') + 1] if '--preview' in argv else None
    C = json.load(open(char_path))
    cell = json.load(open(cell_path))
    face = C['face']
    RES = cell['res']
    csz = cell['cell']
    scale = cell['scale_uv_per_m']
    cx, cz = cell['centre_m']
    u_c, v_c = cell['u_c'], cell['v_c']
    skin = tuple(face['skin_rgb'])
    atlas = Image.new('RGB', (RES, RES), skin)
    names = ['neutral', 'happy', 'angry', 'sad', 'surprised', 'blink', 'wink', 'grin', 'shout']
    feats = face['features']
    order = {'sclera': 0, 'iris': 1, 'eye': 2, 'highlight': 3, 'brow': 4, 'nose': 5, 'mouth': 6, 'blush': -1}
    cpx = int(round(csz * RES))
    for idx, name in enumerate(names):
        i, j = idx % 3, idx // 3
        F = expression(feats, name, face.get('iris'))
        img = Image.new('RGB', (cpx * SS, cpx * SS), skin)
        dr = ImageDraw.Draw(img)

        def to_px(x, z):
            u = (u_c + (x - cx) * scale) / csz  # 0..1 inside the cell
            v = (v_c + (z - cz) * scale) / csz
            return (u * cpx * SS, (1.0 - v) * cpx * SS)
        for f, on_feature in stack([f for f in F if len(f['poly']) >= 3], order):
            p = f['poly']
            col = tuple(f.get('fill', f['colour']))
            if f['kind'] == 'blush' and not on_feature:
                col = tuple(int(0.5 * a + 0.5 * b) for a, b in zip(col, skin))
            dr.polygon([to_px(x, z) for x, z in p], fill=col)
        if not any(f['kind'] == 'nose' for f in F):
            ez = face.get('eye_row_z', cz)
            mz = [f for f in F if f['kind'] == 'mouth']
            mz = bbox(mz[0]['poly'])[3] if mz else cz - 0.02
            nz = mz + 0.5 * (ez - mz)
            a, b = to_px(0.004, nz + 0.004), to_px(-0.002, nz - 0.002)
            dr.line([a, b], fill=(60, 40, 40), width=max(1, int(1.2 * SS)))
        small = img.resize((cpx, cpx), Image.LANCZOS)
        # cell (i, j) sits at u = i/3, v = -j/3, and v = 1 is the image top
        atlas.paste(small, (i * cpx, RES - (j + 1) * cpx))
    atlas.save(out_png)
    if preview:
        atlas.resize((512, 512), Image.LANCZOS).save(preview)
    print('[facepaint] %s: %d features, cells %s' % (out_png, len(feats), ', '.join(names)))


if __name__ == '__main__':
    main(sys.argv[1:])
