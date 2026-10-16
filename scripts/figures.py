"""Build the README figures from a finished run.

    py scripts/figures.py out/punk examples/punk docs [--boxes out/punk_boxes] [--gallery punk,robot] [--blender PATH]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run import find_blender  # noqa: E402

BG = (240, 239, 236)
INK = (60, 60, 64)
DRAWN = (52, 120, 230)  # drawn, not built
BUILT = (240, 120, 40)  # built, not drawn
BOTH = (200, 200, 196)


def font(size):
    return ImageFont.load_default(size=size)


def text(d, xy, s, size):
    """Draw s in INK, setting '²' as a raised 2 (the default font lacks it)."""
    x, y = xy
    for k, part in enumerate(s.split('²')):
        if k:
            sup = font(int(size * 0.6))
            d.text((x + 1, y - size * 0.08), '2', fill=INK, font=sup)
            x += d.textlength('2', font=sup) + 2
        d.text((x, y), part, fill=INK, font=font(size))
        x += d.textlength(part, font=font(size))


def render(blender, blend, prefix, modes, yaws, res, extra=()):
    cmd = [blender, '-b', '--factory-startup', '--python', os.path.join(ROOT, 'sketch3d', 'judge', 'render.py'), '--',
           blend, prefix, '--modes', ','.join(modes), '--yaws', ','.join(str(y) for y in yaws), '--res', res] + list(extra)
    subprocess.run(cmd, check=True, capture_output=True)


def on_bg(im, bg=BG):
    tile = Image.new('RGB', im.size, bg)
    tile.paste(im, mask=im.split()[3])
    return tile


def with_shadow(im, strength=70):
    a = np.asarray(im.split()[3]) > 0
    ys, xs = np.nonzero(a)
    if not len(ys):
        return im
    foot = xs[ys > ys.max() - 0.04 * (ys.max() - ys.min())]
    cx, half = 0.5 * (foot.min() + foot.max()), 0.62 * (foot.max() - foot.min())
    h = max(6, int(0.018 * (ys.max() - ys.min())))
    pad = 3 * h
    out = Image.new('RGBA', (im.width, im.height + pad), (0, 0, 0, 0))
    sh = Image.new('L', out.size, 0)
    ImageDraw.Draw(sh).ellipse([cx - half, ys.max() - h, cx + half, ys.max() + h], fill=strength)
    sh = sh.filter(ImageFilter.GaussianBlur(h))
    out.paste(Image.new('RGBA', out.size, (40, 38, 34, 255)), mask=sh)
    out.alpha_composite(im, (0, 0))
    return out


def trim(im, pad=0):
    a = np.asarray(im.split()[3]) > 0
    ys, xs = np.nonzero(a)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad + 1), min(im.height, ys.max() + pad + 1)))


def first_figure(path, gap=80):
    """Columns of the first inked figure, joining gaps under `gap` px."""
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(int)
    ink = a.min(2) < 215
    ink[ink.mean(1) > 0.4] = False  # a ground line
    cols = np.nonzero(ink.sum(0) > 3)[0]
    breaks = np.nonzero(np.diff(cols) > gap)[0]
    c0, c1 = cols[0], (cols[breaks[0]] if len(breaks) else cols[-1]) + 1
    rows = np.nonzero(ink[:, c0:c1].sum(1) > 1)[0]
    crop = im.crop((int(c0), int(rows[0]), int(c1), int(rows[-1]) + 1)).convert('RGBA')
    px = np.asarray(crop).copy()
    px[px[:, :, :3].min(2) > 235, 3] = 0
    return Image.fromarray(px)


def fit(im, h):
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def hero(run_dir, example, docs, blender, tmp):
    blend = os.path.join(run_dir, 'tex', 'char_tex.blend')
    render(blender, blend, os.path.join(tmp, 'hero'), ['clay', 'wire', 'lit'], [0, 330], '900x1200', ['--pitch', '6'])
    H = 760
    tiles = [(fit(trim(first_figure(os.path.join(example, 'turn.png'))), H), 'input art (front view)')]
    tiles += [(fit(trim(with_shadow(Image.open(os.path.join(tmp, 'hero_lit_000.png')))), H), 'textured, front'),
              (fit(trim(with_shadow(Image.open(os.path.join(tmp, 'hero_lit_330.png')))), H), 'textured, 3/4'),
              (fit(trim(Image.open(os.path.join(tmp, 'hero_clay_330.png'))), H), 'clay'),
              (fit(trim(Image.open(os.path.join(tmp, 'hero_wire_330.png'))), H), 'wireframe')]
    gap, top, bottom = 56, 40, 70
    W = sum(t.width for t, _ in tiles) + gap * (len(tiles) + 1)
    out = Image.new('RGB', (W, H + top + bottom), BG)
    d = ImageDraw.Draw(out)
    x = gap
    for k, (t, label) in enumerate(tiles):
        out.paste(t, (x, top), t)
        tw = d.textlength(label, font=font(26))
        d.text((x + (t.width - tw) / 2, top + H + 22), label, fill=INK, font=font(26))
        if k == 0:
            sep = x + t.width + gap // 2
            d.line([(sep, top + 30), (sep, top + H - 30)], fill=(200, 199, 195), width=2)
        x += t.width + gap
    out.save(os.path.join(docs, 'hero.png'), optimize=True)


def inputs(example, docs):
    ims = [Image.open(os.path.join(example, n)).convert('RGB') for n in ('turn.png', 'parts.png')]
    w = 1100
    ims = [im.resize((w, round(im.height * w / im.width)), Image.LANCZOS) for im in ims]
    out = Image.new('RGB', (w * 2 + 24, ims[0].height), (255, 255, 255))
    out.paste(ims[0], (0, 0))
    out.paste(ims[1], (w + 24, 0))
    out.save(os.path.join(docs, 'input.png'), optimize=True)


def model_of(run_dir):
    C = json.load(open(os.path.join(run_dir, 'char.json')))
    if C.get('kind') == 'prop':
        return os.path.join(run_dir, 'char.blend'), '15', True
    return os.path.join(run_dir, 'tex', 'char_tex.blend'), '6', False


def turntable(run_dir, docs, blender, tmp, step=10, name='turntable.gif', res='420x560'):
    blend, pitch, prop = model_of(run_dir)
    if prop:
        res = 'x'.join(reversed(res.split('x')))
    yaws = list(range(0, 360, step))
    prefix = os.path.join(tmp, os.path.splitext(name)[0])
    render(blender, blend, prefix, ['lit'], yaws, res, ['--pitch', pitch, '--samples', '24'])
    frames = [on_bg(with_shadow(Image.open('%s_lit_%03d.png' % (prefix, y)))) for y in yaws]
    strip = Image.new('RGB', (frames[0].width * len(frames[::6]), frames[0].height))
    for k, f in enumerate(frames[::6]):
        strip.paste(f, (k * f.width, 0))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(os.path.join(docs, name), save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def gallery(names, out_root, docs, blender, H=520, CW=440):
    """One row per example: the art's front figure, the model front and 3/4, clay."""
    def cell(im):
        k = min(CW / float(im.width), H / float(im.height))
        return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    rows = []
    for nm in names:
        run_dir = os.path.join(out_root, nm)
        tmp = os.path.join(run_dir, 'figures')
        os.makedirs(tmp, exist_ok=True)
        blend, pitch, prop = model_of(run_dir)
        render(blender, blend, os.path.join(tmp, 'gal'), ['clay', 'lit'], [0, 330], '950x700' if prop else '700x950', ['--pitch', pitch])
        art = os.path.join(ROOT, 'examples', nm, 'turn.png')
        tiles = [cell(trim(first_figure(art)))]
        tiles += [cell(trim(with_shadow(Image.open(os.path.join(tmp, 'gal_lit_%03d.png' % y))))) for y in (0, 330)]
        tiles += [cell(trim(Image.open(os.path.join(tmp, 'gal_clay_330.png'))))]
        rows.append(tiles)
    gap = 44
    W = 4 * CW + 5 * gap
    out = Image.new('RGB', (W, len(rows) * (H + gap) + gap + 40), BG)
    d = ImageDraw.Draw(out)
    for k, label in enumerate(('input art', 'textured, front', 'textured, 3/4', 'clay')):
        tw = d.textlength(label, font=font(24))
        d.text((gap + k * (CW + gap) + (CW - tw) / 2, 12), label, fill=INK, font=font(24))
    y = 40 + gap
    for r in rows:
        for k, t in enumerate(r):
            x = gap + k * (CW + gap) + (CW - t.width) // 2
            out.paste(t, (x, y + (H - t.height) // 2), t)
        y += H + gap
    out.save(os.path.join(docs, 'gallery.png'), optimize=True)


def judge(run_dir, docs):
    cmp_dir = os.path.join(run_dir, 'judge', 'cmp')
    R = json.load(open(os.path.join(cmp_dir, 'compare.json')))
    tiles = []
    for view in ('front', 'side', 'back'):
        a = np.asarray(Image.open(os.path.join(cmp_dir, 'compare_%s.png' % view)).convert('RGB')).astype(int)
        out = np.full(a.shape, BG, np.uint8)
        for src, dst in (((60, 90, 255), DRAWN), ((255, 60, 60), BUILT), ((150, 80, 200), BOTH)):
            m = (np.abs(a - np.array(src)).sum(2) < 60)
            out[m] = dst
        im = Image.fromarray(out)
        ys, xs = np.nonzero((out != np.array(BG)).any(2))
        im = im.crop((max(0, xs.min() - 12), max(0, ys.min() - 12), min(im.width, xs.max() + 13), min(im.height, ys.max() + 13)))
        n_blobs = sum(1 for b in R['views'][view]['blobs'] if b['kind'] == 'mismatch')
        tiles.append((fit(im, 620), '%s   IoU %.3f   %d blobs' % (view, R['views'][view]['figure_iou'], n_blobs)))
    gap, top, bottom = 50, 30, 120
    W = sum(t.width for t, _ in tiles) + gap * (len(tiles) + 1)
    W = max(W, 1100)
    out = Image.new('RGB', (W, 620 + top + bottom), BG)
    d = ImageDraw.Draw(out)
    x = (W - (sum(t.width for t, _ in tiles) + gap * (len(tiles) - 1))) // 2
    for t, label in tiles:
        out.paste(t, (x, top))
        tw = d.textlength(label, font=font(22))
        d.text((x + (t.width - tw) / 2, top + 620 + 14), label, fill=INK, font=font(22))
        x += t.width + gap
    ly = top + 620 + 62
    lx = (W - 760) // 2
    for col, text in ((BOTH, 'drawn and built'), (DRAWN, 'drawn, not built'), (BUILT, 'built, not drawn')):
        d.rectangle([lx, ly, lx + 22, ly + 22], fill=col)
        d.text((lx + 32, ly - 2), text, fill=INK, font=font(22))
        lx += 260
    out.save(os.path.join(docs, 'judge.png'), optimize=True)


def boxes(run_dir, boxes_dir, docs, blender, tmp):
    rows = []
    for d, tag, title in ((boxes_dir, 'box', 'pile of boxes fitted to the silhouettes'), (run_dir, 'model', 'built model')):
        render(blender, os.path.join(d, 'char.blend'), os.path.join(tmp, tag), ['clay'], [0, 330], '700x950')
        S = json.load(open(os.path.join(d, 'judge', 'summary.json')))
        iou = S['iou']
        lines = ['silhouette IoU  %.3f / %.3f / %.3f' % (iou['front'], iou['side'], iou['back']),
                 'part mismatch  %s mm², %d blob%s over the limit' % (format(S['mismatch_mm2'], ',').replace(',', ' '), S['blobs_over_limit'],
                                                                 '' if S['blobs_over_limit'] == 1 else 's'),
                 'concave edges  %.3f' % S['concave']]
        rows.append(([fit(trim(Image.open(os.path.join(tmp, '%s_clay_%03d.png' % (tag, y)))), 560) for y in (0, 330)], title, lines))
    gap, H = 40, 560
    col_w = [sum(t.width for t in ims) + gap for ims, _, _ in rows]
    W = sum(col_w) + gap * 3
    out = Image.new('RGB', (W, H + 230), BG)
    d = ImageDraw.Draw(out)
    x = gap
    for (ims, title, lines), cw in zip(rows, col_w):
        xx = x
        for t in ims:
            out.paste(t, (xx, 70), t)
            xx += t.width + gap
        d.text((x, 20), title, fill=INK, font=font(30))
        for k, line in enumerate(lines):
            text(d, (x, 70 + H + 24 + k * 38), line, 25)
        x += cw + gap
    out.save(os.path.join(docs, 'boxes.png'), optimize=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('example')
    ap.add_argument('docs')
    ap.add_argument('--boxes', help='run folder of scripts/boxes.py, for docs/boxes.png')
    ap.add_argument('--gallery', help='comma-separated example names (out/<name>, examples/<name>) for docs/gallery.png')
    ap.add_argument('--blender')
    a = ap.parse_args(argv)
    blender = find_blender(a.blender)
    os.makedirs(a.docs, exist_ok=True)
    tmp = os.path.join(a.run_dir, 'figures')
    os.makedirs(tmp, exist_ok=True)
    hero(a.run_dir, a.example, a.docs, blender, tmp)
    inputs(a.example, a.docs)
    turntable(a.run_dir, a.docs, blender, tmp)
    judge(a.run_dir, a.docs)
    if a.gallery:
        names = a.gallery.split(',')
        gallery(names, os.path.dirname(os.path.abspath(a.run_dir)), a.docs, blender)
        for nm in names[1:]:
            turntable(os.path.join(os.path.dirname(os.path.abspath(a.run_dir)), nm), a.docs, blender, tmp,
                      name='turntable_%s.gif' % nm, res='300x400')
    if a.boxes:
        boxes(a.run_dir, a.boxes, a.docs, blender, tmp)
    for n in sorted(os.listdir(a.docs)):
        print('%-16s %7.0f kB' % (n, os.path.getsize(os.path.join(a.docs, n)) / 1024.0))


if __name__ == '__main__':
    main(sys.argv[1:])
