"""Judge a finished build against its sheet and write judge/VERDICT.md.

    py sketch3d/judge/judge.py <outdir>
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get('BLENDER', 'blender')
PY = sys.executable

IOU_MIN = 0.80
BLOB_MAX_MM2 = 8000
TRI_BUDGET = 9000
TRI_MIN = 2000
DENTS_MAX = 0
NEEDLES_MAX = 0
SYM_MAX_MM = 4.0  # p95 mirror distance
SCARS_MAX = 0  # self-intersections repaired by carve
CONCAVE_MIN = 0.05  # share of edge length that folds inward
TIP_REACH_MM = 25.0
YAWS = '0,30,60,90,135,180,225,270,300,330'


def run(cmd, log_path, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    out = p.stdout + '\n' + p.stderr
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(out)
    return p.returncode, out


def blender(script, *args):
    return [BLENDER, '-b', '--factory-startup', '--python', os.path.join(HERE, script), '--'] + list(args)


def load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def scene_objects(blend, log_path):
    expr = ("import bpy,json;"
            "M=[o for o in bpy.data.objects if o.type=='MESH'];"
            "print('OBJ='+json.dumps({o.name:dict(part=o.get('s3d_part',''),cls=o.get('s3d_class',''),"
            "painted=o.get('s3d_painted',''),tris=sum(len(p.vertices)-2 for p in o.data.polygons)) for o in M}));"
            "print('HAIRV='+json.dumps([[round(c,4) for c in (o.matrix_world@v.co)] for o in M if o.get('s3d_part')=='hair' for v in o.data.vertices]))")
    _, out = run([BLENDER, '-b', blend, '--python-expr', expr], log_path)
    objs, hair_v = {}, []
    for line in out.splitlines():
        if line.startswith('OBJ='):
            objs = json.loads(line[4:])
        elif line.startswith('HAIRV='):
            hair_v = json.loads(line[6:])
    return objs, hair_v


def hair_tips(C, hair_v):
    """Match drawn hair spurs of 30 mm and longer to the nearest hair vertex."""
    ok, approx, missing = [], [], []
    ppm = C['calibration']['ppm']
    for panel in ('front', 'side'):
        for i, sp in enumerate(C['hair']['spurs'].get(panel, [])):
            if sp.get('length_m', 0) < 0.03:
                continue
            if sp.get('px') and sp['px'] / max(1.0, sp['length_m'] * ppm) / ppm < 0.008:
                continue  # under 8 mm wide: painted
            tu, tz = sp['tip']['u'], sp['tip']['z']
            best = min((((x if panel == 'front' else y) - tu) ** 2 + (z - tz) ** 2) ** 0.5 for x, y, z in hair_v) if hair_v else 1e9
            mm = best * 1000.0
            tag = '%s%d' % (panel[0], i)
            if mm <= TIP_REACH_MM:
                ok.append(tag)
            elif mm <= 2 * TIP_REACH_MM:
                approx.append('%s (%.0f mm off)' % (tag, mm))
            else:
                missing.append('%s (%.0f mm off)' % (tag, mm))
    return ok, approx, missing


def accounting(C, objs):
    acc = []
    painted = set(k for k, e in C['garments'].items() if e.get('painted'))
    painted |= set(q for q in (objs.get('Body', {}).get('painted') or '').split(',') if q)
    painted |= set((C.get('hair') or {}).get('painted', []))
    for cname, g in C['garments'].items():
        have = [n for n, v in objs.items() if v['cls'] == cname]
        status = 'built' if have else ('painted' if cname in painted else 'MISSING')
        acc.append(dict(feature='part: ' + cname, status=status, by=', '.join(have) if have else ('texture' if status == 'painted' else '')))
    hair_painted = list((C.get('hair') or {}).get('painted', []))
    plates = [n for n in objs if n.startswith('Hair_Plate')]
    if 'Hair' in objs:
        acc.append(dict(feature='hair cap', status='built', by='Hair'))
    elif hair_painted or plates:
        acc.append(dict(feature='hair cap', status='painted' if hair_painted else 'built', by=', '.join(hair_painted + plates)))
    elif (C.get('roles') or {}).get('hair'):
        acc.append(dict(feature='hair cap', status='MISSING', by=''))
    else:
        acc.append(dict(feature='hair cap', status='omitted: no hair drawn', by=''))
    kinds = sorted(set(f['kind'] for f in C['face']['features']))
    acc.append(dict(feature='face (%s)' % ', '.join(kinds), status='painted', by='tex/face_atlas.png'))
    return acc, painted


def prop_accounting(C, objs, outdir):
    painted = set(load_json(os.path.join(outdir, 'prop_painted.json'), []))
    acc = []
    for c in C.get('classes', []):
        have = [n for n, v in objs.items() if v['cls'] == c['name']]
        status = 'built' if have else ('painted' if c['name'] in painted else 'MISSING')
        acc.append(dict(feature='part: ' + c['name'], status=status, by=', '.join(have) if have else ('texture' if status == 'painted' else '')))
    return acc, painted


def symmetric_classes(outdir, C):
    """Return {class: drawn symmetric} from the front mask against its mirror."""
    lab = np.load(os.path.join(outdir, 'labels.npy'))
    lj = load_json(os.path.join(outdir, 'labels.json'))
    pan = {p['name']: p for p in C['calibration']['panels']}
    c0, c1 = lj['panels']['front']
    mid = float(pan['front']['midline'])
    half = int(min(mid - c0, c1 - mid))
    left = lab[:, int(mid) - half:int(mid)]
    right = lab[:, int(mid):int(mid) + half][:, ::-1]
    req = {name: metrics.iou(left == cid, right == cid) >= 0.88 for name, cid in lj['ids'].items()}
    req['__figure__'] = metrics.iou(left > 0, right > 0) >= 0.88
    return req


def repair_scars(outdir):
    scars = {}
    for lg in ('body', 'garments', 'shoes', 'hair', 'accessories'):
        path = os.path.join(outdir, 'build_%s.log' % lg)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding='utf-8', errors='replace'):
            m = re.search(r'\[carve\] (\S+): (\d+) self-intersecting face\(s\) treated', line)
            if m:
                scars[m.group(1)] = scars.get(m.group(1), 0) + int(m.group(2))
    return scars


def contact_sheet(paths, out, bg=(244, 243, 240), cols=None, pad=8):
    ims = [Image.open(p).convert('RGBA') for p in paths if os.path.exists(p)]
    if not ims:
        return
    cols = cols or len(ims)
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for k, im in enumerate(ims):
        tile = Image.new('RGB', im.size, bg)
        tile.paste(im, mask=im.split()[3])
        sheet.paste(tile, (pad + (k % cols) * (w + pad), pad + (k // cols) * (h + pad)))
    sheet.save(out)


def main(argv):
    outdir = os.path.abspath(argv[0])
    C = load_json(os.path.join(outdir, 'char.json'))
    blend = os.path.join(outdir, 'char.blend')
    tex_blend = os.path.join(outdir, 'tex', 'char_tex.blend')
    jd = os.path.join(outdir, 'judge')
    os.makedirs(jd, exist_ok=True)
    checks = []
    prop = C.get('kind') == 'prop'

    run(blender('audit.py', blend, os.path.join(jd, 'audit.json')), os.path.join(jd, 'audit.log'))
    A = load_json(os.path.join(jd, 'audit.json'), {'fails': ['audit did not run'], 'objects': {}})
    checks.append(('audit', not A['fails'], '%d objects, %d failing gates' % (len(A['objects']), len(A['fails'])), A['fails'][:12]))

    objs, hair_v = scene_objects(blend, os.path.join(jd, 'objects.log'))
    tris = sum(v['tris'] for v in objs.values())
    if prop:
        checks.append(('budget', None, '%d triangles in %d objects, n/a for props' % (tris, len(objs)), []))
    else:
        checks.append(('budget', metrics.within_budget(tris, TRI_MIN, TRI_BUDGET), '%d triangles in %d objects (budget %d)' % (tris, len(objs), TRI_BUDGET), []))

    acc, painted_parts = prop_accounting(C, objs, outdir) if prop else accounting(C, objs)
    if prop:
        with open(os.path.join(jd, 'painted.json'), 'w', encoding='utf-8') as f:
            json.dump(sorted(painted_parts), f)
    ok_t, approx_t, miss_t = hair_tips(C, hair_v) if not prop else ([], [], [])
    n_tips = len(ok_t) + len(approx_t) + len(miss_t)
    if n_tips:
        acc.append(dict(feature='hair tips reached (%d drawn >= 30 mm)' % n_tips, status='MISSING' if miss_t else 'built',
                        by='%d within %.0f mm%s%s' % (len(ok_t), TIP_REACH_MM, ('; approximate: ' + ', '.join(approx_t)) if approx_t else '',
                                                      ('; missing: ' + ', '.join(miss_t)) if miss_t else '')))
    missing = [a for a in acc if a['status'] == 'MISSING']
    checks.append(('accounting', not missing, '%d parts, %d missing' % (len(acc), len(missing)), [a['feature'] for a in missing]))

    cmp_dir = os.path.join(jd, 'cmp')
    run(blender('compare.py', os.path.join(outdir, 'char.json'), blend, cmp_dir), os.path.join(jd, 'compare.log'), timeout=1200)
    run([PY, os.path.join(HERE, 'analyse.py'), outdir], os.path.join(jd, 'analyse.log'), timeout=1200)
    R = load_json(os.path.join(cmp_dir, 'compare.json'), {})
    views = [v for v in ('front', 'side', 'back') if not prop or v in R.get('views', {})]
    ious = {v: R.get('views', {}).get(v, {}).get('figure_iou') for v in views}
    blobs = [d for d in R.get('defects', []) if d.get('kind') == 'mismatch']
    big = [b for b in blobs if b['area_mm2'] >= BLOB_MAX_MM2]
    checks.append(('sheet match', all((q or 0) >= IOU_MIN for q in ious.values()) and not big,
                   'silhouette IoU %s; %d mismatch blobs, %d mm2 in total; %d over %d mm2' % (
                       ', '.join('%s %s' % (v, ious[v]) for v in views), len(blobs), R.get('mismatch_mm2', 0), len(big), BLOB_MAX_MM2),
                   [b['msg'] for b in blobs[:8]]))

    run(blender('quality.py', blend, os.path.join(jd, 'quality.json')), os.path.join(jd, 'quality.log'), timeout=1200)
    Q = load_json(os.path.join(jd, 'quality.json'), {})
    sym_req = symmetric_classes(outdir, C) if not prop else {}
    scars = repair_scars(outdir)
    lines, dents, needles, creases, worst_sym = [], 0, 0, 0, 0.0
    for name, q in sorted(Q.items()):
        dents += q.get('n_dents', 0)
        needles += q.get('needles', 0)
        creases += q.get('n_creases', 0)
        sym = q.get('sym_p95_mm')
        req = sym_req.get('__figure__', True) if q.get('part') == 'body' else sym_req.get(q.get('cls'), False)
        if sym is not None and req and q.get('part') in ('body', 'garment'):
            worst_sym = max(worst_sym, sym)
        else:
            sym = None
        if q.get('n_dents') or q.get('needles') or q.get('n_creases') or scars.get(name) or (sym or 0) > SYM_MAX_MM:
            lines.append('%s: dents %d (worst %s mm), needles %d, creases %d%s' % (
                name, q.get('n_dents', 0), q['dents'][0]['mm'] if q.get('dents') else '-', q.get('needles', 0),
                q.get('n_creases', 0), (', symmetry p95 %.1f mm' % sym) if sym is not None else ''))
    n_scars = sum(scars.values())
    if prop:
        checks.append(('surface', dents <= DENTS_MAX and needles <= NEEDLES_MAX and n_scars <= SCARS_MAX,
                       '%d dents, %d needles, %d repaired self-intersections; creases and symmetry n/a for props' % (
                           dents, needles, n_scars), lines[:14]))
    else:
        checks.append(('surface', dents <= DENTS_MAX and needles <= NEEDLES_MAX and worst_sym <= SYM_MAX_MM and n_scars <= SCARS_MAX,
                       '%d dents, %d needles, %d creases, worst symmetry p95 %.1f mm, %d repaired self-intersections' % (
                           dents, needles, creases, worst_sym, n_scars), lines[:14]))

    edge_len = sum(q.get('edge_len_m', 0.0) for q in Q.values())
    concave = sum(q.get('concave_len_m', 0.0) for q in Q.values()) / max(1e-9, edge_len)
    if prop:
        checks.append(('form', None, 'concave edges %.3f of the edge length, n/a for props' % concave, []))
    else:
        checks.append(('form', concave >= CONCAVE_MIN, 'concave edges %.3f of the edge length (minimum %.2f)' % (concave, CONCAVE_MIN), []))

    src = tex_blend if os.path.exists(tex_blend) else blend
    rd = os.path.join(jd, 'renders')
    run(blender('render.py', src, os.path.join(rd, 'tt'), '--modes', 'clay,wire,flat', '--yaws', YAWS), os.path.join(jd, 'render.log'), timeout=2400)
    if not prop:
        run(blender('render.py', src, os.path.join(rd, 'head'), '--modes', 'clay,flat', '--yaws', '0,45,90,180', '--res', '560x560', '--head'),
            os.path.join(jd, 'render_head.log'), timeout=1200)
    for mode in ('clay', 'wire', 'flat'):
        contact_sheet([os.path.join(rd, 'tt_%s_%03d.png' % (mode, int(y))) for y in YAWS.split(',')], os.path.join(jd, 'turntable_%s.png' % mode), cols=5)
    for mode in ('clay', 'flat') if not prop else ():
        contact_sheet([os.path.join(rd, 'head_%s_%03d.png' % (mode, y)) for y in (0, 45, 90, 180)], os.path.join(jd, 'head_%s.png' % mode))

    ok = all(c[1] for c in checks if c[1] is not None)
    md = ['# Verdict: %s' % ('PASS' if ok else 'FAIL'), '', '%s: %s' % ('prop' if prop else 'character', os.path.basename(outdir)),
          'time: %s' % time.strftime('%Y-%m-%d %H:%M'), '', '| check | result | detail |', '|---|---|---|']
    for name, good, detail, extra in checks:
        md.append('| %s | %s | %s |' % (name, 'n/a' if good is None else ('PASS' if good else 'FAIL'), detail))
        md += ['|  |  | %s |' % e for e in extra]
    md += ['', '## Part accounting', '', '| part | status | by |', '|---|---|---|']
    md += ['| %s | %s | %s |' % (a['feature'], a['status'], a.get('by', '')) for a in acc]
    md += ['', '## Objects', '', '| object | part | class | tris |', '|---|---|---|---|']
    md += ['| %s | %s | %s | %d |' % (n, v['part'], v['cls'], v['tris']) for n, v in sorted(objs.items())]
    md += ['', 'Renders in the run folder: judge/turntable_{clay,wire,flat}.png, %sjudge/cmp/compare_{%s}.png' % (
        '' if prop else 'judge/head_{clay,flat}.png, ', ','.join(views)), '']
    with open(os.path.join(jd, 'VERDICT.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    summary = dict(ok=ok, tris=tris, objects=len(objs), iou=ious, mismatch_mm2=R.get('mismatch_mm2', 0), blobs=len(blobs),
                   blobs_over_limit=len(big), checks={c[0]: c[1] for c in checks}, dents=dents, needles=needles,
                   creases=creases, worst_sym_mm=worst_sym, concave=round(concave, 4), audit_fails=len(A['fails']))
    with open(os.path.join(jd, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print('\n'.join(md[:8 + len(checks)]))
    print('[judge] %s' % os.path.join(jd, 'VERDICT.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
