"""Judge control: a pile of boxes fitted to the same sheet's silhouettes.

    py scripts/boxes.py out/punk out/punk_boxes [--band-mm 20] [--blender PATH]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run import find_blender  # noqa: E402

BLENDER_SIDE = r'''
import bpy, bmesh, json, sys
boxes, out = json.load(open(sys.argv[-2])), sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
for k, b in enumerate(boxes):
    (x0, x1), (y0, y1), (z0, z1) = b['x'], b['y'], b['z']
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = x0 if v.co.x < 0 else x1
        v.co.y = y0 if v.co.y < 0 else y1
        v.co.z = z0 if v.co.z < 0 else z1
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new('Box_%d' % k)
    bm.to_mesh(me)
    ob = bpy.data.objects.new('Box_%d' % k, me)
    ob['s3d_part'] = b['part']
    ob['s3d_class'] = b['cls']
    bpy.context.scene.collection.objects.link(ob)
bpy.ops.wm.save_as_mainfile(filepath=out)
print('[boxes] %d boxes -> %s' % (len(boxes), out))
'''


def runs(cols):
    idx = np.nonzero(cols)[0]
    if not len(idx):
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    return list(zip(starts, ends))


def fit_boxes(run_dir, band_mm):
    C = json.load(open(os.path.join(run_dir, 'char.json')))
    lab = np.load(os.path.join(run_dir, 'labels.npy'))
    lj = json.load(open(os.path.join(run_dir, 'labels.json')))
    names = {v: k for k, v in lj['ids'].items()}
    cal = C['calibration']
    ppm, floor = cal['ppm'], int(cal['floor_row'])
    pan = {p['name']: p for p in cal['panels']}
    fc0, fc1 = lj['panels']['front']
    sc0, sc1 = lj['panels']['side']
    mid = float(pan['front']['midline'])
    datum, sign = float(C['datum']['side_col']), float(C['datum'].get('side_sign', 1.0))
    hair = set(C['roles'].get('hair', []))
    band = max(2, int(round(band_mm / 1000.0 * ppm)))
    boxes = []
    for r0 in range(0, floor, band):
        r1 = min(floor, r0 + band)
        front = lab[r0:r1, fc0:fc1]
        side = lab[r0:r1, sc0:sc1] > 0
        if not (front > 0).any() or not side.any():
            continue
        sc = np.nonzero(side.any(0))[0]
        ys = sorted(sign * (sc0 + c - datum) / ppm for c in (sc[0], sc[-1] + 1))
        for a, b in runs((front > 0).any(0)):
            if b - a < 2:
                continue
            ids = front[:, a:b][front[:, a:b] > 0]
            cls = names.get(int(np.bincount(ids).argmax()), 'skin')
            part = 'hair' if cls in hair else ('body' if cls == 'skin' else 'garment')
            boxes.append(dict(x=[(fc0 + a - mid) / ppm, (fc0 + b - mid) / ppm], y=ys,
                              z=[(floor - r1) / ppm, (floor - r0) / ppm], cls=cls, part=part))
    return boxes


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='a finished character run (for the reader files)')
    ap.add_argument('out_dir')
    ap.add_argument('--band-mm', type=float, default=20.0)
    ap.add_argument('--blender')
    a = ap.parse_args(argv)
    blender = find_blender(a.blender)
    os.makedirs(a.out_dir, exist_ok=True)
    for f in ('char.json', 'labels.npy', 'labels.json', 'labels_ink.npy'):
        shutil.copyfile(os.path.join(a.run_dir, f), os.path.join(a.out_dir, f))
    boxes = fit_boxes(a.run_dir, a.band_mm)
    bj = os.path.join(a.out_dir, 'boxes.json')
    json.dump(boxes, open(bj, 'w'))
    script = os.path.join(a.out_dir, 'make_boxes.py')
    open(script, 'w').write(BLENDER_SIDE)
    subprocess.run([blender, '-b', '--factory-startup', '--python', script, '--', bj, os.path.join(a.out_dir, 'char.blend')],
                   check=True, capture_output=True)
    env = dict(os.environ, BLENDER=blender)
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'sketch3d', 'judge', 'judge.py'), a.out_dir],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print('%d boxes' % len(boxes))
    print(p.stdout.split('\n[judge]')[0])


if __name__ == '__main__':
    main(sys.argv[1:])
