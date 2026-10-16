"""Character turnaround -> textured low-poly mesh in Blender, then an independent judge.

    py run.py examples/punk
    py run.py examples/chest --prop --size 0.6
    py run.py --turn T.png --parts P.png --legend prompt_parts.txt --name punk [--height 1.6]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, 'sketch3d')
BLENDER_CANDIDATES = [
    r'C:/Program Files/Blender Foundation/Blender 5.1/blender.exe',
    '/Applications/Blender.app/Contents/MacOS/Blender',
    '/usr/bin/blender',
]
STAGES = ('read', 'build', 'finish', 'judge')


def find_blender(explicit):
    for cand in (explicit, os.environ.get('BLENDER')):
        if cand:
            return cand
    for cand in BLENDER_CANDIDATES:
        if os.path.exists(cand):
            return cand
    found = shutil.which('blender')
    if found:
        return found
    sys.exit('Blender not found: pass --blender or set BLENDER')


class Runner:
    def __init__(self, out, blender):
        self.out = out
        self.blender = blender
        self.env = dict(os.environ, BLENDER=blender, PYTHONIOENCODING='utf-8')

    def py(self, script, *args):
        return [sys.executable, os.path.join(PKG, script)] + list(args)

    def bl(self, script, *args):
        return [self.blender, '-b', '--factory-startup', '--python', os.path.join(PKG, script), '--'] + list(args)

    def step(self, name, cmd, timeout=3600):
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                           cwd=HERE, env=self.env, timeout=timeout)
        log = os.path.join(self.out, 'logs', name + '.log')
        with open(log, 'w', encoding='utf-8') as f:
            f.write(p.stdout + '\n' + p.stderr)
        ok = p.returncode == 0 and 'Traceback' not in p.stdout + p.stderr
        print('%-10s %s %6.1fs' % (name, 'ok  ' if ok else 'FAIL', time.time() - t0))
        if not ok:
            tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-8:]
            for line in tail:
                print('    ' + line[:200])
            print('stopped at %s, see %s' % (name, log))
            sys.exit(1)
        return p.stdout


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('example', nargs='?', help='folder with turn.png, parts.png, prompt_parts.txt')
    ap.add_argument('--turn', help='colour art turnaround (front, side, back[, top])')
    ap.add_argument('--parts', help='part map: the same views, one flat colour per part')
    ap.add_argument('--legend', help='text file with lines "N part name = #RRGGBB"')
    ap.add_argument('--name', help='character name (default: example folder name)')
    ap.add_argument('--height', type=float, default=1.6, help='stature in metres (default 1.6)')
    ap.add_argument('--prop', action='store_true', help='a rigid object instead of a character')
    ap.add_argument('--size', type=float, default=1.0, help='front height of a prop in metres (default 1.0)')
    ap.add_argument('--out', help='output folder (default out/<name>)')
    ap.add_argument('--only', help='comma-separated subset of: ' + ','.join(STAGES))
    ap.add_argument('--blender', help='path to the Blender executable (or set BLENDER)')
    a = ap.parse_args(argv)
    if a.example:
        a.turn = a.turn or os.path.join(a.example, 'turn.png')
        a.parts = a.parts or os.path.join(a.example, 'parts.png')
        a.legend = a.legend or os.path.join(a.example, 'prompt_parts.txt')
        a.name = a.name or os.path.basename(os.path.normpath(a.example))
    if not (a.turn and a.parts and a.legend):
        ap.error('give an example folder or --turn, --parts and --legend')
    a.name = a.name or 'char'
    a.out = os.path.abspath(a.out or os.path.join(HERE, 'out', a.name))
    a.only = a.only.split(',') if a.only else list(STAGES)
    return a


def main(argv):
    a = parse(argv)
    os.makedirs(os.path.join(a.out, 'logs'), exist_ok=True)
    r = Runner(a.out, find_blender(a.blender))
    t0 = time.time()
    tex = os.path.join(a.out, 'tex')
    char = os.path.join(a.out, 'char.json')
    if a.prop:
        if 'read' in a.only:
            r.step('read', r.py('read/prop.py', os.path.abspath(a.parts), os.path.abspath(a.turn), a.out,
                                '--prompt', os.path.abspath(a.legend), '--name', a.name, '--size', str(a.size)))
        if 'build' in a.only:
            r.step('build', r.bl('build/prop_build.py', os.path.join(a.out, 'prop.json'), os.path.join(a.out, 'char.blend')))
        if 'finish' in a.only:
            r.step('export', r.bl('finish/export.py', os.path.join(a.out, 'char.blend'), os.path.join(a.out, 'export', a.name)))
        if 'judge' in a.only:
            print(r.step('judge', r.py('judge/judge.py', a.out), timeout=5400).strip().split('\n[judge]')[0])
        print('done in %.0fs -> %s' % (time.time() - t0, a.out))
        return
    if 'read' in a.only:
        r.step('protocol', r.py('read/protocol.py', os.path.abspath(a.parts), os.path.abspath(a.turn), a.out,
                                '--name', a.name, '--height', str(a.height), '--prompt', os.path.abspath(a.legend)))
        r.step('read', r.py('read/read.py', os.path.abspath(a.parts), os.path.join(a.out, a.name + '_sheet.json'), a.out,
                            '--brief', os.path.join(a.out, 'brief.json')))
    if 'build' in a.only:
        r.step('build', r.py('build/assemble.py', a.out))
    if 'finish' in a.only:
        r.step('uv', r.bl('finish/uv.py', char, os.path.join(a.out, 'body.json'), os.path.join(a.out, 'char.blend'), tex))
        r.step('face', r.py('finish/facepaint.py', char, os.path.join(tex, 'face_cell.json'), os.path.join(tex, 'face_atlas.png'),
                            '--preview', os.path.join(tex, 'face_preview.png')))
        r.step('rig', r.bl('finish/rig.py', char, os.path.join(a.out, 'body.json'), os.path.join(tex, 'char_tex.blend'),
                           os.path.join(a.out, 'rig')))
        r.step('export', r.bl('finish/export.py', os.path.join(tex, 'char_tex.blend'), os.path.join(a.out, 'export', a.name)))
    if 'judge' in a.only:
        print(r.step('judge', r.py('judge/judge.py', a.out), timeout=5400).strip().split('\n[judge]')[0])
    print('done in %.0fs -> %s' % (time.time() - t0, a.out))


if __name__ == '__main__':
    main(sys.argv[1:])
