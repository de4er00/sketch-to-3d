"""Run the build stages in order on one character folder.

    python sketch3d/build/assemble.py <outdir> [--only body,garments,shoes,hair,accessories,polish]
"""
import os
import sys
import subprocess
import time
import shutil

BLENDER = os.environ.get("BLENDER", "blender")
D = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ('body', 'body.py'),
    ('garments', 'garments.py'),
    ('shoes', 'shoes.py'),
    ('hair', 'hair.py'),
    ('accessories', 'accessories.py'),
    ('polish', 'polish.py'),
]


def run_blender(script, args, log_path):
    cmd = [BLENDER, '-b', '--factory-startup', '--python', script, '--'] + args
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = p.stdout + '\n' + p.stderr
    open(log_path, 'w', encoding='utf-8').write(out)
    bad = 'Traceback' in out
    tail = [l for l in out.splitlines() if l.startswith('[') or 'Error' in l or 'Traceback' in l]
    return (not bad) and p.returncode == 0, time.time() - t0, tail


def main(argv):
    outdir = os.path.abspath(argv[0])
    only = argv[argv.index('--only') + 1].split(',') if '--only' in argv else [s for s, _ in STAGES]
    skip_missing = '--skip-missing' in argv
    char = os.path.join(outdir, 'char.json')
    assert os.path.exists(char), 'no char.json in ' + outdir
    body_blend = os.path.join(outdir, 'body.blend')
    body_json = os.path.join(outdir, 'body.json')
    cur = None
    ok_all = True
    for name, script in STAGES:
        if name not in only:
            continue
        sp = os.path.join(D, script)
        if not os.path.exists(sp):
            print('[assemble] %-12s MISSING recipe %s%s' % (name, script, ' (skipped)' if skip_missing else ''))
            if not skip_missing:
                ok_all = False
                break
            continue
        log_path = os.path.join(outdir, 'build_%s.log' % name)
        if name == 'body':
            args = [char, body_blend, '--report', body_json]
            out_blend = body_blend
        else:
            # with --only, continue from the newest earlier output
            inp = cur
            if inp is None:
                idx = [k for k, (nm, _) in enumerate(STAGES) if nm == name][0]
                for nm, _ in reversed(STAGES[1:idx]):
                    cand = os.path.join(outdir, 'char_%s.blend' % nm[0])
                    if os.path.exists(cand):
                        inp = cand
                        break
            inp = inp or body_blend
            out_blend = os.path.join(outdir, 'char_%s.blend' % name[0])
            args = [char, body_json, inp, out_blend]
        ok, dt, tail = run_blender(sp, args, log_path)
        print('[assemble] %-12s %s %5.1fs -> %s' % (name, 'ok  ' if ok else 'FAIL', dt, os.path.basename(out_blend)))
        for l in tail[-12:]:
            print('    ' + l)
        if not ok:
            ok_all = False
            print('[assemble] see ' + log_path)
            break
        cur = out_blend
    if ok_all and cur:
        final = os.path.join(outdir, 'char.blend')
        shutil.copyfile(cur, final)
        print('[assemble] char.blend = %s' % os.path.basename(cur))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
