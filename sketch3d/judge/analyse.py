"""Compare the class renders with the sheet's class masks and list the defects.

    python sketch3d/judge/analyse.py <outdir>
"""
import os, sys, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import metrics

TOL_MM = 6.0  # smaller departures are not defects, an ink line is about 3 mm
MIN_BLOB_MM = 8.0
BAND_MM = 10.0


def srgb(v):
    v = np.asarray(v, np.float32)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(np.maximum(v, 1e-6), 1 / 2.4) - 0.055)


def class_masks_from_render(png, palette):
    rgba = np.asarray(Image.open(png).convert('RGBA')).astype(np.float32) / 255.0
    im, alpha = rgba[:, :, :3], rgba[:, :, 3]
    H, W, _ = im.shape
    names = list(palette)
    cols = srgb(np.array([palette[n] for n in names], np.float32))
    d = ((im[:, :, None, :] - cols[None, None, :, :]) ** 2).sum(3)
    near = d.argmin(2)
    dmin = d.min(2)
    lab = np.zeros((H, W), np.int32)
    for k, n in enumerate(names):
        lab[(near == k) & (alpha > 0.5) & (dmin < 0.05)] = k + 1
    return lab, names


def runs_of(mask_row, min_w=2):
    d = np.diff(np.concatenate([[0], mask_row.astype(np.int8), [0]]))
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a >= min_w]


def match_runs(rs, rr, max_dc):
    """Pair drawn and built runs greedily by centre distance."""
    pairs, used = [], set()
    for a in rs:
        ca = 0.5 * (a[0] + a[1])
        best, bd = None, 1e9
        for j, b in enumerate(rr):
            if j in used:
                continue
            cb = 0.5 * (b[0] + b[1])
            dd = abs(cb - ca)
            if dd < bd and dd < max_dc:
                best, bd = j, dd
        if best is not None:
            used.add(best)
            pairs.append((a, rr[best]))
    matched_d = set(id(p[0]) for p in pairs)
    un_d = [a for a in rs if id(a) not in matched_d]
    un_b = [b for j, b in enumerate(rr) if j not in used]
    return pairs, un_d, un_b


def main(argv):
    outdir = os.path.abspath(argv[0])
    C = json.load(open(os.path.join(outdir, 'char.json')))
    cal = C['calibration']
    ppm = cal['ppm']
    floor_row = int(cal['floor_row'])
    lab_sheet = np.load(os.path.join(outdir, 'labels.npy')).astype(np.int32)
    ink_p = os.path.join(outdir, 'labels_ink.npy')
    ink_sheet = np.load(ink_p).astype(bool) if os.path.exists(ink_p) else np.zeros(lab_sheet.shape, bool)
    # grow the ink by 2 px so a class ending on an outline still matches
    for _ in range(2):
        g = ink_sheet.copy()
        g[1:, :] |= ink_sheet[:-1, :]; g[:-1, :] |= ink_sheet[1:, :]; g[:, 1:] |= ink_sheet[:, :-1]; g[:, :-1] |= ink_sheet[:, 1:]
        ink_sheet = g
    lj = json.load(open(os.path.join(outdir, 'labels.json')))
    ids0 = lj['ids']
    ids = dict(ids0)
    cmp_dir = os.path.join(outdir, 'judge', 'cmp')
    V = json.load(open(os.path.join(cmp_dir, 'compare_views.json')))
    palette = V['palette']
    SW, SH = V['sheet_size']
    roles = C['roles']
    report = dict(views={}, defects=[])
    band_px = max(4, int(BAND_MM / 1000.0 * ppm))
    for view, vd in V['views'].items():
        c0, c1 = vd['c0'], vd['c1']
        mid = vd['mid']
        W_px = c1 - c0
        lab_r, names = class_masks_from_render(vd['path'], palette)
        centre_col = float(vd.get('centre_col', mid))
        shift = int(round(centre_col - c0 - W_px / 2.0))

        def place(lr):
            R_ = np.zeros((SH, W_px), np.int32)
            if shift >= 0:
                R_[:, shift:] = lr[:, :W_px - shift]
            else:
                R_[:, :W_px + shift] = lr[:, -shift:]
            return R_
        R = place(lab_r)
        S = lab_sheet[:, c0:c1]
        INK = ink_sheet[:, c0:c1]
        flip = False
        if view == 'side' and 'side_sign' in (C.get('datum') or {}):
            # the render faces image-left, mirror it for a right-facing profile
            if float(C['datum']['side_sign']) < 0:
                R = place(lab_r[:, ::-1])
                flip = True
        elif view == 'side':
            fig_s = S > 0
            Rf = place(lab_r[:, ::-1])
            fig_r = R > 0
            fig_rf = Rf > 0
            iou_a = (fig_s & fig_r).sum() / max(1, (fig_s | fig_r).sum())
            iou_b = (fig_s & fig_rf).sum() / max(1, (fig_s | fig_rf).sum())
            if iou_b > iou_a:
                R = Rf
                flip = True
        vr = dict(flip=flip, classes={})
        ov = np.full((SH, W_px, 3), 255, np.uint8)
        fs, frr = S > 0, R > 0
        ov[fs & ~frr] = (60, 90, 255)
        ov[frr & ~fs] = (255, 60, 60)
        ov[fs & frr] = (150, 80, 200)
        Image.fromarray(ov).resize((W_px // 2, SH // 2)).save(os.path.join(cmp_dir, 'compare_%s.png' % view))
        fig_iou = metrics.iou(fs, frr)
        vr['figure_iou'] = round(float(fig_iou), 3)
        z_of = lambda r: (floor_row - r) / ppm
        name_to_rid = {n: k + 1 for k, n in enumerate(names)}
        rid_to_name = {v: k for k, v in name_to_rid.items()}
        # all hair classes count as one, painted classes count as skin
        hair_set = set(roles.get('hair', []))
        painted = set(k for k, e in (C.get('garments') or {}).items() if e.get('painted'))
        pj = os.path.join(outdir, 'judge', 'painted.json')
        if os.path.exists(pj):
            painted |= set(json.load(open(pj)))
        painted |= set((C.get('hair') or {}).get('painted', []))
        skin_set = set(roles.get('skin', ['skin']))

        def group(n):
            if n in painted:
                return 'painted'
            if n in hair_set or n == 'hair' or n.startswith('hair_'):
                return 'hair'
            if n in skin_set:
                return 'skin'
            return n
        groups = sorted(set(group(n) for n in list(ids0) + list(names)))
        gid = {g: k + 1 for k, g in enumerate(groups)}
        lut_s = np.zeros(max(ids0.values()) + 1, np.int32)
        for n, cid in ids0.items():
            lut_s[cid] = gid[group(n)]
        lut_r = np.zeros(len(names) + 1, np.int32)
        for n, rid in name_to_rid.items():
            lut_r[rid] = gid[group(n)]
        S = lut_s[S]
        R = lut_r[R]
        ids = {g: k for g, k in gid.items()}
        name_to_rid = dict(gid)
        rid_to_name = {v: k for k, v in gid.items()}
        min_blob_px = (MIN_BLOB_MM / 1000.0 * ppm) ** 2
        blobs = []
        sgn_u = float((C.get('datum') or {}).get('side_sign', 1.0)) if view == 'side' else 1.0
        u_of = (lambda c: sgn_u * (c - (mid - c0)) / ppm) if view != 'back' else (lambda c: ((mid - c0) - c) / ppm)
        for cname, cid in ids.items():
            ms = S == cid
            if not ms.any():
                continue
            if cname == 'painted':
                # paint matches any built surface under it
                continue
            rid = name_to_rid.get(cname)
            mr = (R == rid) if rid else np.zeros_like(ms)
            diff = ms & ~mr
            if not diff.any():
                continue
            for pm, area in metrics.blobs(diff, min_blob_px):
                rr, cc = np.nonzero(pm)
                built_ids = R[pm]
                built_ids = built_ids[built_ids > 0]
                if len(built_ids):
                    bc = np.bincount(built_ids)
                    dname = rid_to_name.get(int(bc.argmax()), '?')
                else:
                    dname = 'nothing'
                if len(built_ids) < 0.5 * area:
                    dname = 'nothing'
                if dname not in ('nothing', '?') and ids.get(dname) is not None and not (S == ids[dname]).any():
                    # the built class is not labelled in this view, so it cannot be judged here
                    blobs.append(dict(kind='unjudged', view=view, drawn=cname, built=dname, area_mm2=round(metrics.area_mm2(area, ppm)),
                                      z=[round(z_of(rr.max()), 3), round(z_of(rr.min()), 3)], u=[round(u_of(cc.min()), 3), round(u_of(cc.max()), 3)],
                                      mm=round(float(np.sqrt(area)) / ppm * 1000, 1), why='the %s class has no pixels in the %s panel' % (dname, view)))
                    continue
                # a drawn patch enclosed by the built class is texture detail
                kind_ = 'mismatch'
                if dname not in ('nothing', '?'):
                    ring = ndi.binary_dilation(pm, structure=np.ones((3, 3), bool)) & ~pm
                    n_ring = int(ring.sum())
                    if n_ring and int((R[ring] == name_to_rid.get(dname, -1)).sum()) >= 0.9 * n_ring:
                        kind_ = 'enclosed'
                blobs.append(dict(kind=kind_, view=view, drawn=cname, built=dname, area_mm2=round(metrics.area_mm2(area, ppm)),
                                  z=[round(z_of(rr.max()), 3), round(z_of(rr.min()), 3)],
                                  u=[round(u_of(cc.min()), 3), round(u_of(cc.max()), 3)],
                                  mm=round(float(np.sqrt(area)) / ppm * 1000, 1),
                                  **({'why': 'drawn %s lies wholly inside the built %s: a decoration in the colour of %s, or a hole the model closed' % (cname, dname, cname)} if kind_ == 'enclosed' else {})))
        for dname, rid in name_to_rid.items():
            mr = R == rid
            cid = ids.get(dname)
            ms = (S == cid) if cid is not None else np.zeros_like(mr)
            diff = mr & ~ms & ~((S == 0) & INK)
            if not diff.any():
                continue
            for pm, area in metrics.blobs(diff, min_blob_px):
                rr, cc = np.nonzero(pm)
                drawn_ids = S[pm]
                drawn_ids = drawn_ids[drawn_ids > 0]
                if len(drawn_ids) >= 0.5 * area:
                    bc = np.bincount(drawn_ids)
                    cname = [n for n, i in ids.items() if i == int(bc.argmax())][0]
                else:
                    cname = 'nothing'
                if cname != 'nothing':
                    continue
                blobs.append(dict(kind='mismatch', view=view, drawn='nothing', built=dname, area_mm2=round(metrics.area_mm2(area, ppm)),
                                  z=[round(z_of(rr.max()), 3), round(z_of(rr.min()), 3)],
                                  u=[round(u_of(cc.min()), 3), round(u_of(cc.max()), 3)],
                                  mm=round(float(np.sqrt(area)) / ppm * 1000, 1)))
        for b in blobs:
            b['msg'] = '%s: drawn %s, built %s, %d mm2 at %s %.2f..%.2f, z %.2f..%.2f' % (
                view, b['drawn'], b['built'], b['area_mm2'], 'y' if view == 'side' else 'x', b['u'][0], b['u'][1], b['z'][0], b['z'][1])
        vr['blobs'] = sorted(blobs, key=lambda b: -b['area_mm2'])
        report['defects'] += blobs
        for cname, cid in ids.items():
            ms = S == cid
            if not ms.any():
                continue
            # keep components of at least 3 % of the class, the rest is detail
            comp, ncomp = ndi.label(ms, structure=np.ones((3, 3), int))
            if ncomp > 1:
                sizes = ndi.sum(np.ones_like(ms), comp, index=np.arange(1, ncomp + 1))
                keep = [i + 1 for i in range(ncomp) if sizes[i] >= 0.03 * sizes.sum()]
                ms = np.isin(comp, keep)
            rid = name_to_rid.get(cname)
            mr = (R == rid) if rid else np.zeros_like(ms)
            area_s, area_r = int(ms.sum()), int(mr.sum())
            inter = int((ms & mr).sum())
            union = int((ms | mr).sum())
            iou = inter / max(1, union)
            cr = dict(iou=round(iou, 3), drawn_px=area_s, built_px=area_r,
                      missing_frac=round(1 - inter / max(1, area_s), 3), extra_frac=round((area_r - inter) / max(1, area_r), 3) if area_r else None)
            rows_s = np.nonzero(ms.any(1))[0]
            rows_r = np.nonzero(mr.any(1))[0]
            z_of = lambda r: (floor_row - r) / ppm
            cr['drawn_z'] = [round(z_of(rows_s.max()), 3), round(z_of(rows_s.min()), 3)]
            if len(rows_r):
                cr['built_z'] = [round(z_of(rows_r.max()), 3), round(z_of(rows_r.min()), 3)]
            if area_r == 0:
                built_anywhere = any(o == cname for o in palette)
                report['defects'].append(dict(kind='hidden_part' if built_anywhere else 'missing_part', view=view, cls=cname,
                                              z=cr['drawn_z'], mm=round(float(np.sqrt(area_s)) / ppm * 1000, 1),
                                              msg='%s: drawn in the %s view (%d px, z %.2f..%.2f) but %s' % (
                                                  cname, view, area_s, cr['drawn_z'][0], cr['drawn_z'][1], 'nothing visible' if built_anywhere else 'no object built')))
                vr['classes'][cname] = cr
                continue
            top_d = z_of(rows_s.min()) - z_of(rows_r.min())
            bot_d = z_of(rows_s.max()) - z_of(rows_r.max())
            for what, dz in (('top', top_d), ('bottom', bot_d)):
                if abs(dz) * 1000 > TOL_MM * 1.5:
                    report['defects'].append(dict(kind='extent', view=view, cls=cname, where=what, mm=round(dz * 1000, 1),
                                                  msg='%s %s: built %s is %.0f mm %s than drawn' % (cname, view, what, abs(dz) * 1000, 'lower' if dz > 0 else 'higher')))
            bands = []
            r_lo, r_hi = min(rows_s.min(), rows_r.min() if len(rows_r) else rows_s.min()), max(rows_s.max(), rows_r.max() if len(rows_r) else rows_s.max())
            max_dc = 0.08 * ppm
            for r0 in range(int(r_lo), int(r_hi) + 1, band_px):
                r1 = min(r0 + band_px, int(r_hi) + 1)
                dws, dcs, ws_all, wr_all, miss_w, extra_w = [], [], [], [], [], []
                for r in range(r0, r1):
                    rs_, rr_ = runs_of(ms[r]), runs_of(mr[r])
                    pairs, un_d, un_b = match_runs(rs_, rr_, max_dc)
                    for a, b in pairs:
                        dws.append((b[1] - b[0]) - (a[1] - a[0]))
                        dcs.append(0.5 * (b[0] + b[1]) - 0.5 * (a[0] + a[1]))
                        ws_all.append(a[1] - a[0]); wr_all.append(b[1] - b[0])
                    miss_w += [a[1] - a[0] for a in un_d]
                    extra_w += [b[1] - b[0] for b in un_b]
                zc = round(z_of(0.5 * (r0 + r1)), 3)
                if not dws:
                    if miss_w and len(miss_w) >= band_px * 0.6:
                        bands.append(dict(z=zc, drawn_w=round(np.mean(miss_w) / ppm * 1000, 1), built_w=0.0))
                    continue
                b = dict(z=zc, drawn_w=round(np.mean(ws_all) / ppm * 1000, 1), built_w=round(np.mean(wr_all) / ppm * 1000, 1),
                         dw=round(float(np.mean(dws) / ppm * 1000), 1), dc=round(float(np.mean(dcs) / ppm * 1000), 1))
                if miss_w and len(miss_w) >= band_px * 0.6:
                    b['missing_run_w'] = round(np.mean(miss_w) / ppm * 1000, 1)
                if extra_w and len(extra_w) >= band_px * 0.6:
                    b['extra_run_w'] = round(np.mean(extra_w) / ppm * 1000, 1)
                bands.append(b)
            cr['bands'] = bands
            # merge consecutive bands with the same sign of error
            def flush(kind, run):
                if len(run) < 2:
                    return
                zs = [b['z'] for b in run]
                vals = [b['dw'] if kind == 'width' else b['dc'] for b in run]
                worst = max(vals, key=abs)
                sgn = '+' if worst > 0 else '-'
                if kind == 'width':
                    msg = '%s %s: z %.2f..%.2f built %s%.0f mm %s than drawn (%s)' % (cname, view, min(zs), max(zs), sgn, abs(worst), 'wider' if worst > 0 else 'narrower',
                                                                                        ', '.join('%.0f/%.0f' % (b['built_w'], b['drawn_w']) for b in run[:6]))
                else:
                    side = ('toward +x (character left)' if worst > 0 else 'toward -x (character right)') if view != 'side' else ('toward the back' if worst > 0 else 'toward the front')
                    if view == 'back':
                        side = side.replace('left', 'right~').replace('right)', 'left)').replace('right~', 'left')
                    msg = '%s %s: z %.2f..%.2f built run centred %.0f mm off %s' % (cname, view, min(zs), max(zs), abs(worst), side)
                report['defects'].append(dict(kind=kind, view=view, cls=cname, z=[round(min(zs), 3), round(max(zs), 3)], mm=round(float(worst), 1), msg=msg))
            for kind, key in (('width', 'dw'), ('position', 'dc')):
                run = []
                for b in bands:
                    v = b.get(key)
                    if v is not None and abs(v) > TOL_MM and (not run or (run[-1][key] > 0) == (v > 0)):
                        run.append(b)
                    else:
                        flush(kind, run)
                        run = [b] if (v is not None and abs(v) > TOL_MM) else []
                flush(kind, run)
            for key, kind in (('missing_run_w', 'missing_run'), ('extra_run_w', 'extra_run')):
                run = []
                for b in bands + [None]:
                    if b is not None and (b.get(key) is not None or (key == 'missing_run_w' and b.get('built_w') == 0.0)):
                        run.append(b)
                        continue
                    if len(run) >= 2:
                        zs = [q['z'] for q in run]
                        w = np.mean([q.get(key, q.get('drawn_w', 0)) for q in run])
                        report['defects'].append(dict(kind=kind, view=view, cls=cname, z=[round(min(zs), 3), round(max(zs), 3)], mm=round(float(w), 1),
                                                      msg='%s %s: z %.2f..%.2f a drawn run ~%.0f mm wide has %s' % (cname, view, min(zs), max(zs), w,
                                                                                                                  'nothing built' if kind == 'missing_run' else 'an extra built run')))
                    run = []
            vr['classes'][cname] = cr
        report['views'][view] = vr
    report['defects'].sort(key=lambda d: -(d.get('area_mm2', 0) if d['kind'] == 'mismatch' else abs(d.get('mm', 0))))
    report['mismatch_mm2'] = sum(d['area_mm2'] for d in report['defects'] if d['kind'] == 'mismatch')
    json.dump(report, open(os.path.join(cmp_dir, 'compare.json'), 'w'), indent=1)
    print('[analyse] figure IoU: %s' % ', '.join('%s %.3f' % (v, report['views'][v]['figure_iou']) for v in report['views']))
    blobs = [d for d in report['defects'] if d['kind'] == 'mismatch']
    total = sum(b['area_mm2'] for b in blobs)
    print('[analyse] %d mismatch blobs, %d mm2 in total:' % (len(blobs), total))
    for d in blobs[:40]:
        print('   %7d mm2  %s' % (d['area_mm2'], d['msg']))
    others = [d for d in report['defects'] if d['kind'] != 'mismatch']
    print('[analyse] %d band/extent defects (largest 15):' % len(others))
    for d in others[:15]:
        print('   %-12s %6.1f mm  %s' % (d['kind'], d.get('mm', 0), d['msg']))
    report['mismatch_mm2'] = total
    return report


if __name__ == '__main__':
    main(sys.argv[1:])
