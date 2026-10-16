"""Class label maps of a turnaround at the sheet's resolution.

Ink takes the class of the nearest colour region, so a run's edge is the outer edge of its outline.
"""
import numpy as np
from scipy import ndimage as ndi


class Labels:
    """lab[r, c] is the class id, 0 for background."""

    def __init__(self, lab, ids, names, panels, ink, bg):
        self.lab, self.ids, self.names, self.panels, self.ink, self.bg = lab, ids, names, panels, ink, bg

    def mask(self, classes, panel=None):
        if isinstance(classes, str):
            classes = [classes]
        m = np.isin(self.lab, [self.ids[c] for c in classes if c in self.ids])
        if panel is not None:
            c0, c1 = self.panels[panel]
            out = np.zeros_like(m)
            out[:, c0:c1] = m[:, c0:c1]
            return out
        return m

    def name_of(self, cid):
        for k, v in self.ids.items():
            if v == cid:
                return k
        return str(cid)

    def figure(self, panel=None):
        m = self.lab > 0
        if panel is not None:
            c0, c1 = self.panels[panel]
            out = np.zeros_like(m)
            out[:, c0:c1] = m[:, c0:c1]
            return out
        return m


def majority(lab, passes=2):
    for _ in range(passes):
        H, W = lab.shape
        pad = np.pad(lab, 1, mode='edge')
        stack = np.stack([pad[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], 0)
        best = np.zeros_like(lab)
        bestc = np.zeros(lab.shape, int)
        for v in np.unique(lab):
            c = (stack == v).sum(0)
            upd = c > bestc
            best[upd] = v
            bestc[upd] = c[upd]
        lab = best
    return lab


def fill_unassigned(lab, unassigned):
    lab = lab.copy()
    src = unassigned | (lab == 0)
    if not src.any():
        return lab
    idx = ndi.distance_transform_edt(src, return_distances=False, return_indices=True)
    filled = lab[idx[0], idx[1]]
    lab[unassigned] = filled[unassigned]
    return lab


def background_mask(whitish, ink, c0, c1, keep_bands):
    """Whitish pixels connected to the panel border or enclosed by ink."""
    sub = np.zeros_like(whitish)
    sub[:, c0:c1] = whitish[:, c0:c1]
    comp, n = ndi.label(sub)
    bg = np.zeros_like(whitish)
    if n == 0:
        return bg
    objs = ndi.find_objects(comp)
    H = whitish.shape[0]
    sizes = ndi.sum(np.ones_like(sub, dtype=np.int64), comp, index=np.arange(1, n + 1))
    figure_area = max(1, int((~whitish[:, c0:c1]).sum()))
    for i, sl in enumerate(objs, 1):
        rs, cs = sl
        touches = rs.start == 0 or rs.stop == H or cs.start <= c0 or cs.stop >= c1
        if touches:
            bg[comp == i] = True
            continue
        if any(rs.start >= r0 and rs.stop <= r1 for r0, r1 in keep_bands):
            continue
        if sizes[i - 1] < 0.004 * figure_area:
            continue
        r0, r1 = max(0, rs.start - 3), min(H, rs.stop + 3)
        q0, q1 = max(0, cs.start - 3), min(whitish.shape[1], cs.stop + 3)
        m = comp[r0:r1, q0:q1] == i
        ring = ndi.binary_dilation(m, iterations=2) & ~m
        if ring.sum() == 0 or ink[r0:r1, q0:q1][ring].mean() > 0.5:
            bg[comp == i] = True
    return bg


def classify(img, sheet, speckle=60):
    """img is float32 RGB 0..255, class ids start at 1 in `parts` order."""
    cal = sheet['calibration']
    pal = sheet['palette']['colours']
    tol = float(sheet['palette'].get('tol', 20))
    parts = sheet['parts']
    SH, SW, _ = img.shape
    names = [n for n in pal if n != 'ink']
    cols = np.array([pal[n] for n in names], np.float32)
    dist = np.full((SH, SW), np.inf, np.float32)
    near = np.zeros((SH, SW), np.int32)
    for k in range(len(names)):
        d = np.sqrt(((img - cols[k]) ** 2).sum(2))
        upd = d < dist
        dist[upd] = d[upd]
        near[upd] = k
    lum = img.mean(2)
    ink = (dist > tol) & (lum < 190)
    whitish = (lum > 235) & (img.min(2) > 225)
    ids = {p['name']: i + 1 for i, p in enumerate(parts)}
    colour_to_parts = {}
    for p in parts:
        for c in p['colours']:
            colour_to_parts.setdefault(c, []).append(p)
    rows = np.arange(SH)[:, None]
    lab_all = np.zeros((SH, SW), np.int32)
    bg_all = np.ones((SH, SW), bool)
    panels = {}
    floor = int(cal['floor_row'])
    for pan in cal['panels']:
        c0, c1 = int(pan['c0']), int(pan['c1'])
        panels[pan['name']] = (c0, c1)
        keep = [tuple(b) for p in parts if 'white' in p['colours']
                and (not p.get('panels') or pan['name'] in p['panels']) for b in p['bands']]
        bg = background_mask(whitish, ink, c0, c1, keep)
        bg[floor - 3:, :] = True
        for r in range(max(0, floor - 60), floor):
            row = ~bg[r, c0:c1]
            if row.mean() < 0.5:
                continue
            d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            st = np.nonzero(d == 1)[0]
            en = np.nonzero(d == -1)[0]
            if len(st) and (en - st).max() > 0.5 * (c1 - c0):
                bg[r, c0:c1] = True
        lab = np.zeros((SH, SW), np.int32)
        un = np.zeros((SH, SW), bool)
        for k, nm in enumerate(names):
            m = (near == k) & ~bg & ~ink
            m[:, :c0] = False
            m[:, c1:] = False
            if nm == 'white':
                m &= whitish
            if not m.any():
                continue
            assigned = np.zeros_like(m)
            for p in colour_to_parts.get(nm, []):
                if p.get('panels') and pan['name'] not in p['panels']:
                    continue
                band = np.zeros((SH, 1), bool)
                for r0, r1 in p['bands']:
                    band |= (rows >= r0) & (rows < r1)
                sel = m & band & ~assigned
                lab[sel] = ids[p['name']]
                assigned |= sel
            un |= m & ~assigned
        figure = ~bg & (lab == 0)
        figure[:, :c0] = False
        figure[:, c1:] = False
        lab = fill_unassigned(lab, figure | un)
        lab[bg] = 0
        # fill pale seams up to 8 px wide that have figure on both sides
        seam = np.zeros_like(bg)
        for axis_ in (1, 0):
            arr = bg if axis_ == 1 else bg.T
            figc = (lab > 0) if axis_ == 1 else (lab > 0).T
            out_ = np.zeros_like(arr)
            for r_ in range(arr.shape[0]):
                line = arr[r_]
                if axis_ == 1:
                    seg = line[c0:c1]
                    off = c0
                else:
                    seg = line
                    off = 0
                if not seg.any():
                    continue
                d_ = np.diff(np.concatenate([[0], seg.astype(int), [0]]))
                st_, en_ = np.nonzero(d_ == 1)[0], np.nonzero(d_ == -1)[0]
                fl = figc[r_]
                for a_, b_ in zip(st_, en_):
                    if b_ - a_ <= 8 and a_ > 0 and b_ + off < fl.shape[0] and fl[a_ + off - 1] and fl[b_ + off]:
                        out_[r_, a_ + off:b_ + off] = True
            seam |= out_ if axis_ == 1 else out_.T
        if seam.any():
            bg &= ~seam
            lab = fill_unassigned(lab, seam)
        lab = majority(lab, 2)
        small = np.zeros_like(bg)
        for pid in ids.values():
            comp, n = ndi.label(lab == pid, structure=np.ones((3, 3), int))
            if n == 0:
                continue
            sizes = ndi.sum(np.ones_like(lab), comp, index=np.arange(1, n + 1))
            tiny = np.nonzero(sizes < speckle)[0] + 1
            if len(tiny):
                small |= np.isin(comp, tiny)
        if small.any():
            lab = fill_unassigned(lab, small)
            lab[bg] = 0
        lab_all[:, c0:c1] = lab[:, c0:c1]
        bg_all[:, c0:c1] = bg[:, c0:c1]
    return Labels(lab_all, ids, names, panels, ink, bg_all)


def absorb_islands(labels, max_frac=0.05, ring_frac=0.8, min_px=12):
    """Relabel small components enclosed by one other class (eyes, studs, buttons) and return them."""
    lab = labels.lab
    out = []
    for pname, (c0, c1) in labels.panels.items():
        sub = lab[:, c0:c1]
        for cname, cid in labels.ids.items():
            m = sub == cid
            total = int(m.sum())
            if total == 0:
                continue
            comp, n = ndi.label(m, structure=np.ones((3, 3), int))
            if n <= 1:
                continue
            sizes = ndi.sum(np.ones_like(m), comp, index=np.arange(1, n + 1))
            objs = ndi.find_objects(comp)
            for i, sl in enumerate(objs, 1):
                px = int(sizes[i - 1])
                if px < min_px or px > max_frac * total:
                    continue
                rs, cs = sl
                r0, r1 = max(0, rs.start - 2), min(sub.shape[0], rs.stop + 2)
                q0, q1 = max(0, cs.start - 2), min(sub.shape[1], cs.stop + 2)
                mm = comp[r0:r1, q0:q1] == i
                ring = ndi.binary_dilation(mm, iterations=2) & ~mm
                vals = sub[r0:r1, q0:q1][ring]
                vals = vals[vals != 0]
                if len(vals) == 0:
                    continue
                host, cnt = np.unique(vals, return_counts=True)
                j = int(cnt.argmax())
                if host[j] == cid or cnt[j] < ring_frac * len(vals):
                    continue
                rr, cc = np.nonzero(comp == i)
                out.append(dict(panel=pname, cls=cname, host=labels.name_of(int(host[j])), px=px,
                                bbox=[int(rr.min()), int(rr.max()) + 1, int(cc.min()) + c0, int(cc.max()) + 1 + c0],
                                centre=[float(rr.mean()), float(cc.mean()) + c0]))
                sub[comp == i] = host[j]
    return out


def preview(labels, sheet, scale=0.25):
    from PIL import Image
    pal = sheet['palette']['colours']
    parts = sheet['parts']
    H, W = labels.lab.shape
    rgb = np.full((H, W, 3), 255, np.uint8)
    for p in parts:
        rgb[labels.lab == labels.ids[p['name']]] = pal[p['colours'][0]]
    im = Image.fromarray(rgb)
    if scale != 1:
        im = im.resize((int(W * scale), int(H * scale)), Image.NEAREST)
    return im
