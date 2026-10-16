"""Station tables per chain: rows where a drawn contour turns, with width and depth."""
from . import geometry as G


def class_boundaries(L, panel, rows, col_of_row, log=None):
    c0, c1 = L.panels[panel]
    out = []
    prev = None
    for r in rows:
        c = int(round(col_of_row(r)))
        c = min(max(c, c0), c1 - 1)
        v = int(L.lab[r, c])
        if v == 0:
            continue
        if prev is not None and v != prev[1] and r - prev[0] < 4:
            out.append((r, L.name_of(prev[1]), L.name_of(v)))
        prev = (r, v)
    return out


def pick_rows(rows, curves, forced, F, tol_mm=0.9, corner_deg=10.0, min_gap_mm=10.0, log=None):
    """Merge dp rows into forced rows at least min_gap_mm apart."""
    tol_px = tol_mm / 1000.0 * F.ppm
    gap_px = min_gap_mm / 1000.0 * F.ppm
    dp = G.dp_rows(rows, curves, tol_px=tol_px, min_gap=max(3, int(gap_px * 0.5)), corner_deg=corner_deg)
    forced = sorted(set(int(r) for r in forced if rows[0] <= r <= rows[-1]))
    # forced rows within 4 px are one boundary drawn thick
    fz = []
    for r in forced:
        if fz and r - fz[-1] < 4:
            continue
        fz.append(r)
    forced = fz
    out = list(forced)
    for r in dp:
        if all(abs(r - f) >= gap_px for f in out):
            out.append(r)
    out = sorted(set(out))
    merged = []
    for r in out:
        if merged and r - merged[-1] < gap_px and not (r in forced and merged[-1] in forced):
            if r in forced:
                merged[-1] = r
            continue
        merged.append(r)
    return merged, dp


def table(rows_pick, F, front, side, datum_col, side_sign, mid, front_sign=1, tags=None, prov=None):
    """Station dicts from front {row: (xl, xr)} and side {row: (front_col, back_col, src)}."""
    out = []
    for r in rows_pick:
        st = dict(row=int(r), z=round(F.z(r), 5), tags=sorted(tags.get(r, [])) if tags else [])
        if r in front:
            xl, xr = front[r]
            st['w_mm'] = round(F.mm(xr - xl), 1)
            st['x_c_mm'] = round(F.mm((0.5 * (xl + xr) - mid) * front_sign), 1)
            st['front_src'] = 'measured'
        else:
            st['w_mm'] = None
            st['front_src'] = 'missing'
        if r in side:
            fc, bc, src = side[r]
            st['d_mm'] = round(F.mm(bc - fc), 1)
            st['y_mid_mm'] = round(F.mm((0.5 * (fc + bc) - datum_col) * side_sign), 1)
            st['side_src'] = src
        else:
            st['d_mm'] = None
            st['side_src'] = 'missing'
        if prov and r in prov:
            st['why'] = prov[r]
        out.append(st)
    return out
