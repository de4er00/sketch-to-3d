# Verdict: FAIL

prop: chest
time: 2026-10-17 04:52

| check | result | detail |
|---|---|---|
| audit | PASS | 3 objects, 0 failing gates |
| budget | n/a | 126 triangles in 3 objects, n/a for props |
| accounting | PASS | 5 parts, 0 missing |
| sheet match | PASS | silhouette IoU front 0.994, side 0.991; 19 mismatch blobs, 5408 mm2 in total; 0 over 8000 mm2 |
|  |  | front: drawn lock_plate, built chest_box, 935 mm2 at x -0.05..0.07, z 0.24..0.29 |
|  |  | side: drawn chest_box, built chest_lid, 575 mm2 at y -0.03..-0.20, z 0.27..0.38 |
|  |  | front: drawn chest_lid, built chest_box, 498 mm2 at x 0.03..0.24, z 0.37..0.38 |
|  |  | side: drawn chest_box, built chest_lid, 498 mm2 at y 0.21..0.04, z 0.38..0.38 |
|  |  | front: drawn chest_lid, built chest_box, 406 mm2 at x -0.25..-0.04, z 0.37..0.38 |
|  |  | side: drawn chest_box, built nothing, 402 mm2 at y -0.03..-0.20, z 0.00..0.26 |
|  |  | front: drawn chest_lid, built nothing, 314 mm2 at x 0.38..0.39, z 0.42..0.55 |
|  |  | side: drawn lock_plate, built nothing, 229 mm2 at y -0.20..-0.22, z 0.25..0.37 |
| surface | FAIL | 0 dents, 25 needles, 0 repaired self-intersections; creases and symmetry n/a for props |
|  |  | Chest_lid_0: dents 0 (worst - mm), needles 19, creases 0 |
|  |  | Lock_plate_0: dents 0 (worst - mm), needles 6, creases 0 |
| form | n/a | concave edges 0.116 of the edge length, n/a for props |

## Part accounting

| part | status | by |
|---|---|---|
| part: chest_box | built | Chest_box_0 |
| part: chest_lid | built | Chest_lid_0 |
| part: iron_straps | painted | texture |
| part: lock_plate | built | Lock_plate_0 |
| part: keyhole | painted | texture |

## Objects

| object | part | class | tris |
|---|---|---|---|
| Chest_box_0 | prop | chest_box | 12 |
| Chest_lid_0 | prop | chest_lid | 72 |
| Lock_plate_0 | prop | lock_plate | 42 |

Renders in the run folder: judge/turntable_{clay,wire,flat}.png, judge/cmp/compare_{front,side}.png
