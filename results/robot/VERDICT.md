# Verdict: FAIL

character: robot
time: 2026-10-17 04:51

| check | result | detail |
|---|---|---|
| audit | FAIL | 15 objects, 2 failing gates |
|  |  | Glove_L: floating 6.4 |
|  |  | Glove_R: floating 6.4 |
| budget | PASS | 3774 triangles in 15 objects (budget 9000) |
| accounting | PASS | 11 parts, 0 missing |
| sheet match | FAIL | silhouette IoU front 0.911, side 0.774, back 0.853; 154 mismatch blobs, 292980 mm2 in total; 7 over 8000 mm2 |
|  |  | side: drawn nothing, built skin, 31148 mm2 at y -0.02..0.34, z 1.21..1.59 |
|  |  | side: drawn nothing, built boot_soles, 14873 mm2 at y -0.16..0.24, z 0.00..0.04 |
|  |  | side: drawn skin, built nothing, 12478 mm2 at y -0.09..0.15, z 1.40..1.60 |
|  |  | side: drawn nothing, built boot, 11135 mm2 at y -0.15..0.23, z 0.05..0.08 |
|  |  | side: drawn skin, built shoulder_pads, 10197 mm2 at y 0.03..0.16, z 0.97..1.11 |
|  |  | back: drawn skin, built nothing, 9124 mm2 at x 0.05..-0.21, z 1.25..1.60 |
|  |  | side: drawn jetpack, built nothing, 8128 mm2 at y 0.18..0.30, z 0.89..1.22 |
|  |  | back: drawn boot, built nothing, 7575 mm2 at x -0.08..-0.34, z 0.04..0.32 |
| surface | FAIL | 0 dents, 5 needles, 64 creases, worst symmetry p95 35.7 mm, 0 repaired self-intersections |
|  |  | Armour_shorts_L: dents 0 (worst - mm), needles 0, creases 2 |
|  |  | Armour_shorts_R: dents 0 (worst - mm), needles 0, creases 2 |
|  |  | Body: dents 0 (worst - mm), needles 0, creases 54, symmetry p95 2.9 mm |
|  |  | Chest_armour_plate: dents 0 (worst - mm), needles 0, creases 1, symmetry p95 17.8 mm |
|  |  | Jetpack_0: dents 0 (worst - mm), needles 4, creases 0 |
|  |  | Shoulder_pads: dents 0 (worst - mm), needles 0, creases 0, symmetry p95 35.7 mm |
|  |  | Shoulder_pads_Sleeve_L: dents 0 (worst - mm), needles 0, creases 1 |
|  |  | Shoulder_pads_Sleeve_R: dents 0 (worst - mm), needles 1, creases 4 |
| form | PASS | concave edges 0.173 of the edge length (minimum 0.05) |

## Part accounting

| part | status | by |
|---|---|---|
| part: visor | painted | texture |
| part: chest_armour_plate | built | Chest_armour_plate |
| part: shoulder_pads | built | Shoulder_pads, Shoulder_pads_Sleeve_L, Shoulder_pads_Sleeve_R |
| part: waist_belt | painted | texture |
| part: armour_shorts | built | Armour_shorts, Armour_shorts_L, Armour_shorts_R |
| part: boot | built | Boot_L, Boot_R |
| part: boot_soles | built | Boot_soles_L, Boot_soles_R |
| part: glove | built | Glove_L, Glove_R |
| part: jetpack | built | Jetpack_0 |
| hair cap | omitted: no hair drawn |  |
| face (blush, eye, iris) | painted | tex/face_atlas.png |

## Objects

| object | part | class | tris |
|---|---|---|---|
| Armour_shorts | garment | armour_shorts | 174 |
| Armour_shorts_L | garment | armour_shorts | 113 |
| Armour_shorts_R | garment | armour_shorts | 122 |
| Body | body | skin | 2494 |
| Boot_L | shoe | boot | 144 |
| Boot_R | shoe | boot | 144 |
| Boot_soles_L | shoe | boot_soles | 44 |
| Boot_soles_R | shoe | boot_soles | 44 |
| Chest_armour_plate | garment | chest_armour_plate | 263 |
| Glove_L | garment | glove | 16 |
| Glove_R | garment | glove | 16 |
| Jetpack_0 | accessory | jetpack | 66 |
| Shoulder_pads | garment | shoulder_pads | 70 |
| Shoulder_pads_Sleeve_L | garment | shoulder_pads | 32 |
| Shoulder_pads_Sleeve_R | garment | shoulder_pads | 32 |

Renders in the run folder: judge/turntable_{clay,wire,flat}.png, judge/head_{clay,flat}.png, judge/cmp/compare_{front,side,back}.png
