# Verdict: FAIL

character: punk
time: 2026-10-17 04:51

| check | result | detail |
|---|---|---|
| audit | PASS | 14 objects, 0 failing gates |
| budget | PASS | 4438 triangles in 14 objects (budget 9000) |
| accounting | PASS | 14 parts, 0 missing |
| sheet match | FAIL | silhouette IoU front 0.889, side 0.926, back 0.89; 224 mismatch blobs, 187982 mm2 in total; 1 over 8000 mm2 |
|  |  | side: drawn vest, built skin, 12607 mm2 at y 0.08..-0.00, z 0.78..0.99 |
|  |  | side: drawn cargo_shorts, built skin, 7892 mm2 at y 0.09..-0.03, z 0.58..0.76 |
|  |  | side: drawn backpack_straps, built vest, 7289 mm2 at y 0.14..-0.00, z 0.87..1.12 |
|  |  | side: drawn nothing, built shoe_soles, 6679 mm2 at y 0.14..-0.12, z 0.00..0.04 |
|  |  | back: drawn hair, built skin, 4898 mm2 at x -0.01..-0.05, z 1.19..1.44 |
|  |  | front: drawn hair, built nothing, 4762 mm2 at x -0.06..0.06, z 1.44..1.60 |
|  |  | front: drawn backpack_straps, built vest, 4672 mm2 at x -0.14..-0.09, z 0.92..1.12 |
|  |  | side: drawn nothing, built vest, 4669 mm2 at y 0.08..-0.05, z 1.00..1.17 |
| surface | PASS | 0 dents, 0 needles, 54 creases, worst symmetry p95 2.4 mm, 0 repaired self-intersections |
|  |  | Body: dents 0 (worst - mm), needles 0, creases 54, symmetry p95 0.8 mm |
| form | PASS | concave edges 0.189 of the edge length (minimum 0.05) |

## Part accounting

| part | status | by |
|---|---|---|
| part: vest | built | Vest |
| part: t_shirt | painted | texture |
| part: belt | painted | texture |
| part: cargo_shorts | built | Cargo_shorts, Cargo_shorts_L, Cargo_shorts_R |
| part: sock | painted | texture |
| part: shoe | built | Shoe_L, Shoe_R |
| part: shoe_soles | built | Shoe_soles_L, Shoe_soles_R |
| part: glove | built | Glove_L, Glove_R |
| part: spiked_wrist_cuff | built | Spiked_wrist_cuff_L |
| part: backpack_body | built | Backpack_body_0 |
| part: backpack_straps | painted | texture |
| part: ear_stud | painted | texture |
| hair cap | painted | stubble_hair, Hair_Plate_0 |
| face (brow, eye, mouth, nose, sclera) | painted | tex/face_atlas.png |

## Objects

| object | part | class | tris |
|---|---|---|---|
| Backpack_body_0 | accessory | backpack_body | 58 |
| Body | body | skin | 2750 |
| Cargo_shorts | garment | cargo_shorts | 205 |
| Cargo_shorts_L | garment | cargo_shorts | 192 |
| Cargo_shorts_R | garment | cargo_shorts | 183 |
| Glove_L | garment | glove | 32 |
| Glove_R | garment | glove | 32 |
| Hair_Plate_0 | hair | mohawk_hair_fin | 240 |
| Shoe_L | shoe | shoe | 144 |
| Shoe_R | shoe | shoe | 144 |
| Shoe_soles_L | shoe | shoe_soles | 44 |
| Shoe_soles_R | shoe | shoe_soles | 44 |
| Spiked_wrist_cuff_L | garment | spiked_wrist_cuff | 32 |
| Vest | garment | vest | 338 |

Renders in the run folder: judge/turntable_{clay,wire,flat}.png, judge/head_{clay,flat}.png, judge/cmp/compare_{front,side,back}.png
