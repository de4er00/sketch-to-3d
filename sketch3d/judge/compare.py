"""Render each sheet view with flat class colours at the sheet's px/m.

    blender -b --factory-startup --python sketch3d/judge/compare.py -- <char.json> <char.blend> <outdir>
"""
import bpy, os, sys, json, math
from mathutils import Vector

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, ROOT)


def render_view(scene, cam, look, up_right, centre, ortho, res, path):
    cam.location = centre + look * 5.0
    direction = (centre - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = ortho
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, blend, outdir = [os.path.abspath(a) for a in argv[:3]]
    C = json.load(open(char_path))
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    cal = C['calibration']
    ppm = cal['ppm']
    floor_row = cal['floor_row']
    pan = {p['name']: p for p in cal['panels']}
    meshes = [o for o in scene.objects if o.type == 'MESH']
    classes = sorted(set(o.get('s3d_class', 'skin') for o in meshes))
    palette = {}
    n = len(classes)
    L = int(math.ceil((n + 2) ** (1 / 3.0))) + 1
    k = 0
    for i in range(L):
        for j in range(L):
            for m in range(L):
                # white and black are the background
                if k < n and (i, j, m) != (L - 1, L - 1, L - 1) and (i, j, m) != (0, 0, 0):
                    palette[classes[k]] = (i / (L - 1), j / (L - 1), m / (L - 1))
                    k += 1
    for o in meshes:
        c = palette[o.get('s3d_class', 'skin')]
        o.color = (c[0], c[1], c[2], 1.0)
    scene.render.engine = 'BLENDER_WORKBENCH'
    sh = scene.display.shading
    sh.light = 'FLAT'
    sh.color_type = 'OBJECT'
    sh.show_shadows = False
    sh.show_cavity = False
    scene.display.render_aa = 'OFF'
    scene.render.film_transparent = True
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    world = scene.world or bpy.data.worlds.new('W')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    cam = bpy.data.objects.new('CmpCam', bpy.data.cameras.new('CmpCam'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    sheet = C['sheet']
    _im = bpy.data.images.load(sheet)
    SW, SH = _im.size[0], _im.size[1]
    bpy.data.images.remove(_im)
    H_m = SH / ppm
    z_c = (floor_row - SH / 2.0) / ppm
    views = {}
    looks = dict(front=Vector((0, -1, 0)), side=Vector((1, 0, 0)), back=Vector((0, 1, 0)))
    for name in [v for v in ('front', 'side', 'back') if v in pan]:  # a prop sheet has no back view
        look = looks[name]
        mid_col = C['datum']['side_col'] if name == 'side' else pan[name]['midline']
        c0, c1 = int(pan[name]['c0']), int(pan[name]['c1'])
        W_px = c1 - c0
        # the model axis is off the panel centre, shift the camera by the difference
        centre_col = 0.5 * (c0 + c1)
        off = (centre_col - float(mid_col)) / ppm
        if name == 'front':
            centre = Vector((off, 0.0, z_c))
        elif name == 'back':
            centre = Vector((-off, 0.0, z_c))  # image right = -X
        else:
            sgn = float(C['datum'].get('side_sign', 1.0))
            centre = Vector((0.0, sgn * off, z_c))
        path = os.path.join(outdir, 'cmp_render_%s.png' % name)
        # ortho_scale spans the longer side of the frame
        ortho = max(W_px, SH) / ppm
        render_view(scene, cam, look, None, centre, ortho, (W_px, SH), path)
        views[name] = dict(path=path, c0=c0, c1=c1, mid=mid_col, centre_col=centre_col, ortho=ortho, res=[W_px, SH])
    json.dump(dict(palette={k: list(v) for k, v in palette.items()}, views=views, sheet_size=[SW, SH], z_c=z_c, H_m=H_m),
              open(os.path.join(outdir, 'compare_views.json'), 'w'), indent=1)
    print('[compare] rendered %s' % ', '.join(views))


if __name__ == '__main__':
    main()
