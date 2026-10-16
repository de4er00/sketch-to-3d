"""Turntable renders of a .blend: clay, wire, flat, tex or lit.

    blender -b --factory-startup --python sketch3d/judge/render.py -- <file.blend> <out_prefix>
        [--modes clay,wire,flat,tex,lit] [--yaws 0,45,90] [--res 560x760] [--head] [--pitch 0]
"""
import math
import os
import sys

import bpy
from mathutils import Vector

CLAY = (0.78, 0.78, 0.76, 1.0)
WIRE = (0.12, 0.12, 0.14, 1.0)
SMOOTH_DEG = 45.0


def arg(argv, key, default):
    return argv[argv.index(key) + 1] if key in argv else default


def bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def setup_scene(scene, res):
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.display.render_aa = '16'
    scene.view_settings.view_transform = 'Standard'
    if scene.world is None:
        scene.world = bpy.data.worlds.new('World')


def set_mode(scene, mode, meshes, wires):
    sh = scene.display.shading
    sh.show_cavity = mode in ('clay', 'wire', 'tex')
    sh.cavity_type = 'BOTH'
    sh.curvature_ridge_factor = 0.6
    sh.curvature_valley_factor = 0.8
    sh.show_shadows = False
    sh.show_specular_highlight = mode in ('clay', 'wire')
    sh.light = 'FLAT' if mode == 'flat' else 'STUDIO'
    if mode in ('clay', 'wire'):
        sh.color_type = 'OBJECT'
        for o in meshes:
            o.color = CLAY
    else:
        sh.color_type = 'TEXTURE'
    for w in wires:
        w.hide_render = mode != 'wire'


def lit_setup(scene, meshes, samples):
    """Set up EEVEE with base-colour materials and a grey world, return the three lights."""
    for mat in {m for o in meshes for m in o.data.materials if m is not None and m.use_nodes}:
        nt = mat.node_tree
        tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE'), None)
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if tex is None or out is None:
            continue
        for n in [n for n in nt.nodes if n not in (tex, out)]:
            nt.nodes.remove(n)
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Roughness'].default_value = 0.75
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    for o in meshes:
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(SMOOTH_DEG))
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_shadows = True
    scene.eevee.use_raytracing = True
    world = scene.world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.86, 0.85, 0.83, 1.0)
    world.node_tree.nodes['Background'].inputs[1].default_value = 0.55
    lights = []
    for name, energy, size, colour in (('Key', 260, 2.5, (1.0, 0.97, 0.92)), ('Fill', 60, 3.0, (0.9, 0.95, 1.0)),
                                       ('Rim', 220, 1.5, (1.0, 1.0, 1.0))):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.size, data.color = energy, size, colour
        ob = bpy.data.objects.new(name, data)
        scene.collection.objects.link(ob)
        lights.append(ob)
    return lights


# camera space (right, back toward the camera, up), metres
LIGHT_RIG = ((-2.2, 3.0, 3.2), (3.0, 2.0, 1.4), (1.5, -3.0, 2.8))


def place_lights(lights, centre, right, toward):
    for ob, (a, b, c) in zip(lights, LIGHT_RIG):
        ob.location = centre + right * a + toward * b + Vector((0, 0, c))
        ob.rotation_euler = (centre - ob.location).to_track_quat('-Z', 'Y').to_euler()


def add_wires(scene, meshes):
    wires = []
    for o in meshes:
        w = o.copy()
        w.data = o.data
        w.name = o.name + '_wire'
        scene.collection.objects.link(w)
        mod = w.modifiers.new('wire', 'WIREFRAME')
        mod.thickness = 0.0016
        mod.use_replace = True
        mod.use_even_offset = True
        w.color = WIRE
        wires.append(w)
    return wires


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    blend, prefix = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    modes = arg(argv, '--modes', 'clay,wire,flat').split(',')
    yaws = [float(y) for y in arg(argv, '--yaws', '0,30,60,90,135,180,225,270,300,330').split(',')]
    res = tuple(int(v) for v in arg(argv, '--res', '560x760').split('x'))
    pitch = math.radians(float(arg(argv, '--pitch', '0')))
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type in ('ARMATURE', 'LIGHT', 'CAMERA', 'EMPTY'):
            o.hide_render = True
    meshes = [o for o in scene.objects if o.type == 'MESH' and not o.hide_render]
    setup_scene(scene, res)
    lo, hi = bounds(meshes)
    centre = (lo + hi) / 2.0
    height = hi.z - lo.z
    span = max(hi.x - lo.x, hi.y - lo.y)
    if '--head' in argv:
        # head close-up: square frame around the skull
        centre = Vector((0.0, (lo.y + hi.y) / 2.0, hi.z - 0.16 * height))
        scale = 0.36 * height
    else:
        aspect = res[0] / float(res[1])
        span = math.hypot(hi.x - lo.x, hi.y - lo.y)
        scale = 1.06 * (max(span, height * aspect) if aspect >= 1.0 else max(height, span / aspect))
    cam = bpy.data.objects.new('TurnCam', bpy.data.cameras.new('TurnCam'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale
    cam.data.clip_end = 100.0
    wires = add_wires(scene, meshes) if 'wire' in modes else []
    # lit goes last, it replaces the materials and the engine
    for mode in sorted(modes, key=lambda m: m == 'lit'):
        lights = []
        if mode == 'lit':
            lights = lit_setup(scene, meshes, int(arg(argv, '--samples', '64')))
            for w in wires:
                w.hide_render = True
            cam.data.type = 'PERSP'
            cam.data.lens = 85.0
        else:
            set_mode(scene, mode, meshes, wires)
        for yaw in yaws:
            a = math.radians(yaw)
            d = Vector((-math.sin(a) * math.cos(pitch), -math.cos(a) * math.cos(pitch), math.sin(pitch)))
            if mode == 'lit':
                dist = scale / (2.0 * math.tan(cam.data.angle / 2.0))
                cam.location = centre + d * dist
                place_lights(lights, centre, Vector((math.cos(a), -math.sin(a), 0.0)), Vector((d.x, d.y, 0.0)).normalized())
            else:
                cam.location = centre + d * 10.0
            cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = '%s_%s_%03d.png' % (prefix, mode, int(round(yaw)) % 360)
            bpy.ops.render.render(write_still=True)
    print('[render] %d modes x %d yaws -> %s_*' % (len(modes), len(yaws), prefix))


if __name__ == '__main__':
    main()
