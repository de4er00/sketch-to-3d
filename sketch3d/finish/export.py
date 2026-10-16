"""Export the textured character as GLB with the atlas and STL as one mesh, Y up.

    blender -b --factory-startup --python sketch3d/finish/export.py -- <char_tex.blend> <out_prefix>
"""
import math
import os
import sys

import bpy


def lit_materials(meshes):
    """Swap the unlit emission materials for Principled base colour, which viewers expect."""
    done = set()
    for o in meshes:
        for mat in o.data.materials:
            if mat is None or mat.name in done or not mat.use_nodes:
                continue
            done.add(mat.name)
            nt = mat.node_tree
            tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE'), None)
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if tex is None or out is None:
                continue
            for n in [n for n in nt.nodes if n not in (tex, out)]:
                nt.nodes.remove(n)
            bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.inputs['Roughness'].default_value = 0.85
            nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
            nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    blend, prefix = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render]
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    lit_materials(meshes)
    for o in meshes:
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(45.0))

    bpy.ops.export_scene.gltf(filepath=prefix + '.glb', export_format='GLB', use_selection=True,
                              export_apply=True, export_yup=True, export_animations=False)
    bpy.ops.wm.stl_export(filepath=prefix + '.stl', export_selected_objects=True, apply_modifiers=True,
                          forward_axis='NEGATIVE_Z', up_axis='Y', ascii_format=False)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    for ext in ('.glb', '.stl'):
        print('[export] %s%s %d bytes' % (os.path.basename(prefix), ext, os.path.getsize(prefix + ext)))
    print('[export] %d objects, %d triangles' % (len(meshes), tris))


if __name__ == '__main__':
    main()
