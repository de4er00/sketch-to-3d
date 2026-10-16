"""Rig the textured character: armature, skin weights, FBX and two test poses.

    blender -b --factory-startup --python sketch3d/finish/rig.py -- <char.json> <body.json> <char_tex.blend> <outdir>
"""
import bpy, os, sys, json, math
from mathutils import Vector


def ring_centre(me, ring):
    return sum((me.vertices[i].co for i in ring), Vector()) / len(ring)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    char_path, map_path, in_blend, outdir = [os.path.abspath(a) for a in argv[:4]]
    C = json.load(open(char_path))
    M = json.load(open(map_path))
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=in_blend)
    scene = bpy.context.scene
    body = bpy.data.objects['Body']
    me = body.data
    lm = C['landmarks']
    meshes = [o for o in scene.objects if o.type == 'MESH']
    rep = []
    trunk = M['trunk']
    tz = [ring_centre(me, r) for r in trunk]

    def trunk_at(z):
        return min(tz, key=lambda c: abs(c.z - z))
    hips = trunk_at(lm['hip']['z'])
    hips = Vector((0.0, hips.y, lm['hip']['z']))
    waist = trunk_at(lm['waist']['z'])
    chest_z = lm['armpit']['z'] + 0.35 * (lm['shoulder']['z'] - lm['armpit']['z'])
    chest = trunk_at(chest_z)
    upper = trunk_at(lm['shoulder']['z'] - 0.01)
    neck_z = lm['neck']['z']
    neck = trunk_at(neck_z)
    chin_z = C['skull']['chin_z']
    head_top = M['z']['dome_top']
    J = dict(Hips=Vector((0, hips.y, hips.z)), Spine=Vector((0, waist.y, waist.z)), Chest=Vector((0, chest.y, chest.z)),
             UpperChest=Vector((0, upper.y, upper.z)), Neck=Vector((0, neck.y, neck_z)), Head=Vector((0, neck.y, chin_z + 0.005)),
             HeadTop=Vector((0, neck.y - 0.01, head_top)))
    sides = {}
    for side, S in (('L', 'Left'), ('R', 'Right')):
        arm = M['arm'][side]
        ac = [ring_centre(me, r) for r in arm]
        A = C['stations']['arm_' + side]
        s_root = A['stations'][0]['s_m']
        s_elbow = A['landmarks']['elbow']['s_px'] / C['calibration']['ppm']
        dist = [0.0]
        for a, b in zip(ac, ac[1:]):
            dist.append(dist[-1] + (b - a).length)
        i_e = min(range(len(ac)), key=lambda i: abs(dist[i] - (s_elbow - s_root - 0.035 + (dist[1] if len(dist) > 1 else 0))))
        i_e = max(1, min(len(ac) - 2, i_e))
        hand = M['hand'][side]
        hc = ring_centre(me, hand)
        # hand tip: the body vertex farthest from the wrist along the arm
        d = (ac[-1] - ac[-2]).normalized()
        tips = [v.co for v in me.vertices if (v.co - hc).dot(d) > 0.02 and abs((v.co - hc).cross(d).length) < 0.06]
        tip = max(tips, key=lambda p: (p - hc).dot(d)) if tips else hc + d * 0.1
        leg = M['leg'][side]
        lc = [ring_centre(me, r) for r in leg]
        knee_z = lm['knee_' + side]['z']
        ankle_z = lm['ankle_' + side]['z']
        knee = min(lc, key=lambda c: abs(c.z - knee_z))
        ankle = lc[-1]
        foot = M['foot'][side]
        sole = ring_centre(me, foot[1])
        toe = max((me.vertices[i].co for i in foot[1]), key=lambda p: -p.y)
        sides[S] = dict(shoulder_root=Vector((0.3 * ac[0].x, ac[0].y, ac[0].z + 0.01)), shoulder=ac[0], elbow=ac[i_e], wrist=hc, handtip=tip,
                        hip=Vector((lc[0].x, lc[0].y, hips.z)), knee=Vector((knee.x, knee.y, knee_z)), ankle=Vector((ankle.x, ankle.y, ankle_z)),
                        ball=Vector((sole.x, 0.5 * (sole.y + toe.y), sole.z + 0.01)), toetip=Vector((toe.x, toe.y, sole.z + 0.005)))
    arm_data = bpy.data.armatures.new('Rig')
    rig = bpy.data.objects.new('Rig', arm_data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = head, tail
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b
    bone('Root', Vector((0, 0, 0)), Vector((0, -0.1, 0)))
    chain = ['Hips', 'Spine', 'Chest', 'UpperChest', 'Neck', 'Head', 'HeadTop']
    prev = 'Root'
    for a, b in zip(chain, chain[1:]):
        bone(a, J[a], J[b], parent=prev, connect=(prev != 'Root' and prev != 'Hips'))
        prev = a
    for S, P in sides.items():
        bone(S + 'Shoulder', P['shoulder_root'], P['shoulder'], parent='UpperChest')
        bone(S + 'UpperArm', P['shoulder'], P['elbow'], parent=S + 'Shoulder', connect=True)
        bone(S + 'LowerArm', P['elbow'], P['wrist'], parent=S + 'UpperArm', connect=True)
        bone(S + 'Hand', P['wrist'], P['handtip'], parent=S + 'LowerArm', connect=True)
        bone(S + 'UpperLeg', P['hip'], P['knee'], parent='Hips')
        bone(S + 'LowerLeg', P['knee'], P['ankle'], parent=S + 'UpperLeg', connect=True)
        bone(S + 'Foot', P['ankle'], P['ball'], parent=S + 'LowerLeg', connect=True)
        bone(S + 'Toes', P['ball'], P['toetip'], parent=S + 'Foot', connect=True)
    spikes = [o for o in meshes if o.name.startswith('Hair_Spike')]
    for o in spikes:
        vs = [o.matrix_world @ v.co for v in o.data.vertices]
        head_c = J['Head']
        base = min(vs, key=lambda p: (p - head_c).length)
        tipv = max(vs, key=lambda p: (p - head_c).length)
        bone('B_' + o.name, base, tipv, parent='Head')
    bpy.ops.object.mode_set(mode='OBJECT')
    body_bones = [b.name for b in arm_data.bones if not b.name.startswith('B_Hair') and b.name != 'Root']
    for b in arm_data.bones:
        b.use_deform = True
    rep.append('armature Rig: %d bones (%d body, %d hair)' % (len(arm_data.bones), len(body_bones) + 1, len(spikes)))
    def parent_to_rig(o):
        mod = o.modifiers.new('Armature', 'ARMATURE')
        mod.object = rig
        o.parent = rig
    # hair bones excluded, they would steal the scalp
    for b in arm_data.bones:
        b.use_deform = not b.name.startswith('B_Hair')
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    for b in arm_data.bones:
        b.use_deform = True
    # smooth the body weights so the shells inherit soft joints
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    try:
        bpy.ops.object.vertex_group_smooth(group_select_mode='ALL', factor=0.5, repeat=2)
    except Exception:
        pass
    bpy.ops.object.mode_set(mode='OBJECT')
    unweighted = [v.index for v in body.data.vertices if not v.groups]
    rep.append('body: bone heat, %d unweighted verts' % len(unweighted))

    def copy_from(o, src):
        for b in arm_data.bones:
            if b.name not in o.vertex_groups:
                o.vertex_groups.new(name=b.name)
        dt = o.modifiers.new('DT', 'DATA_TRANSFER')
        dt.object = src
        dt.use_vert_data = True
        dt.data_types_verts = {'VGROUP_WEIGHTS'}
        dt.vert_mapping = 'POLYINTERP_NEAREST'
        dt.layers_vgroup_select_src = 'ALL'
        dt.layers_vgroup_select_dst = 'NAME'
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier='DT')
        parent_to_rig(o)

    def rigid(o, bname):
        for g in list(o.vertex_groups):
            o.vertex_groups.remove(g)
        vg = o.vertex_groups.new(name=bname)
        vg.add([v.index for v in o.data.vertices], 1.0, 'REPLACE')
        parent_to_rig(o)
    jacket = bpy.data.objects.get('Jacket')
    for o in meshes:
        if o is body:
            continue
        part = o.get('s3d_part', '')
        name = o.name
        if part == 'garment':
            copy_from(o, body)
            rep.append('%-14s <- body weights (DataTransfer)' % name)
        elif part == 'shoe':
            S = 'Left' if sum(v.co.x for v in o.data.vertices) > 0 else 'Right'
            rigid(o, S + 'Foot')
            rep.append('%-14s -> %sFoot (rigid)' % (name, S))
        elif name.startswith('Hair_Spike'):
            rigid(o, 'B_' + name)
            rep.append('%-14s -> own bone' % name)
        elif part == 'hair':
            rigid(o, 'Head')
            rep.append('%-14s -> Head (rigid)' % name)
        elif part == 'accessory':
            if name.startswith('Stud') and jacket is not None:
                copy_from(o, jacket)
                rep.append('%-14s <- jacket weights' % name)
            else:
                c = sum((v.co for v in o.data.vertices), Vector()) / len(o.data.vertices)
                bn = min([b for b in arm_data.bones if b.name in ('Neck', 'UpperChest', 'Chest', 'Head')], key=lambda b: (0.5 * (b.head_local + b.tail_local) - c).length).name
                rigid(o, bn)
                rep.append('%-14s -> %s (rigid)' % (name, bn))
        else:
            copy_from(o, body)
            rep.append('%-14s <- body weights (default)' % name)
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
            bpy.ops.object.vertex_group_clean(group_select_mode='ALL', limit=0.01)
            bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=4)
            bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=False)
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as e:
            rep.append('finish chain failed on %s: %s' % (o.name, e))
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
    rig_blend = os.path.join(outdir, 'rig.blend')
    bpy.ops.wm.save_as_mainfile(filepath=rig_blend)
    fbx = os.path.join(outdir, 'char.fbx')
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    for o in meshes:
        o.select_set(True)
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, apply_scale_options='FBX_SCALE_ALL', object_types={'ARMATURE', 'MESH'},
                             mesh_smooth_type='FACE', add_leaf_bones=False, use_armature_deform_only=True,
                             primary_bone_axis='Y', secondary_bone_axis='X', bake_anim=False, axis_forward='-Z', axis_up='Y',
                             path_mode='COPY', embed_textures=False)
    rep.append('fbx: %s (%d bytes)' % (fbx, os.path.getsize(fbx)))
    poses_dir = os.path.join(outdir, 'poses')
    os.makedirs(poses_dir, exist_ok=True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='POSE')
    pb = rig.pose.bones

    def rot(name, x=0, y=0, z=0):
        if name in pb:
            pb[name].rotation_mode = 'XYZ'
            pb[name].rotation_euler = (math.radians(x), math.radians(y), math.radians(z))
    poses = {
        'bind': {},
        'walk': dict(LeftUpperLeg=(-25, 0, 0), RightUpperLeg=(25, 0, 0), LeftLowerLeg=(20, 0, 0), RightLowerLeg=(40, 0, 0),
                     LeftUpperArm=(0, 0, 40), RightUpperArm=(0, 0, -40), LeftLowerArm=(0, -30, 0), RightLowerArm=(0, 30, 0),
                     Spine=(0, 0, 5), Head=(0, 0, -8)),
        'turn': dict(Spine=(0, 0, 25), Chest=(0, 0, 10), Head=(0, 0, 20), LeftUpperArm=(0, 0, 35), RightUpperArm=(0, 0, -35), RightLowerArm=(0, 60, 0)),
    }
    cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1.6
    scene.render.resolution_x, scene.render.resolution_y = 600, 800
    scene.render.engine = 'BLENDER_EEVEE'
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    scene.render.film_transparent = False
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)
    for pname, P in poses.items():
        for b in pb:
            b.rotation_euler = (0, 0, 0)
        for bn, r in P.items():
            rot(bn, *r)
        for yaw, tag in ((0, 'front'), (35, 'tq')):
            a = math.radians(yaw)
            cam.location = Vector((math.sin(a) * 4, -math.cos(a) * 4, 0.62))
            cam.rotation_euler = (math.radians(88), 0, a)
            scene.render.filepath = os.path.join(poses_dir, '%s_%s.png' % (pname, tag))
            bpy.ops.render.render(write_still=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    open(os.path.join(outdir, 'RIG_REPORT.txt'), 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
    print('\n'.join('[rig] ' + r for r in rep))


if __name__ == '__main__':
    main()
