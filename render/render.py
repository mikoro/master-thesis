# coding=utf-8

"""Blender rendering automation script."""

import bpy
import os
import random
import string
import signal
import sys
import math
import time
import pickle

from enum import Enum


class TextureType(Enum):
    """The texture type of the head and the background/world."""
    Gray = 1
    Noise = 2
    Texture = 3

# Basic settings
# --------------------------------------------------------------------------------

if os.name == "nt":
    output_path = "D:/Thesis/renders"
    textures_directory = "D:/Thesis/textures"
elif os.name == "posix":
    tmp_dir, wrk_dir = os.environ["TMPDIR"], os.environ["WRKDIR"]
    assert len(tmp_dir) > 0 and len(wrk_dir) > 0
    output_path = tmp_dir + "/output"
    textures_directory = tmp_dir + "/textures"
else:
    raise RuntimeError("Unknown OS")

world_texture = TextureType.Texture
head_texture = TextureType.Texture
environment_textures_path = os.path.join(textures_directory, "environments")
material_textures_path = os.path.join(textures_directory, "materials")
face_textures_path = os.path.join(textures_directory, "faces")
misc_textures_path = os.path.join(textures_directory, "misc")
abs_script_path = os.path.dirname(os.path.abspath(__file__))
head_expressions_data_path = os.path.normpath(os.path.join(abs_script_path, "../models/head_expressions.dat"))
max_process_render_count = 200  # max renders per process (not important on taito, should just be large enough)
max_node_render_count = 82500  # max renders on a single node (by multiple processes, use this + node count to adjust total render count)
max_rendering_time = 60 * 60 * 14  # max rendering time in seconds per process
write_exr = True
write_png = True

# --------------------------------------------------------------------------------

sys.stdout = sys.stderr
ru = random.uniform
rg = random.gauss
rad = math.radians

bpy.context.scene.node_tree.nodes["File Output EXR"].base_path = output_path
bpy.context.scene.node_tree.nodes["File Output PNG"].base_path = output_path

environment_texture_paths = [os.path.join(environment_textures_path, i) for i in os.listdir(environment_textures_path)]
environment_texture_paths = [i for i in environment_texture_paths if os.path.isfile(i) and (i.endswith(".hdr") or i.endswith(".exr"))]

material_texture_paths = [os.path.join(material_textures_path, i) for i in os.listdir(material_textures_path)]
material_texture_paths = [i for i in material_texture_paths if os.path.isfile(i) and i.endswith(".jpg")]

face_texture_paths = [os.path.join(face_textures_path, i) for i in os.listdir(face_textures_path)]
face_texture_paths = [i for i in face_texture_paths if os.path.isfile(i) and i.endswith(".png")]

head_expressions_data = pickle.load(open(head_expressions_data_path, "rb"))
assert head_expressions_data.shape[1] == len(bpy.data.objects["Head"].data.vertices)

interrupted = False


def handler(signum, frame):
    """Handler for the interrupt signal."""
    del signum, frame
    global interrupted
    interrupted = True
    print("Rendering interrupted!")


signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


def set_default_texture_paths():
    """Set texture paths to something valid for current platform."""
    bpy.data.images["environment"].filepath = environment_texture_paths[0]
    bpy.data.images["face_mask"].filepath = os.path.join(misc_textures_path, "face_mask.png")
    bpy.data.images["face_real"].filepath = os.path.join(face_textures_path, "1.png")
    bpy.data.images["material"].filepath = material_texture_paths[0]
    bpy.data.images["noise"].filepath = os.path.join(misc_textures_path, "noise.png")


def random_string(length):
    """Generate random string of the given length."""
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))


def get_world_texture_name(type_):
    """Get the actual blender world texture name given the texture type."""
    if type_ == TextureType.Gray:
        return "WorldGray"
    elif type_ == TextureType.Noise:
        return "WorldNoise"
    elif type_ == TextureType.Texture:
        return "WorldTexture"


def get_head_texture_name(type_):
    """Get the actual blender head texture name given the texture type."""
    if type_ == TextureType.Gray:
        return "HeadGray"
    elif type_ == TextureType.Noise:
        return "HeadNoise"
    elif type_ == TextureType.Texture:
        return "HeadTexture"


def randomize_head():
    """Randomize head parameters."""
    #bpy.data.objects["Head"].hide = True if ru(0.0, 1000.0) < 1.0 else False
    bpy.data.objects["Head"].scale = [ru(0.07, 0.13), ru(0.07, 0.13), ru(0.07, 0.13)]
    bpy.data.objects["Head"].rotation_euler = [rg(rad(90.0), rad(40.0) / 3.0), rg(rad(0.0), rad(45.0) / 3.0), rg(rad(0.0), rad(90.0) / 3.0)]
    bpy.data.objects["Head"].modifiers["SimpleDeformTwist"].angle = ru(rad(-10.0), rad(10.0))
    bpy.data.objects["Head"].modifiers["SimpleDeformBend"].angle = ru(rad(-10.0), rad(20.0))
    bpy.data.objects["Head"].modifiers["SimpleDeformTaper"].factor = ru(-0.15, 0.15)
    bpy.data.objects["Head"].modifiers["SimpleDeformStretch"].factor = ru(-0.03, 0.03)
    bpy.data.objects["Head"].modifiers["Displace"].strength = ru(-0.5, 0.5)  # this sometimes causes artifacts around eyes in uv/mask (triangle faces overlap)


def randomize_camera():
    """Randomize camera parameters."""
    bpy.data.cameras["Camera"].lens = ru(20.0, 70.0)
    bpy.data.cameras["Camera"].shift_x = rg(0.0, 0.6 / 3.0)
    bpy.data.cameras["Camera"].shift_y = rg(0.0, 0.6 / 3.0)


def randomize_head_and_camera_location():
    """This is mostly for seeding the displacement texture generation with different locations."""
    x = ru(-10.0, 10.0)
    y = ru(-10.0, 10.0)
    z = ru(-10.0, 10.0)

    bpy.data.objects["Head"].location[0] = x
    bpy.data.objects["Head"].location[1] = y
    bpy.data.objects["Head"].location[2] = z

    bpy.data.objects["Camera"].location[0] = x
    bpy.data.objects["Camera"].location[1] = y - 3.0
    bpy.data.objects["Camera"].location[2] = z + 0.2


def randomize_light():
    """Randomize light parameters."""
    bpy.data.objects["Light"].rotation_euler = [ru(rad(45.0), rad(135.0)), 0.0, ru(rad(65.0), rad(-65.0))]
    bpy.data.lamps["Light"].shadow_soft_size = ru(0.0, 0.2)
    bpy.data.lamps["Light"].node_tree.nodes["Emission"].inputs["Strength"].default_value = ru(1.0, 5.0)


def randomize_materials():
    """Randomize different material parameters."""
    # randomize noise textures
    scale = ru(0.1, 1.0)
    bpy.data.worlds["WorldNoise"].node_tree.nodes["Mapping"].translation = [ru(-10.0, 10.0), ru(-10.0, 10.0), ru(-10.0, 10.0)]
    bpy.data.worlds["WorldNoise"].node_tree.nodes["Mapping"].scale = [scale, scale, scale]
    scale = ru(0.1, 1.0)
    bpy.data.materials["HeadNoise"].node_tree.nodes["Mapping"].translation = [ru(-10.0, 10.0), ru(-10.0, 10.0), ru(-10.0, 10.0)]
    bpy.data.materials["HeadNoise"].node_tree.nodes["Mapping"].scale = [scale, scale, scale]

    # randomize world texture
    bpy.data.images["environment"].filepath = random.choice(environment_texture_paths)
    bpy.data.worlds["WorldTexture"].node_tree.nodes["Mapping"].scale[1] = ru(-2.0, -0.2) if ru(0.0, 1.0) < 0.5 else ru(0.2, 2.0)
    bpy.data.worlds["WorldTexture"].node_tree.nodes["Mapping"].rotation = [ru(rad(0.0), rad(360.0)), ru(rad(0.0), rad(360.0)), ru(rad(0.0), rad(360.0))]
    bpy.data.worlds["WorldTexture"].node_tree.nodes["Background"].inputs["Strength"].default_value = ru(0.1, 2.0)

    # randomize head texture
    scale = ru(0.1, 2.0)
    bpy.data.images["material"].filepath = random.choice(material_texture_paths)
    bpy.data.images["face_real"].filepath = random.choice(face_texture_paths)
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mapping Material"].translation = [ru(-100.0, 100.0), ru(-100.0, 100.0), ru(-100.0, 100.0)]
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mapping Material"].rotation = [ru(rad(0.0), rad(360.0)), ru(rad(0.0), rad(360.0)), ru(rad(0.0), rad(360.0))]
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mapping Material"].scale = [scale, scale, scale]
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mapping Real"].scale[0] = 1.0 if ru(0.0, 1.0) < 0.5 else -1.0
    bpy.data.materials["HeadTexture"].node_tree.nodes["RGB"].outputs[0].default_value = [ru(0.0, 1.0), ru(0.0, 1.0), ru(0.0, 1.0), 1.0]
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mix Material/Color"].inputs[0].default_value = ru(0.2, 0.8)
    #bpy.data.materials["HeadTexture"].node_tree.nodes["Mix Real/Material"].inputs[0].default_value = ru(0.0, 1.0)
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mix Real/Material"].inputs[0].default_value = 1.0  # only real face textures
    bpy.data.materials["HeadTexture"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = ru(0.3, 0.6)
    bpy.data.materials["HeadTexture"].node_tree.nodes["Mix Shader"].inputs[0].default_value = ru(0.0, 1.0)
    bpy.data.materials["HeadTexture"].node_tree.nodes["Gamma"].inputs[1].default_value = ru(0.8, 2.5)
    bpy.data.materials["HeadTexture"].node_tree.nodes["Hue Saturation Value"].inputs[1].default_value = ru(0.8, 1.1)


def randomize_cycles():
    """Randomize the cycles renderer seed."""
    bpy.data.scenes["Scene"].cycles.seed = random.randint(1, 10000000)


def set_head_expression(index):
    """Set the head model vertices to given expression."""
    for i in range(head_expressions_data.shape[1]):
        bpy.data.objects["Head"].data.vertices[i].co = head_expressions_data[index][i]

    bpy.data.objects["Head"].select = True
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")


def animate_through_head_expressions():
    """Loop through all expressions, display them while waiting a little between each."""
    for i in range(head_expressions_data.shape[0]):
        set_head_expression(i)
        print(i)
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        time.sleep(0.2)


def randomize_head_expression():
    """Set random head expression while preferring those with mouth closed."""
    all_expressions = list(range(head_expressions_data.shape[0]))
    mouth_closed = [53, 54, 55, 56, 79, 80, 93, 94, 121, 122, 123, 178, 179]
    mouth_open = [x for x in all_expressions if x not in mouth_closed]

    if ru(0.0, 1.0) < 0.5:
        index = random.choice(mouth_closed)
    else:
        index = random.choice(mouth_open)

    set_head_expression(index)


def randomize_all():
    """Randomize all parameters at one go."""
    randomize_head_expression()
    randomize_head()
    randomize_camera()
    randomize_head_and_camera_location()
    randomize_light()
    randomize_materials()
    randomize_cycles()


def do_rendering(random_file_name: bool = True, file_name_index: int = 0):
    """Render input, uv and mask images."""
    if random_file_name:
        id_ = random_string(16)
        bpy.context.scene.node_tree.nodes["File Output EXR"].file_slots[0].path = "head_{0}_#.exr".format(id_)
        bpy.context.scene.node_tree.nodes["File Output PNG"].file_slots[0].path = "head_{0}_#.png".format(id_)
    else:
        bpy.context.scene.node_tree.nodes["File Output EXR"].file_slots[0].path = "head_{0}_#.exr".format(file_name_index)
        bpy.context.scene.node_tree.nodes["File Output PNG"].file_slots[0].path = "head_{0}_#.png".format(file_name_index)

    # render shaded
    bpy.data.scenes["Scene"].frame_current = 1
    bpy.data.scenes["Scene"].cycles.samples = 8
    bpy.data.scenes["Scene"].cycles.filter_width = 1.0
    bpy.context.scene.world = bpy.data.worlds.get(get_world_texture_name(world_texture))
    bpy.data.objects["Head"].data.materials[0] = bpy.data.materials.get(get_head_texture_name(head_texture))
    bpy.ops.render.render()

    # render uv
    bpy.data.scenes["Scene"].frame_current = 2
    bpy.data.scenes["Scene"].cycles.samples = 1
    bpy.data.scenes["Scene"].cycles.filter_width = 0.01
    bpy.context.scene.world = bpy.data.worlds.get("WorldBlack")
    bpy.data.objects["Head"].data.materials[0] = bpy.data.materials.get("HeadUv")
    bpy.ops.render.render()

    # render mask
    bpy.data.scenes["Scene"].frame_current = 3
    bpy.data.scenes["Scene"].cycles.samples = 8
    bpy.data.scenes["Scene"].cycles.filter_width = 1.0
    bpy.context.scene.world = bpy.data.worlds.get("WorldBlack")
    bpy.data.objects["Head"].data.materials[0] = bpy.data.materials.get("HeadMask")
    bpy.ops.render.render()

    bpy.data.scenes["Scene"].frame_current = 1


def render_loop():
    """The actual main rendering loop. Do not run this function inside blender, only from command line (it deletes node connections)."""
    if not write_exr:
        bpy.context.scene.node_tree.links.remove(bpy.context.scene.node_tree.nodes["File Output EXR"].inputs[0].links[0])

    if not write_png:
        bpy.context.scene.node_tree.links.remove(bpy.context.scene.node_tree.nodes["File Output PNG"].inputs[0].links[0])

    set_default_texture_paths()
    start_time = time.time()

    for i in range(1, (max_process_render_count + 1)):
        print("Rendering frame {}".format(i))

        randomize_all()
        do_rendering()

        elapsed_seconds = time.time() - start_time

        if i % 100 == 0:
            m, s = divmod(elapsed_seconds, 60)
            h, m = divmod(m, 60)
            print("Time elapsed: {0:02.0f}:{1:02.0f}:{2:02.0f}".format(h, m, s))

        if len(os.listdir(output_path)) >= max_node_render_count + 2:
            print("Max rendering count reached!")
            break

        if elapsed_seconds >= max_rendering_time:
            print("Max rendering time reached!")
            break

        if interrupted:
            break


#set_default_texture_paths()
#set_head_expression(178)
#randomize_head_expression()
#randomize_head()
#randomize_head_and_camera_location()
#randomize_camera()
#randomize_light()
#randomize_materials()
#randomize_all()
#do_rendering(random_file_name=False, file_name_index=3)
#render_loop()
