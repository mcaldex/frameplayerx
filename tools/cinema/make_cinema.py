"""Builds FramePlayerX's cinema rooms in Blender.

    blender -b --factory-startup --python tools/cinema/make_cinema.py -- \\
        --room classic --out DIR --export

The rooms (ROOMS) share their seats, rising rows with lit steps and exits at
the back. --export bakes a room's lighting and writes the player's room files
into DIR; --preview renders views from the seats there instead.

Blender axes: x right, y towards the screen, z up; metres.
"""
import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

# Each room's size and layout; configure() makes one the module's layout.
# FIRST_ROW is the front edge of the first row's platform and FIRST_FLOOR its
# height: raised above the front of the room, so the front rows see the
# screen's middle without craning their necks. BLOCKS are the seats a row
# has between aisles. RISE is how much higher each row is than the one in
# front, or None for just enough to see the screen's bottom edge over it.
# EYE_HEIGHT and EYE_BACK place seated eyes above the row's floor and behind
# the cushion's front edge, LEGROOM is that edge from the platform's front.
ROOMS = {
    # A stadium auditorium: a 1.9:1 screen 12.6 m wide, 12 rows of 17 velvet
    # seats between side aisles, fabric wall panels between walnut fins,
    # wall lights and downlights.
    "classic": dict(SCREEN_W=12.6, SCREEN_ASPECT=1.9, SCREEN_BOTTOM=0.5, BLOCKS=(17,),
                    INNER_AISLE=0.0, ROWS=12, FIRST_ROW=-6.0, FIRST_FLOOR=0.9, RISE=None,
                    AISLE=1.3, CEILING=9.0, SEAT="chair", SEAT_PITCH=0.62, ROW_PITCH=1.15,
                    EYE_HEIGHT=1.22, EYE_BACK=0.28, LEGROOM=0.42, SCHEME="bordeaux"),
    # A premium multiplex kind of room: open, a 2.3:1 screen 10.9 m wide, 10
    # rows of leather recliners, 2, 3 and 2 to a row with wide aisles and
    # half a metre between them, rising just enough to see over each other,
    # dark walls and ceiling lined with neon, doors at the front. The eyes
    # are 27 cm in front of the recliner's backrest and 0.7 m above its
    # cushion.
    "luxe": dict(SCREEN_W=10.9, SCREEN_ASPECT=2.3, SCREEN_BOTTOM=0.5, BLOCKS=(2, 3, 2),
                 INNER_AISLE=1.6, ROWS=10, FIRST_ROW=-3.17, FIRST_FLOOR=0.2, RISE=None,
                 AISLE=1.0, CEILING=8.0, SEAT="recliner", SEAT_PITCH=1.4, ROW_PITCH=1.4,
                 EYE_HEIGHT=1.12, EYE_BACK=0.45, LEGROOM=0.4, SCHEME="slate"),
}
# How far the line from each row's eyes to the screen's bottom edge passes
# above the eyes of the row in front. The rows are empty, so it only has to
# clear the seats in front, which stay below the eyes.
CLEARANCE = 0.06
FRONT_WALL_Y = 0.6
STAGE_Y = -1.0
STAGE_TOP = 0.3
# The black border round the screen, and the widest strip the screen and it
# are cut into to bend.
MASKING = 0.4
STRIP = 0.25


def row_floors():
    """Each row's floor height: RISE above the row in front, or rising by
    whole 5 cm steps so its eyes see over that row to the screen's bottom
    edge."""
    if RISE is not None:
        return [FIRST_FLOOR + k * RISE for k in range(ROWS)]
    floors = [FIRST_FLOOR]
    for k in range(1, ROWS):
        y0, y1 = eye_y(k - 1), eye_y(k)
        needed = SCREEN_BOTTOM + (floors[-1] + EYE_HEIGHT + CLEARANCE - SCREEN_BOTTOM) * y1 / y0
        floor = max(floors[-1] + 0.15, needed - EYE_HEIGHT)
        floors.append(math.ceil(floor / 0.05 - 1e-6) * 0.05)
    return floors


def row_front(k):
    return FIRST_ROW - k * ROW_PITCH


def eye_y(k):
    return row_front(k) - LEGROOM - EYE_BACK


def configure(room):
    """Makes `room`'s size and layout the module's, with what follows from
    them: the seats' width (BLOCK), the side walls (WALL_X), the rows'
    floors and the back wall (BACK_Y)."""
    g = globals()
    g.update(ROOMS[room], ROOM=room)
    g["SCREEN_H"] = SCREEN_W / SCREEN_ASPECT
    g["SEATS"] = sum(BLOCKS)
    g["BLOCK"] = SEATS * SEAT_PITCH + (len(BLOCKS) - 1) * INNER_AISLE
    g["WALL_X"] = BLOCK / 2 + AISLE
    g["FLOORS"] = row_floors()
    g["BACK_Y"] = row_front(ROWS) - 1.6


configure("classic")


def floor_at(y):
    """The floor height at depth y: a row's, or the front's or the back's."""
    if y >= FIRST_ROW:
        return 0.0
    k = min(int((FIRST_ROW - y) / ROW_PITCH), ROWS - 1)
    return FLOORS[k]


# ---------------------------------------------------------------- materials


def material(name, color, rough=0.6, metallic=0.0, emission=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


def nodes_of(m):
    tree = m.node_tree
    return tree.nodes, tree.links, tree.nodes["Principled BSDF"]


def coords(nodes, scale):
    tc = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    nodes.id_data.links.new(tc.outputs["Object"], mapping.inputs["Vector"])
    return mapping.outputs["Vector"]


def ramp(nodes, links, fac, low, high, lo=0.0, hi=1.0):
    r = nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].position = lo
    r.color_ramp.elements[0].color = (*low, 1)
    r.color_ramp.elements[1].position = hi
    r.color_ramp.elements[1].color = (*high, 1)
    links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def fabric(name, color, sheen=0.3, rough=0.88):
    """Upholstery: one even colour with a velvet sheen in that colour. Its
    weave is finer than a baked texel, so it would only bake in as speckle."""
    m = material(name, color, rough)
    bsdf = nodes_of(m)[2]
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Sheen Roughness"].default_value = 0.35
    peak = max(color) or 1.0
    bsdf.inputs["Sheen Tint"].default_value = (*(0.25 + 0.75 * c / peak for c in color), 1)
    return m


def walnut(name, along):
    """Walnut veneer: straight streaks of grain along axis `along` (0 x,
    1 y, 2 z), as on a board cut along its length. Rings would wind round a
    long, thin strip like a rope."""
    m = material(name, (0.12, 0.06, 0.03), 0.45)
    nodes, links, bsdf = nodes_of(m)
    tc = nodes.new("ShaderNodeTexCoord")
    stretch = nodes.new("ShaderNodeMapping")
    scale = [40.0, 40.0, 40.0]
    scale[along] = 0.5
    stretch.inputs["Scale"].default_value = scale
    links.new(tc.outputs["Object"], stretch.inputs["Vector"])
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1.0
    grain.inputs["Detail"].default_value = 8.0
    grain.inputs["Roughness"].default_value = 0.65
    links.new(stretch.outputs["Vector"], grain.inputs["Vector"])
    links.new(ramp(nodes, links, grain.outputs["Fac"], (0.04, 0.02, 0.01),
                   (0.15, 0.08, 0.038), 0.3, 0.7), bsdf.inputs["Base Color"])
    bsdf.inputs["Coat Weight"].default_value = 0.35
    return m


def carpet(name, base=(0.028, 0.028, 0.031), flecks=(0.075, 0.06, 0.05)):
    """A dark carpet with a small scattered pattern of lighter flecks."""
    m = material(name, base, 0.95)
    nodes, links, bsdf = nodes_of(m)
    vec = coords(nodes, 1.0)
    cells = nodes.new("ShaderNodeTexVoronoi")
    cells.inputs["Scale"].default_value = 7.0
    cells.feature = "F1"
    links.new(vec, cells.inputs["Vector"])
    links.new(ramp(nodes, links, cells.outputs["Distance"], flecks, base, 0.035, 0.06),
              bsdf.inputs["Base Color"])
    return m


def build_materials(scheme):
    """The rooms' materials. Schemes: "bordeaux" and "graphite" velvet seats
    with walnut trim for the Classic, "slate" leather seats with graphite
    trim, a brown carpet and darker walls for the Luxe."""
    red = (0.085, 0.006, 0.011)
    if scheme == "slate":
        # Pale blue-grey leather round darker padded insets.
        leather = material("Seat leather", (0.38, 0.41, 0.47), 0.42)
        graphite = material("Graphite", (0.035, 0.037, 0.042), 0.4)
        seats = {"seat": leather, "head": leather, "inlay": graphite, "frame": graphite,
                 "carpet": carpet("Carpet", (0.03, 0.022, 0.017), (0.06, 0.045, 0.034)),
                 "wall": material("Wall", (0.026, 0.027, 0.031), 0.8),
                 "panel": material("Seam", (0.09, 0.095, 0.105), 0.5),
                 "speaker": fabric("Speaker grille", (0.02, 0.02, 0.022), 0.1, 0.9),
                 "neon": material("Neon", (0.0, 0.0, 0.0), 0.5, emission=(0.85, 0.92, 1.0),
                                  strength=10.0),
                 # By the screen: the neon goes out with the house lights, the
                 # exit sign dims with the room.
                 "door neon": material("Door neon", (0.0, 0.0, 0.0), 0.5,
                                       emission=(0.85, 0.92, 1.0), strength=6.0),
                 "front exit": material("Front exit sign", (0.0, 0.0, 0.0), 0.5,
                                        emission=(0.03, 0.9, 0.2), strength=0.5),
                 "front exit text": material("Front exit text", (0.0, 0.0, 0.0), 0.5,
                                             emission=(1.0, 1.0, 1.0), strength=3.0)}
    else:
        seat = red if scheme == "bordeaux" else (0.02, 0.02, 0.023)
        seats = {"seat": fabric("Seat fabric", seat), "head": fabric("Headrest fabric", red),
                 "inlay": walnut("Walnut inlay", 1), "frame": walnut("Walnut", 2),
                 "carpet": carpet("Carpet"),
                 "wall": fabric("Wall fabric", (0.028, 0.03, 0.036), 0.2, 0.95),
                 "panel": fabric("Panel fabric", (0.045, 0.047, 0.055), 0.3, 0.95)}
    mats = {
        **seats,
        "shell": material("Seat shell", (0.012, 0.012, 0.014), 0.55),
        "arm": material("Armrest leather", (0.012, 0.011, 0.011), 0.58),
        "metal": material("Satin black metal", (0.03, 0.03, 0.032), 0.35, 1.0),
        "cup": material("Cupholder", (0.012, 0.012, 0.013), 0.3),
        "riser": material("Riser", (0.012, 0.012, 0.013), 0.8),
        "door": material("Exit door", (0.032, 0.029, 0.026), 0.45),
        "bar": material("Push bar", (0.22, 0.22, 0.23), 0.35),
        "nosing": material("Step nosing", (0.6, 0.6, 0.62), 0.3, 1.0),
        "black": material("Black velvet", (0.006, 0.006, 0.007), 0.95),
        "ceiling": material("Ceiling", (0.012, 0.012, 0.014), 0.9),
        "baffle": fabric("Baffle", (0.03, 0.03, 0.033), 0.1, 0.95),
        "screen": material("Screen", (0.05, 0.05, 0.05), 0.9),
        # Black velvet of its own, so it bends with the screen.
        "masking": material("Masking", (0.006, 0.006, 0.007), 0.95),
        "glass": material("Booth glass", (0.01, 0.012, 0.015), 0.05),
        "led": material("Step light", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 0.55, 0.2),
                        strength=6.0),
        "warm": material("Warm light", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 0.72, 0.45),
                         strength=12.0),
        # A green panel dim enough that its white letters stand out.
        "exit": material("Exit sign", (0.0, 0.0, 0.0), 0.5, emission=(0.03, 0.9, 0.2),
                         strength=0.5),
        "exit text": material("Exit text", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 1.0, 1.0),
                              strength=3.0),
        # The cinema's name over the projection ports: lit, and dimmed with
        # the room.
        "sign text": material("Sign text", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 0.95, 0.9),
                              strength=1.5),
        # Below the screen, so it goes out with the house lights.
        "stage": material("Stage light", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 0.72, 0.45),
                          strength=12.0),
    }
    return mats


# ----------------------------------------------------------------- geometry


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def to_object(name, bm, mats, collection=None, smooth=False):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for m in mats:
        mesh.materials.append(m)
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    return link(bpy.data.objects.new(name, mesh), collection)


def cube(lo, hi, cuts=0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    if cuts:
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    for v in bm.verts:
        v.co = Vector(lo[i] + (v.co[i] + 0.5) * (hi[i] - lo[i]) for i in range(3))
    return bm


def box(name, lo, hi, mat, bevel=0.0, segments=2, collection=None):
    bm = cube(lo, hi)
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=segments,
                        profile=0.5, affect="EDGES", clamp_overlap=True)
    return to_object(name, bm, [mat], collection)


def apply_modifiers(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(graph))
    old = obj.data
    obj.modifiers.clear()
    obj.data = mesh
    bpy.data.meshes.remove(old)


def soft(name, lo, hi, mats, bevel, cuts=3, levels=1, deform=None, collection=None):
    """An upholstered part: a box with rounded edges, smoothed, then shaped
    by `deform` (a function of each vertex's position)."""
    obj = to_object(name, cube(lo, hi, cuts), mats, collection, smooth=True)
    b = obj.modifiers.new("Rounded edges", "BEVEL")
    b.width = bevel
    b.segments = 3
    b.limit_method = "ANGLE"
    b.harden_normals = False
    if levels:
        s = obj.modifiers.new("Smooth", "SUBSURF")
        s.levels = s.render_levels = levels
    apply_modifiers(obj)
    if deform:
        for v in obj.data.vertices:
            v.co = deform(v.co.copy())
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def transform(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()


def profile_slab(name, points, x0, x1, mat, bevel, collection=None):
    """A slab of the polygon `points` (y, z), from x0 to x1."""
    bm = bmesh.new()
    verts = [bm.verts.new((x0, y, z)) for y, z in points]
    face = bm.faces.new(verts)
    ext = bmesh.ops.extrude_face_region(bm, geom=[face])
    moved = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=moved, vec=(x1 - x0, 0, 0))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, profile=0.5,
                        affect="EDGES", clamp_overlap=True)
    return to_object(name, bm, [mat], collection)


def cylinder(name, centre, radius, depth, mat, segments=32, collection=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius,
                          radius2=radius, depth=depth)
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=centre)
    obj = to_object(name, bm, [mat], collection)
    for p in obj.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return obj


def join(objs, name):
    with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs):
        bpy.ops.object.join()
    objs[0].name = name
    objs[0].data.name = name
    return objs[0]


# -------------------------------------------------------------------- seats


def bell(t, width):
    return math.exp(-(t / width) ** 2)


def cushion_shape(v):
    # A crowned top and a rounded "waterfall" front edge.
    if v.z > 0.40:
        v.z += 0.016 * math.cos(math.pi * v.x / 0.52) * math.cos(math.pi * (v.y + 0.26) / 0.56)
    if v.y > -0.09 and v.z > 0.38:
        v.z -= 0.022 * ((v.y + 0.09) / 0.09) ** 2
    return v


def backrest_shape(v):
    """In the backrest's own frame: front face at y = 0, bottom at z = 0."""
    if v.y > -0.05:
        v.y += 0.022 * bell(v.z - 0.2, 0.11)  # lumbar
        # Three channels, stitched from the seat up to the headrest.
        fade = min(1.0, max(0.0, (v.z - 0.03) / 0.04)) * min(1.0, max(0.0, (0.47 - v.z) / 0.04))
        v.y -= 0.01 * fade * sum(bell(v.x - x, 0.009) for x in (-0.087, 0.087))
    v.y += 0.03 * (v.x / 0.26) ** 2  # the sides wrap forwards
    return v


def headrest_shape(v):
    v.y += 0.02 * (v.x / 0.235) ** 2
    return v


def shell_shape(v):
    v.y += 0.03 * (v.x / 0.275) ** 2
    if v.y < -0.15:
        v.y -= 0.025 * (1 - (v.x / 0.278) ** 2)  # a rounded back
    return v


def standard(mats, x, collection=None):
    """A side standard with a walnut inlay, its armrest and cupholder, at
    x: shared by the two seats either side."""
    t = 0.035
    outline = [(0.03, 0.0), (0.07, 0.36), (0.05, 0.56), (-0.6, 0.56), (-0.66, 0.3),
               (-0.62, 0.0)]
    parts = [profile_slab("Standard", outline, x - t, x + t, mats["metal"], 0.01, collection)]
    centre = Vector((sum(p[0] for p in outline) / 6, sum(p[1] for p in outline) / 6))
    inlay = [tuple(centre + (Vector(p) - centre) * 0.72) for p in outline]
    parts.append(profile_slab("Inlay", inlay, x - t - 0.004, x + t + 0.004, mats["inlay"],
                              0.004, collection))
    parts.append(soft("Armrest", (x - 0.05, -0.58, 0.555), (x + 0.05, 0.05, 0.625),
                      [mats["arm"]], 0.024, cuts=2, collection=collection))
    parts.append(cylinder("Cupholder", (x, 0.1, 0.585), 0.045, 0.05, mats["cup"],
                          collection=collection))
    parts.append(cylinder("Cup well", (x, 0.1, 0.6105), 0.036, 0.002, mats["black"],
                          collection=collection))
    return parts


def build_chair(mats, collection):
    """One seat facing +y, its cushion's front edge at the origin's y, with
    its left standard. The rightmost seat of a row also gets an end standard."""
    parts = []
    parts.append(soft("Cushion", (-0.255, -0.53, 0.31), (0.255, 0.0, 0.46), [mats["seat"]],
                      0.045, cuts=4, deform=cushion_shape, collection=collection))
    parts.append(box("Seat pan", (-0.24, -0.5, 0.25), (0.24, -0.06, 0.315), mats["shell"],
                     0.015, collection=collection))
    recline = Matrix.Translation((0, -0.5, 0.43)) @ Matrix.Rotation(math.radians(13), 4, "X")
    back = soft("Backrest", (-0.26, -0.13, 0.0), (0.26, 0.0, 0.64), [mats["seat"]], 0.04,
                cuts=16, deform=backrest_shape, collection=collection)
    transform(back, recline)
    parts.append(back)
    head = soft("Headrest", (-0.235, -0.05, 0.495), (0.235, 0.035, 0.66), [mats["head"]],
                0.035, cuts=4, deform=headrest_shape, collection=collection)
    transform(head, recline)
    parts.append(head)
    shell = soft("Back shell", (-0.278, -0.2, -0.03), (0.278, -0.095, 0.665), [mats["shell"]],
                 0.03, cuts=3, deform=shell_shape, collection=collection)
    transform(shell, recline)
    parts.append(shell)
    parts += standard(mats, -SEAT_PITCH / 2, collection)
    return join(parts, "Seat")


def build_end_standard(mats, collection):
    return join(standard(mats, -SEAT_PITCH / 2, collection), "End standard")


# The Luxe's recliner: "Recliner Brown_Leather" by Puneet Sharma
# (https://sketchfab.com/3d-models/recliner-brown-leather-4e3e06cf74fe48c8b7472a3549f6b5d2),
# CC-BY 4.0, downloaded as glTF; --seat-model names its scene.gltf.
SEAT_MODEL = None


def build_recliner(mats, collection):
    """The downloaded recliner as one mesh, at its real size: turned to face
    +y, its front edge at the origin's y, standing on z = 0."""
    if SEAT_MODEL is None:
        sys.exit("The Luxe's recliner needs --seat-model (see docs/DEVELOPMENT.md, Cinema rooms)")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(SEAT_MODEL))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in meshes:
        world = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = world
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o)
    chair = join(meshes, "Seat")
    chair.data.transform(chair.matrix_world)
    chair.matrix_world = Matrix.Identity(4)
    # It faces -y as modelled.
    chair.data.transform(Matrix.Rotation(math.pi, 4, "Z"))
    points = [v.co for v in chair.data.vertices]
    lo = Vector((min(p[i] for p in points) for i in range(3)))
    hi = Vector((max(p[i] for p in points) for i in range(3)))
    chair.data.transform(Matrix.Translation((-(lo.x + hi.x) / 2, -hi.y, -lo.z)))
    for m in chair.data.materials:
        m.name = "Recliner leather"
        nodes, links, bsdf = nodes_of(m)
        # Its brown is so dark that baked it reads as black.
        colour = bsdf.inputs["Base Color"].links
        if colour:
            lift = nodes.new("ShaderNodeVectorMath")
            lift.operation = "SCALE"
            lift.inputs["Scale"].default_value = 2.0
            links.new(colour[0].from_socket, lift.inputs[0])
            links.new(lift.outputs["Vector"], bsdf.inputs["Base Color"])
    for c in chair.users_collection:
        c.objects.unlink(chair)
    collection.objects.link(chair)
    return chair


# Each seat kind: its builder; the builder of the standard closing a block's
# right end (none where seats have their own armrests); how much of its detail
# the nearer and further copies keep; how far the nearer copy's bake looks for
# the full seat's surface (cage, ray; metres); whether that copy keeps the
# model's own texture layout rather than a new one; and whether its baked
# light keeps the shine, which leather needs.
SEAT_KINDS = {
    "chair": dict(build=build_chair, end=build_end_standard, near=0.1, far=0.3,
                  reach=(0.03, 0.08), own_layout=False, shine=False),
    "recliner": dict(build=build_recliner, end=None, near=0.05, far=0.15,
                     reach=(0.05, 0.12), own_layout=True, shine=True),
}


def seat_columns():
    """Each seat's x, left to right, and whether it ends its block."""
    x = -BLOCK / 2
    for n in BLOCKS:
        for i in range(n):
            yield x + (i + 0.5) * SEAT_PITCH, i == n - 1
        x += n * SEAT_PITCH + INNER_AISLE


def middle_column():
    """The seat on the screen's centre line."""
    columns = list(seat_columns())
    return min(range(SEATS), key=lambda i: abs(columns[i][0]))


def seat_positions():
    """(x, y, z) of each seat's origin, front row first, left to right."""
    for k in range(ROWS):
        for x, _ in seat_columns():
            yield x, row_front(k) - LEGROOM, FLOORS[k]


def end_positions():
    """Where the end standards go: right of each block's last seat."""
    for k in range(ROWS):
        for x, last in seat_columns():
            if last:
                yield x + SEAT_PITCH, row_front(k) - LEGROOM, FLOORS[k]


def place_seats(chair, end, collection):
    columns = list(seat_columns())
    for k in range(ROWS):
        y, z = row_front(k) - LEGROOM, FLOORS[k]
        for i, (x, last) in enumerate(columns):
            n = k * SEATS + i
            link(bpy.data.objects.new(f"Seat {n}", chair.data), collection).location = (x, y, z)
            if last and end:
                link(bpy.data.objects.new(f"End {n}", end.data), collection).location = (
                    x + SEAT_PITCH, y, z)


def aisles():
    """The aisles' x ranges: either side of the seats, and between blocks."""
    w = WALL_X - 0.02
    ranges, x = [(-w, -BLOCK / 2)], -BLOCK / 2
    for n in BLOCKS[:-1]:
        x += n * SEAT_PITCH
        ranges.append((x, x + INNER_AISLE))
        x += INNER_AISLE
    return ranges + [(BLOCK / 2, w)]


# --------------------------------------------------------------------- room


def build_floor_and_rows(mats, c):
    """Floor and stage, the rows' platforms with their risers, lit step edges
    in every aisle, and steps up the aisles to a raised first row."""
    w = WALL_X
    box("Floor", (-w, BACK_Y, -0.2), (w, FRONT_WALL_Y, 0.0), mats["carpet"], collection=c)
    box("Stage", (-w, STAGE_Y, 0.0), (w, FRONT_WALL_Y, STAGE_TOP), mats["black"], 0.01,
        collection=c)
    for k in range(ROWS + 1):
        y0 = row_front(k)
        y1 = row_front(k + 1) if k < ROWS else BACK_Y
        z = FLOORS[min(k, ROWS - 1)]
        if k == ROWS:
            z = FLOORS[-1] + 0.15
        if z <= 0:
            continue
        box(f"Row {k} riser", (-w, y0 - 0.02, 0.0), (w, y0, z), mats["riser"], collection=c)
        box(f"Row {k} floor", (-w, y1, 0.0), (w, y0 - 0.02, z), mats["carpet"], collection=c)
        for x0, x1 in aisles():
            box(f"Row {k} nosing", (x0, y0 - 0.05, z - 0.02), (x1, y0 + 0.004, z + 0.004),
                mats["nosing"], collection=c)
            box(f"Row {k} step light", (x0, y0 - 0.004, z - 0.05), (x1, y0 + 0.006, z - 0.035),
                mats["led"], collection=c)
    steps = round(FIRST_FLOOR / 0.18)
    if steps < 1:
        return
    rise, tread = FIRST_FLOOR / steps, 0.3
    for i in range(1, steps):
        y0, z = FIRST_ROW + (steps - i) * tread, i * rise
        for x0, x1 in aisles():
            box(f"Front step {i}", (x0, y0 - tread, 0.0), (x1, y0, z), mats["carpet"],
                collection=c)
            box(f"Front step {i} nosing", (x0, y0 - 0.05, z - 0.02), (x1, y0 + 0.004, z + 0.004),
                mats["nosing"], collection=c)
            box(f"Front step {i} light", (x0, y0 - 0.004, z - 0.05), (x1, y0 + 0.006, z - 0.035),
                mats["led"], collection=c)


def strips(name, lo, hi, mat, collection):
    """A box cut across x into strips no wider than STRIP, so the player can
    bend it with the screen, without its back, which faces the wall: the
    player draws both sides of every face, and from the back rows a face
    centimetres behind another shows through it."""
    bm = cube(lo, hi)
    n = max(1, math.ceil((hi[0] - lo[0]) / STRIP))
    for k in range(1, n):
        x = lo[0] + (hi[0] - lo[0]) * k / n
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                               plane_co=(x, 0.0, 0.0), plane_no=(1.0, 0.0, 0.0))
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > 0.5], context="FACES")
    return to_object(name, bm, [mat], collection)


def build_screen_wall(mats, c):
    """The black front wall, the screen and its masking, cut into strips to
    bend with the screen."""
    w = WALL_X
    box("Front wall", (-w, FRONT_WALL_Y, 0.0), (w, FRONT_WALL_Y + 0.2, CEILING), mats["black"],
        collection=c)
    sx, top, m = SCREEN_W / 2, SCREEN_BOTTOM + SCREEN_H, MASKING
    strips("Screen", (-sx, 0.0, SCREEN_BOTTOM), (sx, 0.02, top), mats["screen"], c)
    for lo, hi in (((-sx - m, -0.06, max(STAGE_TOP, SCREEN_BOTTOM - m)), (sx + m, 0.02,
                                                                           SCREEN_BOTTOM)),
                   ((-sx - m, -0.06, top), (sx + m, 0.02, top + m)),
                   ((-sx - m, -0.06, SCREEN_BOTTOM), (-sx, 0.02, top)),
                   ((sx, -0.06, SCREEN_BOTTOM), (sx + m, 0.02, top))):
        strips("Masking", lo, hi, mats["masking"], c)


def max_curvature():
    """The most the screen may curve, as 1 over its radius: its masking's
    outer edges then come forward to 10 cm short of the stage's front."""
    half, depth = SCREEN_W / 2 + MASKING, -STAGE_Y - 0.1
    lo, hi = half, 1e4
    for _ in range(60):
        r = (lo + hi) / 2
        if r * (1 - math.cos(half / r)) > depth:
            lo = r
        else:
            hi = r
    return 1 / hi


def build_back_exits(mats, c):
    """The exits, off the back platform under their signs; none by the
    screen to catch the eye."""
    for side in (-1, 1):
        x = side * WALL_X
        lo, hi = sorted((x - side * 0.04, x - side * 0.08))
        top_y, top_z = BACK_Y + 0.8, FLOORS[-1] + 0.15
        exit_door(mats, x, side, top_y, top_z, c)
        box("Exit sign", (lo, top_y - 0.2, top_z + 2.35), (hi, top_y + 0.2, top_z + 2.5),
            mats["exit"], 0.005, collection=c)
        lettering("EXIT", Vector((x - side * 0.082, top_y, top_z + 2.425)),
                  Vector((0, -side, 0)), Vector((-side, 0, 0)), 0.12, mats["exit text"], c)


def build_classic(mats, c):
    """The Classic: fabric wall panels between walnut fins with wall lights,
    a cove of light along each wall, acoustic baffles with downlights."""
    w = WALL_X
    build_floor_and_rows(mats, c)
    box("Stage light", (-w, STAGE_Y - 0.012, STAGE_TOP - 0.05),
        (w, STAGE_Y + 0.02, STAGE_TOP - 0.035), mats["stage"], collection=c)
    # Walls: black near the screen, fabric panels between walnut fins further back.
    for side in (-1, 1):
        x = side * w
        box("Side wall", (min(x, x + side * 0.2), BACK_Y, 0.0), (max(x, x + side * 0.2),
            FRONT_WALL_Y, CEILING), mats["wall"], collection=c)
        box("Front black", (x - side * 0.01 if side > 0 else x, -4.2, 0.0),
            (x if side > 0 else x + 0.01, FRONT_WALL_Y, CEILING), mats["black"], collection=c)
        y = -4.4
        while y - 2.2 > BACK_Y + 0.4:
            base = floor_at(y - 1.1) + 0.85
            inner = x - side * 0.05
            lo, hi = sorted((x, inner))
            box("Wall panel", (lo, y - 2.12, base), (hi, y - 0.08, 8.3), mats["panel"], 0.02,
                collection=c)
            lo, hi = sorted((x, x - side * 0.08))
            box("Walnut fin", (lo, y - 0.07, floor_at(y) + 0.1), (hi, y, 8.6), mats["frame"],
                0.008, collection=c)
            sconce = Vector((x - side * 0.07, y - 1.1, min(base + 1.9, 7.6)))
            lo, hi = sorted((x - side * 0.05, x - side * 0.085))
            box("Wall light", (lo, sconce.y - 0.035, sconce.z - 0.25), (hi, sconce.y + 0.035,
                sconce.z + 0.25), mats["warm"], 0.006, collection=c)
            light_area(f"Wall wash {side} {y:.1f}", sconce + Vector((-side * 0.12, 0, 0.3)),
                       sconce + Vector((side * 0.3, 0, 2.5)), 22.0, 0.18)
            light_area(f"Wall wash down {side} {y:.1f}", sconce + Vector((-side * 0.12, 0, -0.3)),
                       sconce + Vector((side * 0.3, 0, -2.0)), 10.0, 0.18)
            y -= 2.2
        lo, hi = sorted((x - side * 0.12, x - side * 0.15))
        box("Cove light", (lo, BACK_Y + 0.2, 8.62), (hi, -4.2, 8.66), mats["warm"], collection=c)
        light_area(f"Cove {side}", Vector((x - side * 0.4, (BACK_Y - 4.2) / 2, 8.55)),
                   Vector((x - side * 3.0, (BACK_Y - 4.2) / 2, 7.0)), 260.0, 0.2,
                   size_y=abs(BACK_Y + 4.2), rotate_z=True)
    build_screen_wall(mats, c)
    # Ceiling with acoustic baffles and downlights.
    box("Ceiling", (-w, BACK_Y, CEILING), (w, FRONT_WALL_Y, CEILING + 0.1), mats["ceiling"],
        collection=c)
    y = -3.0
    while y > BACK_Y + 0.8:
        box("Baffle", (-w + 0.3, y - 0.05, 8.25), (w - 0.3, y + 0.05, CEILING), mats["baffle"],
            0.01, collection=c)
        if y < -5.0:
            for x in (-4.2, -1.4, 1.4, 4.2):
                at = Vector((x, y - 0.8, CEILING - 0.01))
                cylinder("Downlight", at, 0.07, 0.01, mats["warm"], 24, c)
                light_area(f"Downlight {x} {y:.1f}", at - Vector((0, 0, 0.02)),
                           at - Vector((0, 0, 5)), 40.0, 0.14, shape="DISK", spread=70)
        y -= 1.6
    # Back wall with the projection ports, and the exits.
    box("Back wall", (-w, BACK_Y - 0.2, 0.0), (w, BACK_Y, CEILING), mats["wall"], collection=c)
    port = max(7.3, FLOORS[-1] + 2.3)
    for x in (-1.2, 0.0, 1.2):
        box("Projection port", (x - 0.35, BACK_Y - 0.02, port), (x + 0.35, BACK_Y + 0.01,
                                                                port + 0.45),
            mats["glass"], 0.01, collection=c)
    build_back_exits(mats, c)


def build_luxe(mats, c):
    """The Luxe, a multiplex kind of room: open and dark, its rows a low
    step apart, the side walls crossed by diagonal seams into big diamonds
    with bars of neon along them, the ceiling's edges sloping down to the
    walls lined with neon, double doors with neon chevrons at the front
    clear of the screen, and the cinema's name over the booth at the back."""
    w, top, edge = WALL_X, CEILING, 1.6
    build_floor_and_rows(mats, c)
    build_screen_wall(mats, c)
    box("Ceiling", (-w, BACK_Y, top), (w, FRONT_WALL_Y, top + 0.1), mats["wall"], collection=c)
    # Soft light over the seats, standing in for what the room's light bounces.
    mid = (row_front(0) + row_front(ROWS)) / 2
    light_area("Fill", Vector((0.0, mid, top - 0.3)), Vector((0.0, mid, 0.0)), 500.0,
               2 * (w - 1.0), size_y=abs(row_front(ROWS) - row_front(0)), rotate_z=True)
    for side in (-1, 1):
        x = side * w
        box("Side wall", (min(x, x + side * 0.2), BACK_Y, 0.0), (max(x, x + side * 0.2),
            FRONT_WALL_Y, top), mats["wall"], collection=c)
        # The ceiling slopes down to the wall, neon along both its edges.
        inner = x - side * edge
        bm = bmesh.new()
        bm.faces.new([bm.verts.new(p) for p in ((x, BACK_Y, top - edge), (x, FRONT_WALL_Y, top - edge),
                                                (inner, FRONT_WALL_Y, top), (inner, BACK_Y, top))])
        to_object("Ceiling slope", bm, [mats["wall"]], c)
        for name, xs, z in (("Low neon", (x - side * 0.03, x - side * 0.07), top - edge - 0.04),
                            ("High neon", (inner - side * 0.02, inner + side * 0.02), top - 0.05)):
            lo, hi = sorted(xs)
            box(name, (lo, BACK_Y + 0.3, z), (hi, -1.2, z + 0.04), mats["neon"], collection=c)
        light_area(f"Slope {side}", Vector((x - side * 0.5, (BACK_Y - 0.9) / 2, top - edge)),
                   Vector((x - side * 3.0, (BACK_Y - 0.9) / 2, 0.0)), 300.0, 0.2,
                   size_y=abs(BACK_Y + 1.5), rotate_z=True)
        # Diagonal seams cross the wall into diamonds, a bar of neon near
        # the top of each, with a small light and a speaker where they meet.
        out = Vector((-side * 0.025, 0.0, 0.0))
        front, span = -3.2, -3.2 - (BACK_Y + 0.4)
        bays = max(1, round(span / 4.0))
        bay = span / bays
        high = top - edge - 0.25
        for k in range(bays):
            y = front - k * bay
            low = floor_at(y - bay / 2) + 1.0
            for (y0, z0), (y1, z1) in (((y, low), (y - bay, high)), ((y, high), (y - bay, low))):
                a, b = Vector((x, y0, z0)), Vector((x, y1, z1))
                strip("Seam", a, b, 0.05, out, mats["panel"], c)
                up, down = (b, a) if z1 > z0 else (a, b)
                strip("Wall neon", down.lerp(up, 0.62), down.lerp(up, 0.88), 0.035, out * 1.6,
                      mats["neon"], c)
            meet = Vector((x - side * 0.05, y - bay / 2, (low + high) / 2))
            lo, hi = sorted((x - side * 0.03, x - side * 0.08))
            box("Wall light", (lo, meet.y - 0.03, meet.z - 0.2), (hi, meet.y + 0.03, meet.z + 0.2),
                mats["neon"], 0.005, collection=c)
            light_area(f"Wall wash {side} {k}", meet + Vector((-side * 0.1, 0, 0.25)),
                       meet + Vector((side * 0.3, 0, 2.5)), 8.0, 0.12)
            if k:
                # Under the diamonds' upper tips, clear of the sloping ceiling.
                lo, hi = sorted((x - side * 0.02, x - side * 0.3))
                box("Speaker", (lo, y - 0.22, high - 0.85), (hi, y + 0.22, high - 0.2),
                    mats["speaker"], 0.02, collection=c)
        neon_doors(mats, x, side, -1.95, c)
    # The back wall, with the projection booth and the cinema's name.
    box("Back wall", (-w, BACK_Y - 0.2, 0.0), (w, BACK_Y, top), mats["wall"], collection=c)
    # Above the back row's heads, the name between it and the ceiling.
    booth = FLOORS[-1] + 0.15 + 1.75
    box("Booth", (-2.2, BACK_Y, booth), (2.2, BACK_Y + 0.6, booth + 0.9), mats["wall"], 0.02,
        collection=c)
    box("Booth neon", (-2.2, BACK_Y + 0.58, booth - 0.04), (2.2, BACK_Y + 0.62, booth),
        mats["neon"], collection=c)
    for x in (-1.2, 0.0, 1.2):
        box("Projection port", (x - 0.35, BACK_Y + 0.59, booth + 0.25),
            (x + 0.35, BACK_Y + 0.62, booth + 0.7), mats["glass"], 0.01, collection=c)
    lettering("Cinema", Vector((0.0, BACK_Y + 0.004, booth + 1.3)), Vector((-1, 0, 0)),
              Vector((0, 1, 0)), 0.5, mats["sign text"], c)


def neon_doors(mats, x, side, y, collection):
    """Double doors in the side wall at x, centred on y, a neon chevron on
    each leaf, under an exit sign: by the screen, so the neon goes out with
    the house lights and the sign dims with the room."""
    def off_wall(depth):
        return sorted((x, x - side * depth))

    c, top, half = collection, 2.3, 0.8
    lo, hi = off_wall(0.03)
    box("Exit door", (lo, y - half, 0.0), (hi, y + half, top), mats["door"], 0.004, collection=c)
    lo, hi = off_wall(0.05)
    for y0, y1, z0, z1 in ((y - half - 0.08, y - half, 0.0, top + 0.08),
                           (y + half, y + half + 0.08, 0.0, top + 0.08),
                           (y - half - 0.08, y + half + 0.08, top, top + 0.08),
                           (y - 0.008, y + 0.008, 0.0, top)):
        box("Door frame", (lo, y0, z0), (hi, y1, z1), mats["frame"], 0.004, collection=c)
    out = Vector((-side * 0.02, 0.0, 0.0))
    face = x - side * 0.03
    for leaf in (-1, 1):
        y0, y1 = sorted((y + leaf * 0.12, y + leaf * (half - 0.12)))
        tip = Vector((face, (y0 + y1) / 2, 1.0))
        for z_top in (2.0, 1.6):
            a, b = Vector((face, y0, z_top)), Vector((face, y1, z_top))
            point = tip + Vector((0, 0, z_top - 2.0))
            strip("Door chevron", a, point, 0.03, out, mats["door neon"], c)
            strip("Door chevron", point, b, 0.03, out, mats["door neon"], c)
    lo, hi = off_wall(0.08)
    box("Exit sign", (lo, y - 0.2, top + 0.2), (hi, y + 0.2, top + 0.35), mats["front exit"],
        0.005, collection=c)
    lettering("EXIT", Vector((x - side * 0.082, y, top + 0.275)), Vector((0, -side, 0)),
              Vector((-side, 0, 0)), 0.12, mats["front exit text"], c)


def strip(name, a, b, width, out, mat, collection):
    """A bar `width` across from point a to point b, standing off its surface
    by `out` (the surface's normal, as long as the bar is deep)."""
    along = b - a
    x, z = along.normalized(), out.normalized()
    turn = Matrix((x, z.cross(x), z)).transposed().to_4x4()
    bm = cube((0.0, -width / 2, 0.0), (along.length, width / 2, out.length))
    bmesh.ops.transform(bm, matrix=Matrix.Translation(a) @ turn, verts=bm.verts[:])
    return to_object(name, bm, [mat], collection)


BUILDERS = {"classic": build_classic, "luxe": build_luxe}


def facet_panel(x, side, y0, y1, z0, z1, fold, mat, collection):
    """A wall panel on the side wall at x, folded out into the room along a
    diagonal so its two halves catch the light differently: from the bottom
    back to the top front if `fold` is 1, the other diagonal if -1."""
    out = Vector((-side, 0.0, 0.0))
    corners = [Vector((x, y0, z0)), Vector((x, y1, z0)), Vector((x, y1, z1)),
               Vector((x, y0, z1))]
    ridge = (0, 2) if fold > 0 else (1, 3)
    bm = bmesh.new()
    back = [bm.verts.new(p) for p in corners]
    front = [bm.verts.new(p + out * (0.32 if i in ridge else 0.03))
             for i, p in enumerate(corners)]
    a, b = ridge
    for o in (i for i in range(4) if i not in ridge):
        bm.faces.new((front[a], front[o], front[b]))
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((back[i], back[j], front[j], front[i]))
    bm.faces.new(back[::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return to_object("Wall facet", bm, [mat], collection)


def lettering(text, at, across, facing, size, mat, collection):
    """`text` as flat letters centred on `at`, reading along `across` and
    facing `facing`. Letters are faces of their own, so they stay sharp
    however coarse the baked texture."""
    curve = bpy.data.curves.new(text, "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = curve.align_y = "CENTER"
    font = link(bpy.data.objects.new(text, curve), collection)
    # Letters lie in their x-y plane facing +z: turn them to face the room.
    turn = Matrix((across, facing.cross(across), facing)).transposed().to_4x4()
    font.matrix_world = Matrix.Translation(at) @ turn
    bpy.context.view_layer.update()
    mesh = bpy.data.meshes.new_from_object(font.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    mesh.transform(font.matrix_world)
    mesh.materials.clear()
    mesh.materials.append(mat)
    bpy.data.objects.remove(font)
    return link(bpy.data.objects.new(f"Letters {text}", mesh), collection)


def exit_door(mats, x, side, y, floor, collection):
    """An exit door in the side wall at x, centred on y: a dark leaf with a
    push bar in a frame, so it shows against the dark wall."""
    def off_wall(depth):
        return sorted((x, x - side * depth))

    top, half = floor + 2.2, 0.45
    lo, hi = off_wall(0.03)
    box("Exit door", (lo, y - half, floor), (hi, y + half, top), mats["door"], 0.004,
        collection=collection)
    lo, hi = off_wall(0.06)
    for y0, y1, z0, z1 in ((y - half - 0.08, y - half, floor, top + 0.08),
                           (y + half, y + half + 0.08, floor, top + 0.08),
                           (y - half - 0.08, y + half + 0.08, top, top + 0.08)):
        box("Door frame", (lo, y0, z0), (hi, y1, z1), mats["frame"], 0.006,
            collection=collection)
    lo, hi = off_wall(0.09)
    box("Push bar", (lo, y - half + 0.08, floor + 0.98), (hi, y + half - 0.08, floor + 1.04),
        mats["bar"], 0.012, collection=collection)


LIGHTS = []


def light_area(name, at, target, power, size, size_y=None, shape="SQUARE", spread=None,
               rotate_z=False):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.color = (1.0, 0.8, 0.6)
    if size_y is not None:
        data.shape = "RECTANGLE"
        data.size = size
        data.size_y = size_y
    else:
        data.shape = shape
        data.size = size
    if spread:
        data.spread = math.radians(spread)
    obj = link(bpy.data.objects.new(name, data))
    obj.location = at
    obj.rotation_euler = (target - at).to_track_quat("-Z", "Y" if not rotate_z else "X").to_euler()
    LIGHTS.append(obj)
    return obj


# ------------------------------------------------------------------ preview


def use_gpu(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except (RuntimeError, TypeError):
            continue
        for device in prefs.devices:
            device.use = device.type == backend
        if any(d.use for d in prefs.devices):
            scene.cycles.device = "GPU"
            return


def preview(out, frame, scheme, wanted):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    use_gpu(scene)
    scene.cycles.samples = 160
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.6
    scene.world = bpy.data.worlds.new("Dark")
    scene.world.color = (0, 0, 0)
    screen = bpy.data.objects["Screen"]
    film = bpy.data.materials.new("Film")
    film.use_nodes = True
    nodes, links, bsdf = nodes_of(film)
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(frame))
    tc = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    # The front face's generated coordinates are x and z.
    sep = nodes.new("ShaderNodeSeparateXYZ")
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(sep.outputs["Z"], comb.inputs["Y"])
    links.new(comb.outputs["Vector"], mapping.inputs["Vector"])
    # A 16:9 picture inside the wider screen: pillarboxed.
    fill = (16 / 9) / SCREEN_ASPECT
    mapping.inputs["Scale"].default_value = (1 / fill, 1, 1)
    mapping.inputs["Location"].default_value = (-(1 - fill) / 2 / fill, 0, 0)
    links.new(mapping.outputs["Vector"], image.inputs["Vector"])
    image.extension = "CLIP"
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    links.new(image.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.6
    screen.data.materials[0] = film
    cam = link(bpy.data.objects.new("Eye", bpy.data.cameras.new("Eye")))
    cam.data.lens_unit = "FOV"
    cam.data.angle = math.radians(92)
    cam.data.clip_start = 0.05
    scene.camera = cam
    row = 3
    eye = Vector((0.0, eye_y(row), FLOORS[row] + EYE_HEIGHT))
    centre = Vector((0.0, 0.0, SCREEN_BOTTOM + SCREEN_H / 2))
    # A level view with the lens shifted up, so the screen keeps its shape.
    level = Vector((0.0, centre.y, eye.z))
    shots = [
        ("seat", eye, level, True, 0.16),
        ("seat-dark", eye, level, False, 0.16),
        ("seats", Vector((WALL_X - 0.6, eye_y(6) + 0.9, FLOORS[6] + 1.7)),
         Vector((1.5, eye_y(5), FLOORS[5] + 0.5)), True, 0.0),
        ("room", Vector((WALL_X - 1.0, -2.6, 2.4)), Vector((-1.5, eye_y(6), FLOORS[6] + 1.0)),
         True, 0.0),
        ("down", eye, eye + Vector((0, 0.9, -1.0)), True, 0.0),
    ]
    for name, at, look, lit, shift in shots:
        if wanted and name not in wanted:
            continue
        cam.location = at
        cam.rotation_euler = (look - at).to_track_quat("-Z", "Y").to_euler()
        cam.data.shift_y = shift
        for light in LIGHTS:
            light.hide_render = not lit
        scene.render.filepath = str(out / f"{ROOM}-{scheme}-{name}.png")
        bpy.ops.render.render(write_still=True)
        print("Rendered", scene.render.filepath, flush=True)


# ------------------------------------------------------------------- export

# Materials that stay lit when the house lights go down, and those that go
# out with them.
GLOW = {"Step light", "Exit sign", "Exit text"}
OUT = {"Stage light", "Door neon"}
# Drawn plain white from their shape: letters are too fine for the baked
# texture to hold.
WHITE = {"Exit text", "Sign text", "Front exit text"}
# Bent with the screen when it curves.
BENDS = {"Screen", "Masking"}
# The room's origin in the player: the eyes of the middle seat of row 5.
ORIGIN_ROW = 4
# Seats nearer than this (metres) are drawn in full detail.
NEAR_DETAIL = 3.2
# The baked textures hold the light, about a centimetre a texel on the walls;
# the player lays a tiling pattern over them for what is finer (see
# detail_texture): material: (pattern, strength, metres a tile).
VELVET, WOVEN, CARPET, GRAIN = range(4)
DETAIL = {
    "Seat fabric": (VELVET, 0.55, 0.12),
    "Headrest fabric": (VELVET, 0.55, 0.12),
    "Seat leather": (GRAIN, 0.45, 0.07),
    "Recliner leather": (GRAIN, 0.3, 0.06),
    "Wall fabric": (WOVEN, 0.45, 0.5),
    "Panel fabric": (WOVEN, 0.45, 0.5),
    "Baffle": (WOVEN, 0.35, 0.5),
    "Carpet": (CARPET, 0.6, 0.3),
    "Armrest leather": (GRAIN, 0.5, 0.06),
    "Seat shell": (GRAIN, 0.35, 0.08),
}


def origin():
    return Vector((0.0, eye_y(ORIGIN_ROW), FLOORS[ORIGIN_ROW] + EYE_HEIGHT))


def to_player(v, offset=True):
    """Blender (x right, y forward, z up) to the player's room metres (x
    right, y up, -z forward), from the origin seat's eyes."""
    if offset:
        v = v - origin()
    return (v.x, v.z, -v.y)


def select(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def unwrap(objs, margin):
    select(objs, objs[0])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin,
                             area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def decimated(source, name, ratio, collection):
    obj = link(bpy.data.objects.new(name, source.data.copy()), collection)
    obj.location = source.location.copy()
    obj.rotation_euler = source.rotation_euler.copy()
    d = obj.modifiers.new("Fewer faces", "DECIMATE")
    d.ratio = ratio
    apply_modifiers(obj)
    return obj


def bake_into(image):
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nodes = m.node_tree.nodes
        node = nodes.get("Bake target") or nodes.new("ShaderNodeTexImage")
        node.name = "Bake target"
        node.image = image
        for n in nodes:
            n.select = False
        node.select = True
        nodes.active = node


def bake(active, image, kind, samples, sources=(), reach=(0.03, 0.08), shine=False):
    """Bakes `kind` (COMBINED light, DIFFUSE colour or NORMAL) into `image`,
    from `sources` if any, looking `reach` (cage, ray; metres) for them.
    The light leaves out glossy reflections unless `shine`: they change with
    where you look from, though leather looks flat without them."""
    bake_into(image)
    select([*sources, active], active)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    settings = scene.render.bake
    settings.use_selected_to_active = bool(sources)
    settings.cage_extrusion = reach[0]
    settings.max_ray_distance = reach[1] if sources else 0.0
    settings.margin = 8
    settings.margin_type = "EXTEND"
    # The seat and the end standard share an image: keep what each baked.
    settings.use_clear = False
    # The colour pass is the surfaces' colour alone.
    lit = kind == "COMBINED"
    settings.use_pass_direct = settings.use_pass_indirect = lit
    settings.use_pass_color = not lit
    settings.use_pass_diffuse = settings.use_pass_emit = True
    settings.use_pass_glossy = lit and shine
    settings.use_pass_transmission = False
    settings.normal_space = "OBJECT"
    bpy.ops.object.bake(type=kind)
    print("Baked", kind.lower(), "of", active.name, flush=True)


def baked(name, size, parts, samples, path, quality, reach=(0.03, 0.08), shine=False):
    """Bakes the light on `parts`, (object, sources) pairs sharing one
    texture, and writes it to `path` denoised. Path tracing leaves grain
    that baking keeps; the surfaces' colour and direction, baked alongside,
    guide the denoiser so their edges and patterns stay sharp."""
    passes = []
    for kind, n in (("COMBINED", samples), ("DIFFUSE", 16), ("NORMAL", 16)):
        image = bpy.data.images.new(f"{name} {kind.lower()}", size, size, alpha=False,
                                    float_buffer=True)
        for obj, sources in parts:
            bake(obj, image, kind, n, sources, reach, shine)
        passes.append(image)
    denoise(*passes, path, quality)


def denoise(noisy, albedo, normal, path, quality):
    """Writes `noisy` as a JPEG after Open Image Denoise, run by the
    compositor of a scene made for it."""
    main = bpy.context.scene
    scene = bpy.data.scenes.new("Denoise")
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = noisy.size
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = main.view_settings.view_transform
    scene.view_settings.exposure = main.view_settings.exposure
    # Rendering needs a camera, though the compositor replaces its picture.
    camera = bpy.data.objects.new("Denoise camera", bpy.data.cameras.new("Denoise camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    tree = bpy.data.node_groups.new("Denoise", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    nodes, links = tree.nodes, tree.links

    def source(image):
        node = nodes.new("CompositorNodeImage")
        node.image = image
        return node.outputs["Image"]

    oidn = nodes.new("CompositorNodeDenoise")
    oidn.inputs["HDR"].default_value = True
    links.new(source(noisy), oidn.inputs["Image"])
    links.new(source(albedo), oidn.inputs["Albedo"])
    # A baked normal is stored as 0.5 + n / 2.
    unpack = nodes.new("ShaderNodeVectorMath")
    unpack.operation = "MULTIPLY_ADD"
    links.new(source(normal), unpack.inputs[0])
    unpack.inputs[1].default_value = (2.0, 2.0, 2.0)
    unpack.inputs[2].default_value = (-1.0, -1.0, -1.0)
    links.new(unpack.outputs["Vector"], oidn.inputs["Normal"])
    links.new(oidn.outputs["Image"], nodes.new("NodeGroupOutput").inputs["Image"])
    scene.compositing_node_group = tree
    bpy.ops.render.render(scene=scene.name)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = quality
    bpy.data.images["Render Result"].save_render(str(path), scene=scene)
    print("Denoised", path.name, flush=True)


def detail_texture(path, size=512, seed=11):
    """Writes the tiling patterns, one a channel: velvet pile, a woven
    fabric, carpet pile and a leather grain. Each averages mid-grey, so the
    player's smaller mipmaps of it fade to nothing where a pattern would be
    too fine to see and could only shimmer."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(f, f))

    def noise(period):
        """Noise with features about `period` pixels across; it repeats
        with the tile, being made from the tile's own frequencies."""
        spectrum = np.fft.fft2(rng.standard_normal((size, size)))
        keep = np.exp(-0.5 * ((radius - 1 / period) / (0.5 / period)) ** 2)
        n = np.real(np.fft.ifft2(spectrum * keep))
        return n / n.std()

    def slubs(length, width):
        """A fabric's thicker and thinner threads: streaks `length` pixels
        long and `width` across, along both the warp and the weft."""
        fy, fx = np.meshgrid(f, f, indexing="ij")
        total = 0
        for along, across in ((fx, fy), (fy, fx)):
            spectrum = np.fft.fft2(rng.standard_normal((size, size)))
            keep = np.exp(-0.5 * ((along * length) ** 2 + (across * width) ** 2))
            keep[0, 0] = 0.0
            n = np.real(np.fft.ifft2(spectrum * keep))
            total = total + n / n.std()
        return total

    def cells(across):
        """Leather's creases: dark where two of `across` x `across` jittered
        cells meet, the cells wrapping round the tile."""
        cell = size / across
        jitter = rng.random((across, across, 2))
        y, x = np.mgrid[0:size, 0:size] + 0.5
        ci, cj = (x // cell).astype(int), (y // cell).astype(int)
        nearest = np.full((2, size, size), np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = ci + di, cj + dj
                at = jitter[i % across, j % across]
                d = np.hypot(x - (i + at[..., 0]) * cell, y - (j + at[..., 1]) * cell)
                nearest[1] = np.minimum(nearest[1], np.maximum(nearest[0], d))
                nearest[0] = np.minimum(nearest[0], d)
        crease = (nearest[1] - nearest[0]) / cell
        return np.clip(crease / 0.3, 0.0, 1.0) ** 0.5

    y, x = np.mgrid[0:size, 0:size]
    threads = 2 * np.pi * 80 * np.stack([x, y]) / size
    patterns = [
        noise(4) + 0.3 * noise(48),
        # Slubs a few centimetres long on the walls' tile still show from the
        # seats; the weave itself only close up.
        slubs(60, 5) + 0.35 * (np.cos(threads[0]) + np.cos(threads[1])) + 0.3 * noise(3),
        noise(3) + 0.7 * noise(12),
        cells(40) + 0.15 * noise(3),
    ]
    channels = []
    for p in patterns:
        p = (p - p.mean()) / p.std()
        channels.append(np.clip(np.round(127.5 + 127.5 * p / 3.0), 0, 255))
    rgba = np.stack(channels, axis=-1).astype(np.uint8)
    rows = b"".join(b"\0" + row.tobytes() for row in rgba)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
    print("Wrote", path.name, flush=True)


def detail_bytes(normal, material):
    """A vertex's last four bytes: flags (bit 0 stays lit with the house
    lights down, bits 1-2 the pattern, bits 3-4 the player axis the face
    mostly faces, which the pattern is projected along, bit 5 goes out with
    the house lights, bit 6 plain white, bit 7 bends with the screen), the
    pattern's strength out of 255 and its tile in centimetres. The player
    works out where in the pattern a vertex is from its position, which
    keeps vertices small."""
    pattern, strength, tile = DETAIL.get(material, (0, 0.0, 0.01))
    facing = to_player(normal, offset=False)
    axis = max(range(3), key=lambda i: abs(facing[i]))
    flags = (int(material in GLOW) | pattern << 1 | axis << 3 | int(material in OUT) << 5
             | int(material in WHITE) << 6 | int(material in BENDS) << 7)
    return flags, round(strength * 255), round(tile * 100), 0


def mesh_record(obj, texture, cull, near, far, instances, offset):
    """One mesh of the player's room file: vertices (position, texture
    coordinate, detail_bytes), triangles, and where it is drawn."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    names = [m.name if m else "" for m in mesh.materials] or [""]
    index, vertices, indices = {}, bytearray(), []
    world = obj.matrix_world if offset else Matrix.Identity(4)
    for tri in mesh.loop_triangles:
        name = names[tri.material_index]
        extra = detail_bytes(world.to_3x3() @ tri.normal, name)
        for loop in tri.loops:
            co = world @ mesh.vertices[mesh.loops[loop].vertex_index].co
            p = to_player(co, offset)
            u, v = uv[loop].uv
            packed = struct.pack("<5f4B", *p, u, 1.0 - v, *extra)
            if packed not in index:
                index[packed] = len(index)
                vertices.extend(packed)
            indices.append(index[packed])
    head = struct.pack("<IIffIII", texture, cull, near, far, len(index), len(indices),
                       len(instances))
    body = vertices + struct.pack(f"<{len(indices)}I", *indices)
    body += b"".join(struct.pack("<4f", *to_player(Vector(at)), 0.0) for at in instances)
    print(f"{obj.name}: {len(indices) // 3} triangles x {len(instances)}", flush=True)
    return head + body


def export(out, mats, chair, end, seats):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    use_gpu(scene)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.6
    scene.world = bpy.data.worlds.new("Dark")
    scene.world.color = (0, 0, 0)
    work = bpy.data.collections.new("Export")
    scene.collection.children.link(work)
    # The shell: everything but the seats, as one mesh.
    room = bpy.data.collections["Room"]
    # Lettering is its own mesh, drawn after the room: it lies all but on the
    # sign behind it, and the later drawing wins a tie.
    words = [o for o in room.objects if o.name.startswith("Letters ")]
    for o in words:
        room.objects.unlink(o)
        work.objects.link(o)
    shell = join([o for o in room.objects if o.type == "MESH"], "Shell")
    lettering = join(words, "Lettering")
    lettering.data.uv_layers.new(name="UV")
    # Lighter seats for the player; a chair's detail comes from the full one.
    kind = SEAT_KINDS[SEAT]
    near_ratio, far_ratio = kind["near"], kind["far"]
    source = bpy.data.objects[f"Seat {ORIGIN_ROW * SEATS + middle_column()}"]
    near = decimated(source, "Seat near", near_ratio, work)
    props = [near]
    seat_parts = [(near, [source])]
    if end:
        last = next(i for i, (_, end_of_block) in enumerate(seat_columns()) if end_of_block)
        end_at = bpy.data.objects[f"End {ORIGIN_ROW * SEATS + last}"]
        end_bake = link(bpy.data.objects.new("End standard bake", end.data), work)
        end_bake.location = end_at.location.copy()
        end_at.hide_render = True
        props.append(end_bake)
        seat_parts.append((end_bake, ()))
    bpy.context.view_layer.update()
    unwrap([shell], 0.002)
    if not kind["own_layout"]:
        unwrap(props, 0.004)
    baked("Shell", 4096, [(shell, ())], 512, out / "cinema-hall.jpg", 95)
    baked("Seats", 2048, seat_parts, 512, out / "props.jpg", 95, kind["reach"], kind["shine"])
    far = decimated(near, "Seat far", far_ratio, work)
    positions = list(seat_positions())
    records = [
        mesh_record(shell, 0, 0, 0.0, 1e9, [tuple(origin())], True),
        mesh_record(lettering, 0, 0, 0.0, 1e9, [tuple(origin())], True),
        mesh_record(near, 1, 1, 0.0, NEAR_DETAIL, positions, False),
        mesh_record(far, 1, 1, NEAR_DETAIL, 1e9, positions, False),
    ]
    if end:
        records.append(mesh_record(end_bake, 1, 1, 0.0, 1e9, list(end_positions()), False))
    # The player's textures: cinema-hall.jpg, props.jpg, then detail.png.
    detail_texture(out / "detail.png")
    (out / "cinema.bin").write_bytes(
        b"FPXC" + struct.pack("<III", 4, len(records), 2) + b"".join(records))
    eye0 = origin()
    params = {
        "screen_width": SCREEN_W,
        "screen_height": SCREEN_H,
        "screen_center": list(to_player(Vector((0.0, 0.0, SCREEN_BOTTOM + SCREEN_H / 2)))),
        "max_curvature": max_curvature(),
        "seats": [list(to_player(Vector((0.0, eye_y(k), FLOORS[k] + EYE_HEIGHT))))
                  for k in range(ROWS)],
    }
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print("Exported to", out, "origin", tuple(eye0), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export", action="store_true",
                        help="bake and write the player's room files into DIR")
    parser.add_argument("--frame", type=Path, help="a 16:9 picture for the screen in previews")
    parser.add_argument("--room", choices=tuple(ROOMS), default="classic")
    parser.add_argument("--seat-model", type=Path,
                        help="the Luxe's recliner: the downloaded model's scene.gltf")
    parser.add_argument("--scheme", choices=("bordeaux", "graphite", "slate"),
                        help="the seats' colours, if not the room's own")
    parser.add_argument("--shots", default="", help="comma-separated preview names")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure(args.room)
    global SEAT_MODEL
    SEAT_MODEL = args.seat_model
    args.scheme = args.scheme or SCHEME
    print("Row floors", [round(f, 2) for f in FLOORS], flush=True)
    mats = build_materials(args.scheme)
    room = bpy.data.collections.new("Room")
    seats = bpy.data.collections.new("Seats")
    parts = bpy.data.collections.new("Seat parts")
    for c in (room, seats, parts):
        bpy.context.scene.collection.children.link(c)
    BUILDERS[ROOM](mats, room)
    build_seat, build_end = SEAT_KINDS[SEAT]["build"], SEAT_KINDS[SEAT]["end"]
    chair = build_seat(mats, parts)
    end = build_end(mats, parts) if build_end else None
    print("Seat", len(chair.data.polygons), "faces", flush=True)
    parts.hide_render = True
    place_seats(chair, end, seats)
    if args.preview:
        preview(args.out, args.frame, args.scheme, [s for s in args.shots.split(",") if s])
    if args.export:
        export(args.out, mats, chair, end, seats)


if __name__ == "__main__":
    main()
