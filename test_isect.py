import math
import rowan
import gsd.hoomd
import hoomd
from coxeter.shapes import ConvexPolyhedron
import numpy as np

tet = ConvexPolyhedron(vertices=[[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape["tet"] = {"vertices": tet.vertices.tolist()}
mc.nselect = 1
mc.d["tet"] = 0
mc.a["tet"] = 0


cases = [
    ("perfect_overlap", [0.0, 0.0, 0.0], [1, 0, 0, 0], True),
    # HOOMD does not handle out of box coordinates
    # ("particle_at_infinity", [math.inf, 0.0, 0.0], [1, 0, 0, 0], False),
    # ("particle_at_negative_infinity", [-math.inf, 0.0, 0.0], [1, 0, 0, 0], False),
    # (
    #     "tip_tip_intersection_exact",
    #     [2.0, 2.0, 2.0],
    #     rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
    #     True,
    # ),  # Passes in f64, fails in rust. This is OK to skip
    (
        "tip_tip_intersection_imprecise",
        [1.999, 1.999, 1.999],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
        True,
    ),
    (
        "tip_tip_intersection_nooverlap",
        [2.001, 2.001, 2.001],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
        False,
    ),
    ("unrotated_tip_tip_intersection_exact", [2.0, 2.0, 0.0], [1, 0, 0, 0], True),
    (
        "unrotated_tip_tip_intersection_imprecise",
        [1.999, 1.999, 0.0],
        [1, 0, 0, 0],
        True,
    ),
    (
        "unrotated_tip_tip_intersection_nooverlap",
        [2.001, 2.001, 0.0],
        [1, 0, 0, 0],
        False,
    ),
    ("tip_edge_intersection_exact", [1.0, 1.0, 2.0], [1, 0, 0, 0], True),
    ("tip_edge_intersection_imprecise", [1.0, 1.0, 1.999999], [1, 0, 0, 0], True),
    ("tip_edge_intersection_nooverlap", [1.0, 1.0, 2.001], [1, 0, 0, 0], False),
    (
        "parallel_edge_edge_intersection_exact",
        [1.0, 1.0, 2.0],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
        True,
    ),
    (
        "parallel_edge_edge_intersection_imprecise",
        [1.0, 1.0, 1.999],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
        True,
    ),
    (
        "parallel_edge_edge_intersection_nooverlap",
        [1.0, 1.0, 2.001],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2),
        False,
    ),
    ("orthogonal_edge_edge_intersection_exact", [1.0, 0.0, 2.0], [1, 0, 0, 0], True),
    (
        "orthogonal_edge_edge_intersection_imprecise",
        [1.0, 0.0, 1.999],
        [1, 0, 0, 0],
        True,
    ),
    (
        "orthogonal_edge_edge_intersection_nooverlap",
        [1.0, 0.0, 2.001],
        [1, 0, 0, 0],
        False,
    ),
    (
        "nonorthogonal_edge_edge_intersection_exact",
        [1.0, 0.0, 2.0],
        rowan.from_axis_angle([0.0, 0.0, 1.0], math.pi / 3.1),
        True,
    ),
    (
        "nonorthogonal_edge_edge_intersection_imprecise",
        [1.0, 0.0, 1.999],
        rowan.from_axis_angle([0.0, 0.0, 1.0], math.pi / 3.1),
        True,
    ),
    (
        "nonorthogonal_edge_edge_intersection_nooverlap",
        [1.0, 0.0, 2.01],
        rowan.from_axis_angle([0.0, 0.0, 1.0], math.pi / 3.1),
        False,
    ),
    ("partial_aligned_overlap_exact", [0.0, 1.0, -1.0], [1, 0, 0, 0], True),
    ("partial_aligned_overlap_imprecise", [0.0, 1.0, -0.999], [1, 0, 0, 0], True),
    ("partial_parallel_overlap", [0.0, 0.0, -1.0], [1, 0, 0, 0], True),
    (
        "vertex_into_edge_shallow_exact",
        [0.0, 1.0, 2.0],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 4),
        True,
    ),
    (
        "vertex_into_edge_shallow_imprecise",
        [0.0, 0.999, 2.0],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 4),
        True,
    ),
    (
        "vertex_into_edge_deep_exact",
        [0.0, 1.0, 1.0],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 4),
        True,
    ),
    (
        "vertex_into_edge_deep_imprecise",
        [0.0, 0.999, 1.0],
        rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 4),
        True,
    ),
    # (
    #     "vertex_face_imprecise",
    #     [1.0, 1.0, 2.0],
    #     rowan.from_axis_angle([1.0, 0.0, 0.0], math.pi / 4),
    #     True,
    # ),  # Fails in rust, also fails in HOOMD
    (
        "vertex_face_nooverlap",
        [1.2765, -1.2765, 1.2765],
        rowan.from_axis_angle([1.0, 1.0, 0.0], math.pi / 2),
        False,
    ),
    (
        "vertex_face_near_exact",
        [1.275, -1.275, 1.275],
        rowan.from_axis_angle([1.0, 1.0, 0.0], math.pi / 2),
        True,
    ),
]

f = gsd.hoomd.Frame()
f.particles.N = 2
f.particles.types = ["tet"]
f.particles.type_shapes = [tet.gsd_shape_spec]
f.configuration.box = [100, 100, 100, 0, 0, 0]

with gsd.hoomd.open("test.gsd", "w") as traj:
    traj.append(f)

for label, pos, versor, expected in cases:
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.operations.integrator = mc

    f.particles.position = [np.zeros(3), pos]
    f.particles.orientation = [[1, 0, 0, 0], versor]

    sim.create_state_from_snapshot(f)
    sim.run(0)
    try:
        assert sim.operations.integrator.overlaps == expected, (
            f"Case {label}: \nGot {sim.operations.integrator.overlaps}, expected {expected}\n"
        )
    except AssertionError:
        print(
            f"HOOMD-Blue Xenocollide did not match expected result for case `{label}`"
            f"\nExpected {expected}, got {sim.operations.integrator.overlaps != 0}"
        )


tet2 = ConvexPolyhedron([
    # nonorthogonal_edge_edge_intersection_nooverlap
    # [0.6803197528322115, 1.3776082678217132, 3.01],
    # [2.377608267821713, 0.31968024716778853, 1.0099999999999998],
    # [-0.3776082678217132, -0.31968024716778853, 1.0099999999999998],
    # [1.3196802471677884, -1.3776082678217132, 3.01],

    # orthogonal_edge_edge_intersection_nooverlap
    # [2, 1, 3.001], [2, -1, 1.001], [0, 1, 1.001], [0, -1, 3.001]

    # Shifted slightly
    # [2, 1, 3.001], [2, -1, 1.001], [0, 1, 1.001], [0, -1, 3.001]

    # tip_edge_intersection_exact
    [2, 2, 3], [2, 0, 1], [0, 2, 1], [0, 0, 3]
])
# tet2.centroid = tet2.centroid - [0, 0, 0.001]
print(tet2.centroid)


f.particles.types = ["A", "B"]
f.particles.typeid = [0, 1]
print(tet2.centroid)
f.particles.position = [[0,0,0], tet2.centroid[:]]
tet2.centroid = [0, 0, 0]
print(tet2.centroid)
print(f.particles.position)
f.particles.type_shapes.append(tet2.gsd_shape_spec)
f.particles.orientation = [[1,0,0,0], [1,0,0,0]]


sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape["A"] = {"vertices": tet.vertices.tolist()}
mc.shape["B"] = {"vertices": tet2.vertices.tolist()}
sim.create_state_from_snapshot(f)
sim.operations.integrator = mc
sim.run(0)
assert sim.operations.integrator.overlaps == True # noqa: E712
