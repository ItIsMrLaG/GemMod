import taichi as ti
import catsim.constants as const
from catsim.tools import get_distance


@ti.kernel
def init_cats_with_custom_points(
    n: ti.i32,
    radius: ti.f32,
    cats: ti.template(),
    points: ti.template(),
):
    for i in range(n):
        cats[i].radius = radius
        cats[i].set_point(points[i])
        cats[i].move_pattern = const.MOVE_PATTERN_RANDOM_ID
        cats[i].status = const.INTERACTION_NO


@ti.kernel
def set_cat_init_positions(n: ti.i32, r: ti.f32, cats: ti.template()):
    for idx in range(n):
        cats[idx].init_cat(idx, r)


@ti.kernel
def primitive_update_states(
    n: ti.i32,
    cats: ti.template(),
    statuses: ti.types.ndarray(),
    distance_type: ti.i32,
    r0: ti.f32,
    r1: ti.f32,
):
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = get_distance(cats[i].point, cats[j].point, distance_type)
            if dist <= r0:
                statuses[i] = const.INTERACTION_LEVEL_0
            elif dist <= r1:
                statuses[i] = ti.max(const.INTERACTION_LEVEL_1, statuses[i])
            else:
                statuses[i] = ti.max(const.INTERACTION_NO, statuses[i])
