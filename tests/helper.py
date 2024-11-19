from cat_simulation.cat import init_cat_env, Cat
import taichi as ti
from cat_simulation.tools import get_distance
import cat_simulation.constants as const


def init_cats(n, r0, r1, width, height):
    init_cat_env(
        r0,
        r0,
        r1,
        width,
        height,
        const.MOVE_PATTERN_RANDOM_ID,
        const.RED_COLOR,
        const.YELLOW_COLOR,
        const.GREEN_COLOR,
        const.DISABLE_PROB_INTER,
    )
    cats = Cat.field(shape=(n,))
    return cats


def init_consts(n, radius, r0, r1, width, height):
    cats = init_cats(n, r0, r1, width, height)
    points = ti.Vector.field(n=2, dtype=float, shape=(n,))
    return cats, points, radius, n


@ti.kernel
def init_cats_with_custom_points(
    n: ti.i32, cats: ti.template(), points: ti.template(), radius: ti.f32
):
    for i in range(n):
        cats[i].radius = radius
        cats[i]._set_point(points[i])
        cats[i].move_pattern = const.MOVE_PATTERN_RANDOM_ID
        cats[i].status = const.INTERACTION_NO
        cats[i].color = const.GREEN_COLOR


@ti.kernel
def set_cat_init_positions(n: ti.i32, r: ti.f32, cats: ti.template()):
    for idx in range(n):
        cats[idx].init_cat(r)


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
