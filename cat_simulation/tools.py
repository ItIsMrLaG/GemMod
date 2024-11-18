import taichi as ti
import taichi.math as tm

from cat_simulation.constants import *


@ti.func
def euclidean_distance(point1: tm.vec2, point2: tm.vec2) -> ti.f32:
    return (point1 - point2).norm()


@ti.func
def manhattan_distance(point1: tm.vec2, point2: tm.vec2) -> ti.f32:
    return ti.abs(point1.x - point2.x) + ti.abs(point1.y - point2.y)


@ti.func
def chebyshev_distance(point1: tm.vec2, point2: tm.vec2) -> ti.f32:
    return ti.max(ti.abs(point1.x - point2.x), ti.abs(point1.y - point2.y))


@ti.func
def get_distance(p1: tm.vec2, p2: tm.vec2, distance_type: ti.i32) -> ti.f32:
    ans: ti.f32 = 0

    if distance_type == MANHATTAN_DISTANCE:
        ans = manhattan_distance(p1, p2)
    elif distance_type == CHEBYSHEV_DISTANCE:
        ans = chebyshev_distance(p1, p2)
    else:
        ans = euclidean_distance(p1, p2)

    return ans


@ti.func
def move_pattern_random(point_: tm.vec2) -> tm.vec2:
    # unsigned values
    xd_u = ti.random() * MOVE_RADIUS
    yd_u = ti.random() * MOVE_RADIUS

    # generate sign: sign_f(x) = -2x + 1
    #   if sign_f(1) -> -1
    #   if sign_f(0) -> +1
    xd_s = -2 * (ti.random(dtype=ti.i32) % 2) + 1
    yd_s = -2 * (ti.random(dtype=ti.i32) % 2) + 1

    new_x = point_[0] + xd_u * xd_s
    new_y = point_[1] + yd_u * yd_s

    if not (PLATE_W_MIN <= new_x <= PLATE_WIDTH):
        new_x = point_[0] - xd_u * xd_s

    if not (PLATE_H_MIN <= new_y <= PLATE_HEIGHT):
        new_y = point_[1] - yd_u * yd_s

    return ti.math.vec2([new_x, new_y])
