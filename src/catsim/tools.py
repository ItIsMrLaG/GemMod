import taichi as ti
import taichi.math as tm

from catsim.constants import (
    CHEBYSHEV_DISTANCE,
    MANHATTAN_DISTANCE,
)


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
def move_pattern_random(
    point_: tm.vec2, move_r: ti.f32, plate_w: ti.i32, plate_h: ti.i32
) -> tm.vec2:
    # unsigned values
    xd_u = ti.random() * move_r
    yd_u = ti.random() * move_r

    # generate sign: sign_f(x) = -2x + 1
    #   if sign_f(1) -> -1
    #   if sign_f(0) -> +1
    xd_s = -2 * (ti.random(dtype=ti.i32) % 2) + 1
    yd_s = -2 * (ti.random(dtype=ti.i32) % 2) + 1

    new_x = point_[0] + xd_u * xd_s
    new_y = point_[1] + yd_u * yd_s

    if not (0 <= new_x <= plate_w):
        new_x = point_[0] - xd_u * xd_s

    if not (0 <= new_y <= plate_h):
        new_y = point_[1] - yd_u * yd_s

    return ti.math.vec2([new_x, new_y])


@ti.func
def move_pattern_line(
    point_: tm.vec2,
    old_point_: tm.vec2,
    move_r: ti.f32,
    plate_w: ti.i32,
    plate_h: ti.i32,
) -> tm.vec2:
    delta = point_ - old_point_
    n_point_ = point_ + delta

    random_point_ = move_pattern_random(point_, move_r, plate_w, plate_h)

    if not (0 <= n_point_[0] <= plate_w):
        n_point_[0] = random_point_[0]

    if not (0 <= n_point_[1] <= plate_h):
        n_point_[1] = random_point_[1]

    return n_point_


@ti.func
def move_pattern_phis(
    point_: tm.vec2, old_point_: tm.vec2, plate_w: ti.i32, plate_h: ti.i32
) -> tm.vec2:
    delta = point_ - old_point_
    n_point_ = point_ + delta

    if not (0 <= n_point_[0] <= plate_w):
        n_point_[0] = 0 if n_point_[0] <= 0 else plate_w
        n_point_[0] -= delta[0]

    if not (0 <= n_point_[1] <= plate_h):
        n_point_[1] = 0 if n_point_[1] <= 0 else plate_h
        n_point_[1] -= delta[1]

    return n_point_
