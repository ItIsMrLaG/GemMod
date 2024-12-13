import taichi as ti
import taichi.math as tm

from catsim.enums import (
    CHEBYSHEV_DISTANCE,
    MANHATTAN_DISTANCE,
)

EPS = 1


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


@ti.kernel
def get_distance_py(p1: tm.vec2, p2: tm.vec2, distance_type: ti.i32) -> ti.f32:
    return get_distance(p1, p2, distance_type)


@ti.func
def is_boundary_point(p: tm.vec2, max_x: ti.f32, max_y: ti.f32) -> ti.i8:
    min_y = 0.0
    min_x = 0.0
    ret = 0

    """    4
      -----------
    1 |  field  | 3
      -----------
           2         """
    # TODO: add assert (width and height > 10 maybe)

    if p[0] <= min_x + EPS:
        ret = 1
    if max_x - EPS <= p[0]:
        ret = 3

    if p[1] <= min_y + EPS:
        ret = 2
    if max_y - EPS <= p[1]:
        ret = 4

    return ret


@ti.func
def get_sides_distance(
    p1: tm.vec2,
    p2: tm.vec2,
    right_bound: ti.f32,
    upper_bound: ti.f32,
    distance_type: ti.i32,
) -> ti.f32:
    p1_mark = is_boundary_point(p1, right_bound, upper_bound)
    p2_mark = is_boundary_point(p2, right_bound, upper_bound)
    assert p1_mark and p2_mark

    ret: ti.f32 = -1

    x0_y0 = tm.vec2([0, 0])
    xn_y0 = tm.vec2([right_bound, 0])
    x0_yn = tm.vec2([0, upper_bound])
    xn_yn = tm.vec2([right_bound, upper_bound])

    # SAME SIDE
    if p1_mark == p2_mark:
        ret = get_distance(p1, p2, distance_type)

    # OPPOSITE SIDES
    elif p1_mark % 2 == p2_mark % 2:
        _height_dist = get_distance(x0_y0, x0_yn, distance_type)
        _width_dist = get_distance(x0_y0, xn_y0, distance_type)
        _adding_dist = _height_dist * ((p1_mark + 1) % 2) + _width_dist * (p1_mark % 2)

        """        4
             *b*-------+c+
            1 |  field  | 3
             +a+-------*d*
                   2
        *b* == attr_p1 | +a+ == x0_y0
        *d* == attr_p2 | +c+ == xn_yn
        """

        attr_p1 = x0_yn * ((p1_mark + 1) % 2) + xn_y0 * (p1_mark % 2)
        attr_p2 = xn_y0 * ((p1_mark + 1) % 2) + x0_yn * (p1_mark % 2)

        _p1 = p1
        _p2 = p2
        if p1_mark > p2_mark:
            _p1 = p2
            _p2 = p1

        _attr_dist1 = get_distance(x0_y0, _p1, distance_type) + get_distance(
            attr_p1, _p2, distance_type
        )
        _attr_dist2 = get_distance(xn_yn, _p2, distance_type) + get_distance(
            attr_p2, _p1, distance_type
        )
        _attr_dist = tm.min(_attr_dist1, _attr_dist2)

        ret = _adding_dist + _attr_dist

    # NEIGHBORING SIDES
    else:
        """        4
             (b)-------(c)
            1 |  field  | 3
             (a)-------(d)
                   2
           (b) = 5 | (d) = 5
           (a) = 3 | (c) = 7
        """

        _corn_p1, _corn_p2 = x0_y0, xn_yn

        if p1_mark + p2_mark == 5:
            _corn_p1, _corn_p2 = x0_yn, xn_y0

        _dist1 = get_distance(_corn_p1, p1, distance_type) + get_distance(
            _corn_p1, p2, distance_type
        )
        _dist2 = get_distance(_corn_p2, p1, distance_type) + get_distance(
            _corn_p2, p2, distance_type
        )
        ret = tm.min(_dist1, _dist2)

    assert ret >= 0
    return ret


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


@ti.kernel
def init_cats(cats_n: ti.i32, cat_radius: ti.f32, cats: ti.template(), v_st: ti.i8):
    for idx in range(cats_n):
        cats[idx].init_cat(idx, cat_radius, visibility_status=v_st)
