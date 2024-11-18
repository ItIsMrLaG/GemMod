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
