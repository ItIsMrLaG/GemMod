import taichi as ti
import taichi.math as tm

from cat_simulation.constants import *


@ti.func
def move_pattern_random(point_: ti.template()) -> ti.math.vec2:
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


@ti.dataclass
class Cat:
    point: tm.vec2
    status: ti.i32

    # moving
    move_pattern: ti.i32
    prev_point: tm.vec2

    # pixel_pos
    pixel: tm.ivec2

    @ti.func
    def init_pos(self):
        # self.prev_point = [0, 0] # by default all values == 0
        self.point = tm.vec2([ti.random() * PLATE_WIDTH, ti.random() * PLATE_HEIGHT])

    @ti.func
    def move(self):
        self.status = GREEN_INTERACTION_LEVEL
        if self.move_pattern == MOVE_PATTERN_RANDOM_ID:
            self.prev_point = self.point
            self.point = move_pattern_random(self.point)

        # TODO: use (point and prev_point) to determine move vector + maybe use different dist_func there
        # elif

    @ti.func
    def update_pixel(self):
        self.pixel[0] = ti.min(ti.round(self.point[0], dtype=ti.i32), PLATE_WIDTH - 1)
        self.pixel[1] = ti.min(ti.round(self.point[1], dtype=ti.i32), PLATE_HEIGHT - 1)

    @ti.func
    def paint_pixel(self, pixels: ti.template()):
        # TODO: maybe something like that
        # ti.atomic_max(PIXELS[F_CATS[idx].pixel[0], F_CATS[idx].pixel[1]], BLACK_COLOR)
        pixels[self.pixel[0], self.pixel[1]] = BLACK_COLOR
