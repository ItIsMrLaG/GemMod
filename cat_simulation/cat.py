import random

import taichi as ti
import taichi.math as tm

from cat_simulation.tools import get_distance, move_pattern_random
from cat_simulation.constants import *


@ti.dataclass
class Cat:
    radius: ti.f32

    status: ti.i32
    color: ti.i32

    point: tm.vec2
    norm_point: tm.vec2

    move_pattern: ti.i32
    prev_point: tm.vec2

    @ti.func
    def _set_point(self, point: tm.vec2):
        self.point = point
        self.norm_point = tm.vec2([point[0] / PLATE_WIDTH, point[1] / PLATE_HEIGHT])

    @ti.func
    def _update_status_with_color(self, status: ti.i32):
        self.status = status
        if status == INTERACTION_LEVEL_0:
            self.color = RED_COLOR
        elif status == INTERACTION_LEVEL_1:
            self.color = YELLOW_COLOR
        else:
            self.color = GREEN_COLOR

    @ti.func
    def init_cat(self):
        self.radius = CAT_RADIUS
        point = tm.vec2([ti.random() * PLATE_WIDTH, ti.random() * PLATE_HEIGHT])
        self._set_point(point)
        self.color = GREEN_COLOR

    @ti.func
    def move(self):
        self._update_status_with_color(INTERACTION_NO)

        if self.move_pattern == MOVE_PATTERN_RANDOM_ID:
            self.prev_point = tm.vec2([self.point[0], self.point[1]])
            point = move_pattern_random(self.point)
            self._set_point(point)

        # TODO: use (point and prev_point) to determine move vector + maybe use different dist_func there

    @ti.func
    def fight_with(self, other_cat: ti.template(), distance_type: ti.i32):
        dist = get_distance(self.point, other_cat.point, distance_type)
        # TODO: тут проверка на углы дома, наверное
        self_status: ti.i32 = INTERACTION_NO

        if dist <= RADIUS_0:
            self_status = INTERACTION_LEVEL_0

        elif dist <= RADIUS_1:  # ti.random() <= 1.0 / (dist * dist)
            self_status = ti.max(self.status, INTERACTION_LEVEL_1)

        else:
            self_status = INTERACTION_NO

        self._update_status_with_color(self_status)
