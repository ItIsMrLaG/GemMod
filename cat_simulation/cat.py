import taichi as ti
import taichi.math as tm

from cat_simulation.tools import get_distance, move_pattern_random
from cat_simulation.constants import (
    RED_COLOR,
    YELLOW_COLOR,
    GREEN_COLOR,
    INTERACTION_LEVEL_0,
    INTERACTION_LEVEL_1,
    INTERACTION_NO,
    MOVE_PATTERN_RANDOM_ID
)

_RADIUS_0: ti.f32
_RADIUS_1: ti.f32
_PLATE_WIDTH: ti.i32
_PLATE_HEIGHT: ti.i32


def init_cat_env(
        r0: ti.f32,
        r1: ti.f32,
        width: ti.i32,
        height: ti.i32
):
    global _RADIUS_0, _RADIUS_1, _PLATE_WIDTH, _PLATE_HEIGHT
    _RADIUS_0 = r0
    _RADIUS_1 = r1
    _PLATE_WIDTH = width
    _PLATE_HEIGHT = height


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
        self.norm_point = tm.vec2([point[0] / _PLATE_WIDTH, point[1] / _PLATE_HEIGHT])

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
    def init_cat(self, cat_r: ti.f32):
        self.radius = cat_r
        point = tm.vec2([ti.random() * _PLATE_WIDTH, ti.random() * _PLATE_HEIGHT])
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

        if dist <= _RADIUS_0:
            self_status = INTERACTION_LEVEL_0

        elif dist <= _RADIUS_1:  # ti.random() <= 1.0 / (dist * dist)
            self_status = ti.max(self.status, INTERACTION_LEVEL_1)

        else:
            self_status = INTERACTION_NO

        self._update_status_with_color(self_status)
