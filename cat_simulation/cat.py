import taichi as ti
import taichi.math as tm

from cat_simulation.run_config import COLOR_LEVEL_1
from cat_simulation.tools import (
    get_distance,
    move_pattern_random,
    move_pattern_line,
    move_pattern_phis
)
from cat_simulation.constants import (
    # LEVELS #
    INTERACTION_LEVEL_0,
    INTERACTION_LEVEL_1,
    INTERACTION_NO,
    # PATTERNS #
    MOVE_PATTERN_RANDOM_ID,
    MOVE_PATTERN_LINE_ID,
    MOVE_PATTERN_PHIS_ID
)

_RADIUS_0: ti.f32
_RADIUS_1: ti.f32
_PLATE_WIDTH: ti.i32
_PLATE_HEIGHT: ti.i32
_MOVE_RADIUS: ti.f32
_MOVE_PATTERN: ti.i32

_COLOR_LEVEL_0: ti.i32
_COLOR_LEVEL_1: ti.i32
_COLOR_LEVEL_NO: ti.i32


def init_cat_env(
        move_radius: ti.f32,
        r0: ti.f32,
        r1: ti.f32,
        width: ti.i32,
        height: ti.i32,
        move_pattern: ti.i32,
        l0_color: ti.i32,
        l1_color: ti.i32,
        lNO_color: ti.i32
):
    global _RADIUS_0, _RADIUS_1, _PLATE_WIDTH, _PLATE_HEIGHT, _MOVE_RADIUS, _MOVE_PATTERN
    _MOVE_RADIUS = move_radius
    _RADIUS_0 = r0
    _RADIUS_1 = r1
    _PLATE_WIDTH = width
    _PLATE_HEIGHT = height
    _MOVE_PATTERN = move_pattern

    global _COLOR_LEVEL_0, _COLOR_LEVEL_1, _COLOR_LEVEL_NO
    _COLOR_LEVEL_0 = l0_color
    _COLOR_LEVEL_1 = l1_color
    _COLOR_LEVEL_NO = lNO_color


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
            self.color = _COLOR_LEVEL_0

        elif status == INTERACTION_LEVEL_1:
            self.color = _COLOR_LEVEL_1

        else:
            self.color = _COLOR_LEVEL_NO

    @ti.func
    def init_cat(self, cat_r: ti.f32):
        point = tm.vec2([ti.random() * _PLATE_WIDTH, ti.random() * _PLATE_HEIGHT])
        self.radius = cat_r
        self._set_point(point)
        self.prev_point = move_pattern_random(self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT)
        self.color = _COLOR_LEVEL_NO
        self.move_pattern = _MOVE_PATTERN

    @ti.func
    def move(self):
        self._update_status_with_color(INTERACTION_NO)
        point = self.point

        if self.move_pattern == MOVE_PATTERN_RANDOM_ID:
            point = move_pattern_random(self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT)
            self._set_point(point)

        elif self.move_pattern == MOVE_PATTERN_LINE_ID:
            point = move_pattern_line(self.point, self.prev_point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT)

        elif self.move_pattern == MOVE_PATTERN_PHIS_ID:
            point = move_pattern_phis(self.point, self.prev_point, _PLATE_WIDTH, _PLATE_HEIGHT)

        self.prev_point = tm.vec2([self.point[0], self.point[1]])
        self._set_point(point)

    @ti.func
    def fight_with(self, other_cat: ti.template(), distance_type: ti.i32):
        dist = get_distance(self.point, other_cat.point, distance_type)

        self_status: ti.i32 = INTERACTION_NO

        if dist <= _RADIUS_0:
            self_status = INTERACTION_LEVEL_0

        elif dist <= _RADIUS_1:  # ti.random() <= 1.0 / (dist * dist)
            self_status = ti.max(self.status, INTERACTION_LEVEL_1)

        else:
            self_status = ti.max(self.status, INTERACTION_NO)

        self._update_status_with_color(self_status)
