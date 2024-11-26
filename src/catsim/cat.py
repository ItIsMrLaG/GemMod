import taichi as ti
import taichi.math as tm

from catsim.constants import (
    # PROBABILISTIC INTERACTION #
    DISABLE_PROB_INTER,
    INTERACTION_LEVEL_0,
    INTERACTION_LEVEL_1,
    INTERACTION_NO,
    # PATTERNS #
    MOVE_PATTERN_LINE_ID,
    MOVE_PATTERN_PHIS_ID,
    MOVE_PATTERN_RANDOM_ID,
)
from catsim.tools import (
    get_distance,
    move_pattern_line,
    move_pattern_phis,
    move_pattern_random,
)

_RADIUS_0: ti.f32
_RADIUS_1: ti.f32
_PLATE_WIDTH: ti.i32
_PLATE_HEIGHT: ti.i32
_MOVE_RADIUS: ti.f32
_MOVE_PATTERN: ti.i32
_PROB_INTER: ti.i32


def init_cat_env(
    move_radius: ti.f32,
    r0: ti.f32,
    r1: ti.f32,
    width: ti.i32,
    height: ti.i32,
    move_pattern: ti.i32,
    prob_inter: ti.i32,
):
    global \
        _RADIUS_0, \
        _RADIUS_1, \
        _PLATE_WIDTH, \
        _PLATE_HEIGHT, \
        _MOVE_RADIUS, \
        _MOVE_PATTERN, \
        _PROB_INTER
    _MOVE_RADIUS = move_radius
    _RADIUS_0 = r0
    _RADIUS_1 = r1
    _PLATE_WIDTH = width
    _PLATE_HEIGHT = height
    _MOVE_PATTERN = move_pattern
    _PROB_INTER = prob_inter


@ti.dataclass
class Cat:
    radius: ti.f32

    status: ti.i32
    move_pattern: ti.i32

    point: tm.vec2
    norm_point: tm.vec2
    prev_point: tm.vec2

    @ti.func
    def _set_point(self, point: tm.vec2):
        self.point = point
        self.norm_point = tm.vec2([point[0] / _PLATE_WIDTH, point[1] / _PLATE_HEIGHT])

    @ti.func
    def init_cat(self, cat_r: ti.f32):
        point = tm.vec2([ti.random() * _PLATE_WIDTH, ti.random() * _PLATE_HEIGHT])
        self.radius = cat_r
        self._set_point(point)
        self.prev_point = move_pattern_random(
            self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT
        )
        self.move_pattern = _MOVE_PATTERN

    @ti.func
    def move(self):
        self.status = INTERACTION_NO

        prev_point = self.prev_point
        self.prev_point = self.point

        if self.move_pattern == MOVE_PATTERN_RANDOM_ID:
            self._set_point(
                move_pattern_random(
                    self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT
                )
            )

        elif self.move_pattern == MOVE_PATTERN_LINE_ID:
            self._set_point(
                move_pattern_line(
                    self.point,
                    prev_point,
                    _MOVE_RADIUS,
                    _PLATE_WIDTH,
                    _PLATE_HEIGHT,
                )
            )

        elif self.move_pattern == MOVE_PATTERN_PHIS_ID:
            self._set_point(
                move_pattern_phis(self.point, prev_point, _PLATE_WIDTH, _PLATE_HEIGHT)
            )

        else:
            assert False

    @ti.func
    def fight_with(self, other_cat: ti.template(), distance_type: ti.i32):
        dist = get_distance(self.point, other_cat.point, distance_type)

        if dist <= _RADIUS_0:
            self.status = INTERACTION_LEVEL_0

        elif dist <= _RADIUS_1 and (
            _PROB_INTER == DISABLE_PROB_INTER or ti.random() < 1.0 / (dist * dist)
        ):
            self.status = ti.max(self.status, INTERACTION_LEVEL_1)

        else:
            self.status = ti.max(self.status, INTERACTION_NO)
