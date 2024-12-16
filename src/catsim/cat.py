import taichi as ti
import taichi.math as tm

from catsim.config import OBSERVABLE_ANGLE_SPAN
from catsim.enums import (
    ALWAYS_VISIBLE,
    INTERACTION_LEVEL_0,
    INTERACTION_LEVEL_1,
    INTERACTION_NO,
    INVISIBLE,
    MOVE_PATTERN_LINE,
    MOVE_PATTERN_PHIS,
    MOVE_PATTERN_RANDOM,
    VISIBLE,
)
from catsim.tools import (
    get_distance,
    get_distance_py,
    get_sides_distance,
    is_boundary_point,
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
_MAX_DISTANCE: ti.f32
_DISTANCE_TYPE: ti.i32

_PROB_INTER: bool
_BORDER_INTER: bool


def init_cat_env(
    move_radius: ti.f32,
    r0: ti.f32,
    r1: ti.f32,
    width: ti.i32,
    height: ti.i32,
    move_pattern: ti.i32,
    prob_inter: bool,
    distance_type: ti.i32,
    border_inter: bool,
):
    global \
        _RADIUS_0, \
        _RADIUS_1, \
        _PLATE_WIDTH, \
        _PLATE_HEIGHT, \
        _MOVE_RADIUS, \
        _MOVE_PATTERN, \
        _PROB_INTER, \
        _BORDER_INTER, \
        _DISTANCE_TYPE, \
        _MAX_DISTANCE

    _MOVE_RADIUS = move_radius
    _RADIUS_0 = r0
    _RADIUS_1 = r1
    _PLATE_WIDTH = width
    _PLATE_HEIGHT = height
    _MOVE_PATTERN = move_pattern
    _PROB_INTER = prob_inter
    _BORDER_INTER = border_inter
    _DISTANCE_TYPE = distance_type

    p0 = tm.vec2([0, 0])
    p1 = tm.vec2([0, _PLATE_WIDTH])
    p2 = tm.vec2([_PLATE_HEIGHT, 0])
    _MAX_DISTANCE = (
        get_distance_py(p0, p1, _DISTANCE_TYPE) * 2
        + get_distance_py(p0, p2, _DISTANCE_TYPE) * 2
    )


@ti.dataclass
class Cat:
    idx: ti.i32
    radius: ti.f32
    move_pattern: ti.i32

    status: ti.i32

    point: tm.vec2
    norm_point: tm.vec2
    prev_point: tm.vec2

    # indicates whether a cat is observed
    # by someone with ALWAYS_VISIBLE status
    observed: bool

    # maintained only for cats with ALWAYS_VISIBLE status
    observable_angle: tm.vec2

    visibility_status: ti.i8
    fixed_point: tm.vec2
    distance_invisible: ti.f32

    @ti.func
    def set_point(self, point: tm.vec2):
        self.point = point

        # todo: maybe remove it (calculate it statically)
        self.norm_point = tm.vec2([point[0] / _PLATE_WIDTH, point[1] / _PLATE_HEIGHT])

    @ti.func
    def update_visibility_status(self):
        # ON VISIBLE STATE #
        if self.visibility_status == VISIBLE:
            prob = ti.random(dtype=ti.i32) % 2 + ti.random(dtype=ti.i32) % 3
            if prob and is_boundary_point(self.point, _PLATE_WIDTH, _PLATE_HEIGHT) != 0:
                self.fixed_point = self.point
                self.distance_invisible = 0
                self.visibility_status = INVISIBLE

        # ON INVISIBLE STATE #
        elif self.visibility_status == INVISIBLE:
            flag = 0
            if is_boundary_point(self.point, _PLATE_WIDTH, _PLATE_HEIGHT):
                if (
                    get_sides_distance(
                        self.point,
                        self.fixed_point,
                        _PLATE_WIDTH,
                        _PLATE_HEIGHT,
                        _DISTANCE_TYPE,
                    )
                    <= self.distance_invisible
                ):
                    self.distance_invisible = 0
                    self.visibility_status = VISIBLE
                    flag = 1

            if flag == 0:
                new_dist = self.distance_invisible + _MOVE_RADIUS
                self.distance_invisible = tm.max(new_dist, _MAX_DISTANCE)

    @ti.func
    def init_border_inter(self, visibility_status: ti.i8):
        assert visibility_status != INVISIBLE
        self.visibility_status = visibility_status
        self.update_visibility_status()

    @ti.func
    def init_cat(
        self,
        idx: ti.i32,
        cat_r: ti.f32,
        visibility_status: ti.i8 = VISIBLE,
    ):
        self.idx = idx
        self.radius = cat_r
        self.prev_point = move_pattern_random(
            self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT
        )
        self.move_pattern = _MOVE_PATTERN

        self.observed = False
        self.observable_angle = tm.vec2([0, 0])

        if _BORDER_INTER:
            self.init_border_inter(visibility_status)

    @ti.func
    def move(self):
        self.status = INTERACTION_NO

        prev_point = self.prev_point
        self.prev_point = self.point

        if self.move_pattern == MOVE_PATTERN_RANDOM:
            self.set_point(
                move_pattern_random(
                    self.point, _MOVE_RADIUS, _PLATE_WIDTH, _PLATE_HEIGHT
                )
            )

        elif self.move_pattern == MOVE_PATTERN_LINE:
            self.set_point(
                move_pattern_line(
                    self.point,
                    prev_point,
                    _MOVE_RADIUS,
                    _PLATE_WIDTH,
                    _PLATE_HEIGHT,
                )
            )

        elif self.move_pattern == MOVE_PATTERN_PHIS:
            self.set_point(
                move_pattern_phis(self.point, prev_point, _PLATE_WIDTH, _PLATE_HEIGHT)
            )

        # reset `observed` on every move. would be set for observed cats later
        self.observed = False

        # maintain ALWAYS_VISIBLE cats' observable angle
        if self.visibility_status == ALWAYS_VISIBLE:
            direction_angle = tm.atan2(
                self.point[1] - self.prev_point[1], self.point[0] - self.prev_point[0]
            )
            self.observable_angle[0] = direction_angle - OBSERVABLE_ANGLE_SPAN
            self.observable_angle[1] = direction_angle + OBSERVABLE_ANGLE_SPAN

        if _BORDER_INTER:
            self.update_visibility_status()

    @ti.func
    def fight_with(self, other_cat: ti.template()):
        if (
            other_cat.visibility_status == VISIBLE
            or other_cat.visibility_status == ALWAYS_VISIBLE
        ):
            dist = get_distance(self.point, other_cat.point, _DISTANCE_TYPE)

            if dist > _RADIUS_1 or (_PROB_INTER and ti.random() >= 1.0 / (dist * dist)):
                self.status = ti.max(self.status, INTERACTION_NO)

            else:
                observing = True
                if self.visibility_status == ALWAYS_VISIBLE:
                    relative_angle = tm.atan2(
                        other_cat.point[1] - self.point[1],
                        other_cat.point[0] - self.point[0],
                    )
                    if (
                        self.observable_angle[0]
                        < relative_angle
                        < self.observable_angle[1]
                    ) or dist <= _RADIUS_0:
                        other_cat.observed = True
                    else:
                        observing = False

                if observing:
                    self.status = (
                        INTERACTION_LEVEL_0
                        if dist <= _RADIUS_0
                        else ti.max(self.status, INTERACTION_LEVEL_1)
                    )
