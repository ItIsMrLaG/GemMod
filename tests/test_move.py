import pytest
import taichi as ti
import taichi.math as tm

from catsim.tools import move_pattern_line, move_pattern_random


@ti.data_oriented
class TestMovePattern:
    @ti.kernel
    def move_random(
        self, point: tm.vec2, move_r: ti.f32, plate_w: ti.i32, plate_h: ti.i32
    ) -> tm.vec2:
        return move_pattern_random(point, move_r, plate_w, plate_h)

    @ti.kernel
    def move_line(
        self,
        point: tm.vec2,
        old_point: tm.vec2,
        move_r: ti.f32,
        plate_w: ti.i32,
        plate_h: ti.i32,
    ) -> tm.vec2:
        return move_pattern_line(point, old_point, move_r, plate_w, plate_h)

    @pytest.mark.parametrize(
        "x, y, move_radius",
        [
            (10, 50, 8),
            (42, 17, 15),
            (59, 10, 45),
            (1, 2, 10),
        ],
    )
    def test_move_radius(self, x: int, y: int, move_radius: int):
        WIDTH, HEIGHT = 1000, 1000

        point = tm.vec2(x, y)
        new_point = self.move_random(
            point=point, move_r=move_radius, plate_w=WIDTH, plate_h=HEIGHT
        )

        assert (
            new_point[0] - point[0] <= move_radius
            and new_point[1] - point[1] <= move_radius
        )

        new_point = self.move_line(
            point=point,
            old_point=new_point,
            move_r=move_radius,
            plate_w=WIDTH,
            plate_h=HEIGHT,
        )

        assert (
            new_point[0] - point[0] <= move_radius
            and new_point[1] - point[1] <= move_radius
        )
