import pytest
import taichi as ti
import taichi.math as tm

from catsim.tools import is_boundary_point, get_sides_distance
import catsim.constants as const


@ti.data_oriented
class TestInvisibleMotion:
    @ti.kernel
    def is_boundary(self, point: tm.vec2, max_x: ti.f32, max_y: ti.f32) -> ti.i8:
        return is_boundary_point(point, max_x, max_y)

    @ti.kernel
    def get_distance(
            self,
            p1: tm.vec2,
            p2: tm.vec2,
            right_bound: ti.f32,
            upper_bound: ti.f32,
            distance_type: ti.i32,
    ) -> ti.f32:
        return get_sides_distance(p1, p2, right_bound, upper_bound, distance_type)

    """    4
      -----------
    1 |  field  | 3
      -----------
           2         """

    @pytest.mark.parametrize(
        "x, y, max_x, max_y, expected_res",
        [
            # Boundary points
            (0, 0, 100, 100, 2),
            (50, 0, 100, 100, 2),
            (100, 0, 100, 100, 2),
            (0, 100, 100, 100, 4),
            (50, 100, 100, 100, 4),
            (100, 100, 100, 100, 4),
            (0, 50, 100, 100, 1),
            (100, 50, 100, 100, 3),

            # Points inside the field
            (50, 50, 100, 100, 0),
            (25, 25, 100, 100, 0),

            # Points out of bounds
            (-1, 50, 100, 100, 1),
            (101, 50, 100, 100, 3),
            (50, -1, 100, 100, 2),
            (50, 101, 100, 100, 4),

            # Points on boundaries including EPS
            (0 + const.EPS, 50, 100, 100, 1),
            (100 - const.EPS, 50, 100, 100, 3),
            (50, 0 + const.EPS, 100, 100, 2),
            (50, 100 - const.EPS, 100, 100, 4),

            (0 + const.EPS, 0 + const.EPS, 100, 100, 2),
            (100 - const.EPS, 0 + const.EPS, 100, 100, 2),
            (0 + const.EPS, 100 - const.EPS, 100, 100, 4),
            (100 - const.EPS, 100 - const.EPS, 100, 100, 4),

            # In another field size
            (100, 100, 200, 200, 0),
            (0, 100, 500, 500, 1),
            (200, 100, 200, 200, 3),
            (25, 0, 50, 50, 2),
            (25, 1000, 1000, 1000, 4),
        ],
    )
    def test_is_boundary_point(self, x, y, max_x, max_y, expected_res):
        point = tm.vec2(x, y)
        actual_res = self.is_boundary(point, max_x, max_y)
        assert actual_res == expected_res

    @pytest.mark.parametrize(
        "x1, y1, x2, y2, max_x, max_y, distance_type, expected_res",
        [
            # Points on one side
            (0, 0, 0, 50, 100, 100, const.EUCLIDEAN_DISTANCE, 50),
            (0, 25, 0, 75, 100, 100, const.EUCLIDEAN_DISTANCE, 50),

            (0, 0, 50, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 50),
            (25, 0, 75, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 50),

            (100, 0, 100, 50, 100, 100, const.EUCLIDEAN_DISTANCE, 50),
            (100, 25, 100, 75, 100, 100, const.EUCLIDEAN_DISTANCE, 50),

            (0, 100, 50, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 50),
            (25, 100, 75, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 50),

            # Points on opposite sides (left and right sides)
            (0, 0, 100, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 100),
            (0, 50, 100, 50, 100, 100, const.EUCLIDEAN_DISTANCE, 200),
            (0, 100, 100, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 100),

            # Points on opposite sides (top and bottom sides)
            (0, 0, 0, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 100),
            (50, 0, 50, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 200),
            (100, 0, 100, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 100),

            # Points on adjacent sides
            (0, 0, 50, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 150),
            (0, 50, 100, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 150),

            (100, 0, 50, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 150),
            (100, 50, 0, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 150),

            (0, 100, 50, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 150),
            (0, 50, 100, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 150),

            (100, 100, 50, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 150),
            (100, 50, 0, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 150),

            # Corner points
            (0, 0, 100, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 200),
            (0, 100, 100, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 200),

            # In another field size
            (0, 0, 0, 200, 200, 200, const.EUCLIDEAN_DISTANCE, 200),
            (0, 200, 200, 0, 200, 200, const.EUCLIDEAN_DISTANCE, 400),
            (0, 500, 500, 0, 500, 500, const.EUCLIDEAN_DISTANCE, 1000),

            # With fractional coordinates
            (0, 0, 0, 25.5, 100, 100, const.EUCLIDEAN_DISTANCE, 25.5),

            (0, 100, 100.5, 0, 100, 100, const.EUCLIDEAN_DISTANCE, 200.5),

            (0, 0, 50.5, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 150.5),
            (100, 0, 50.5, 100, 100, 100, const.EUCLIDEAN_DISTANCE, 149.5),

        ],
    )
    def test_get_sides_distance(self, x1, y1, x2, y2, max_x, max_y, distance_type, expected_res):
        point1 = tm.vec2(x1, y1)
        point2 = tm.vec2(x2, y2)
        actual_res = self.get_distance(point1, point2, max_x, max_y, distance_type)
        assert actual_res == expected_res
