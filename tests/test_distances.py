import pytest
import taichi as ti
import taichi.math as tm
from catsim.tools import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    get_distance,
)
from catsim.constants import (
    EUCLIDEAN_DISTANCE,
    CHEBYSHEV_DISTANCE,
    MANHATTAN_DISTANCE,
)


@ti.data_oriented
class TestDistanceFunctions:
    @ti.kernel
    def get_euclidean(self, p1: tm.vec2, p2: tm.vec2) -> ti.f32:
        return euclidean_distance(p1, p2)

    @ti.kernel
    def get_manhattan(self, p1: tm.vec2, p2: tm.vec2) -> ti.f32:
        return manhattan_distance(p1, p2)

    @ti.kernel
    def get_chebyshev(self, p1: tm.vec2, p2: tm.vec2) -> ti.f32:
        return chebyshev_distance(p1, p2)

    @ti.kernel
    def get_distance_by_type(
        self, p1: tm.vec2, p2: tm.vec2, distance_type: ti.i32
    ) -> ti.f32:
        return get_distance(p1, p2, distance_type)

    @pytest.mark.parametrize(
        "p1, p2, expected",
        [
            (tm.vec2(0, 0), tm.vec2(3, 4), 5.0),
            (tm.vec2(1, 1), tm.vec2(4, 5), 5.0),
            (tm.vec2(-2, -3), tm.vec2(1, 1), 5.0),
            (tm.vec2(-1.5, 0), tm.vec2(1.0, 0), 2.5),
            (tm.vec2(-1e5, -1e5), tm.vec2(-1e5, -1e5), 0.0),
            (tm.vec2(1000, 2000), tm.vec2(3000, 4000), 2828.4271247),
            (tm.vec2(1e5, 1e5), tm.vec2(-1e4, -1e6), 1105486.3183232),
        ],
    )
    def test_euclidean(self, p1, p2, expected):
        result = self.get_euclidean(p1, p2)
        assert pytest.approx(result, 0.001) == expected

    @pytest.mark.parametrize(
        "p1, p2, expected",
        [
            (tm.vec2(0, 0), tm.vec2(3, 4), 7.0),
            (tm.vec2(-2, -3), tm.vec2(1, 1), 7.0),
            (tm.vec2(-5, -10), tm.vec2(-6, -3), 8.0),
            (tm.vec2(-1, -2), tm.vec2(5, -7), 11.0),
        ],
    )
    def test_manhattan(self, p1, p2, expected):
        result = self.get_manhattan(p1, p2)
        assert pytest.approx(result, 0.001) == expected

    @pytest.mark.parametrize(
        "p1, p2, expected",
        [
            (tm.vec2(0, 0), tm.vec2(3, 4), 4.0),
            (tm.vec2(-2, -3), tm.vec2(1, -5), 3.0),
            (tm.vec2(-2, -3), tm.vec2(1, -7), 4.0),
            (tm.vec2(-5, -10), tm.vec2(-6, -3), 7.0),
        ],
    )
    def test_chebyshev(self, p1, p2, expected):
        result = self.get_chebyshev(p1, p2)
        assert pytest.approx(result, 0.001) == expected

    @pytest.mark.parametrize(
        "dist_type, p1, p2, expected",
        [
            (EUCLIDEAN_DISTANCE, tm.vec2(0, 0), tm.vec2(3, 4), 5.0),
            (MANHATTAN_DISTANCE, tm.vec2(0, 0), tm.vec2(3, 4), 7.0),
            (CHEBYSHEV_DISTANCE, tm.vec2(0, 0), tm.vec2(3, 4), 4.0),
        ],
    )
    def test_distance(self, dist_type, p1, p2, expected):
        result = self.get_distance_by_type(p1, p2, dist_type)
        assert pytest.approx(result, 0.001) == expected
