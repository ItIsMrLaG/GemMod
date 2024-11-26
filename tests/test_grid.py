import pytest
import taichi as ti
import taichi.math as tm
from helper import (
    init_cats,
    init_cats_with_custom_points,
    init_consts,
    primitive_update_states,
    set_cat_init_positions,
)

import catsim.constants as const
from catsim.grid import setup_grid, update_statuses


@ti.data_oriented
class TestUpdateStatus:
    def test_concrete_case(self):
        F_CATS, F_POINTS, RADIUS, N = init_consts(5, 1, 2, 8, 50, 50)
        F_POINTS[0] = tm.vec2(0.0, 0.0)
        F_POINTS[1] = tm.vec2(2.0, 0.0)
        F_POINTS[2] = tm.vec2(8.0, 0.0)
        F_POINTS[3] = tm.vec2(10.0, 10.0)
        F_POINTS[4] = tm.vec2(50.0, 50.0)
        init_cats_with_custom_points(N, F_CATS, F_POINTS, RADIUS)
        setup_grid(N, 8.0, 50.0, 50.0)
        update_statuses(F_CATS, const.EUCLIDEAN_DISTANCE)

        expected_statuses = [
            const.INTERACTION_LEVEL_0,
            const.INTERACTION_LEVEL_0,
            const.INTERACTION_LEVEL_1,
            const.INTERACTION_NO,
            const.INTERACTION_NO,
        ]
        expected_colors = [
            const.RED_COLOR,
            const.RED_COLOR,
            const.YELLOW_COLOR,
            const.GREEN_COLOR,
            const.GREEN_COLOR,
        ]

        for i in range(N):
            assert F_CATS[i].status == expected_statuses[i]
            assert F_CATS[i].color == expected_colors[i]

    @pytest.mark.parametrize(
        "N, R0, R1, RADIUS, WIDTH, HEIGHT, distance_type",
        [
            (10, 2, 8, 1, 100, 100, const.EUCLIDEAN_DISTANCE),
            (100, 2, 8, 1, 500, 500, const.EUCLIDEAN_DISTANCE),
            (1000, 2, 8, 1, 500, 500, const.EUCLIDEAN_DISTANCE),
            (10000, 2, 8, 1, 1000, 1000, const.EUCLIDEAN_DISTANCE),
            (10000, 2, 8, 1, 1500, 2000, const.EUCLIDEAN_DISTANCE),
            (50000, 2, 8, 1, 1000, 1000, const.EUCLIDEAN_DISTANCE),
        ],
    )
    def test_primitive_func(self, N, R0, R1, RADIUS, WIDTH, HEIGHT, distance_type):
        setup_grid(cat_n=N, r1=R1, width=WIDTH, height=HEIGHT)
        F_CATS = init_cats(N, R0, R1, WIDTH, HEIGHT)
        set_cat_init_positions(N, RADIUS, F_CATS)

        expected_statuses = ti.ndarray(dtype=ti.i32, shape=(N,))
        primitive_update_states(N, F_CATS, expected_statuses, distance_type, R0, R1)
        update_statuses(F_CATS, distance_type)
        for i in range(N):
            assert F_CATS[i].status == expected_statuses[i]
