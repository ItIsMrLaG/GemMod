import pytest
import taichi as ti
import taichi.math as tm
from catsim.spawner import Spawner

from helper import (
    init_cats_with_custom_points,
    primitive_update_states,
)

import catsim.constants as const
from catsim.cat import Cat, init_cat_env
from catsim.constants import (
    DISABLE_PROB_INTER,
    MOVE_PATTERN_RANDOM_ID,
    DISABLE_BORDER_INTER,
)
from catsim.grid import setup_grid, update_statuses


@ti.data_oriented
class TestUpdateStatus:
    def test_concrete_case(self):
        N, RADIUS, R0, R1, WIDTH, HEIGHT = 5, 1, 2, 8, 50, 50

        init_cat_env(
            move_radius=R0,
            r0=R0,
            r1=R1,
            width=WIDTH,
            height=HEIGHT,
            move_pattern=MOVE_PATTERN_RANDOM_ID,
            prob_inter=DISABLE_PROB_INTER,
            border_inter=DISABLE_BORDER_INTER,
            distance_type=const.EUCLIDEAN_DISTANCE,
        )

        points = ti.Vector.field(n=2, dtype=float, shape=(N,))
        points[0] = tm.vec2(0.0, 0.0)
        points[1] = tm.vec2(2.0, 0.0)
        points[2] = tm.vec2(8.0, 0.0)
        points[3] = tm.vec2(10.0, 10.0)
        points[4] = tm.vec2(50.0, 50.0)

        cats = Cat.field(shape=(N,))
        init_cats_with_custom_points(
            n=N,
            radius=RADIUS,
            cats=cats,
            points=points,
        )

        setup_grid(N, float(R1), float(WIDTH), float(HEIGHT))
        update_statuses(cats)

        expected_statuses = [
            const.INTERACTION_LEVEL_0,
            const.INTERACTION_LEVEL_0,
            const.INTERACTION_LEVEL_1,
            const.INTERACTION_NO,
            const.INTERACTION_NO,
        ]

        for i in range(N):
            assert cats[i].status == expected_statuses[i]

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

        init_cat_env(
            move_radius=R0,
            r0=R0,
            r1=R1,
            width=WIDTH,
            height=HEIGHT,
            move_pattern=MOVE_PATTERN_RANDOM_ID,
            prob_inter=DISABLE_PROB_INTER,
            border_inter=DISABLE_BORDER_INTER,
            distance_type=distance_type,
        )

        cats = Cat.field(shape=(N,))
        spawner = Spawner(
            width=WIDTH,
            height=HEIGHT,
            cats_n=N,
            cat_radius=RADIUS,
            cats=cats,
        )
        spawner.set_cat_init_positions(N, const.RANDOM_SEED)

        expected_statuses = ti.ndarray(dtype=ti.i32, shape=(N,))
        primitive_update_states(N, cats, expected_statuses, distance_type, R0, R1)

        update_statuses(cats)

        for i in range(N):
            assert cats[i].status == expected_statuses[i]
