import taichi as ti
import catsim.cat as cat
from catsim.constants import (
    DISABLE_PROB_INTER,
    MOVE_PATTERN_RANDOM_ID,
    MOVE_PATTERN_LINE_ID,
)


def init_cat_env(move_radius, r0, r1, width, height, move_pattern, prob_inter):
    cat.init_cat_env(
        move_radius,
        r0,
        r1,
        width,
        height,
        move_pattern,
        prob_inter,
    )


def test_init_cat_env():
    cat.init_cat_env(
        move_radius=10.5,
        r0=2.0,
        r1=8.0,
        width=100,
        height=200,
        move_pattern=MOVE_PATTERN_RANDOM_ID,
        prob_inter=DISABLE_PROB_INTER,
    )
    assert cat._MOVE_RADIUS == 10.5
    assert cat._RADIUS_0 == 2.0
    assert cat._RADIUS_1 == 8.0
    assert cat._PLATE_WIDTH == 100
    assert cat._PLATE_HEIGHT == 200
    assert cat._MOVE_PATTERN == MOVE_PATTERN_RANDOM_ID
    assert cat._PROB_INTER == DISABLE_PROB_INTER


@ti.kernel
def init_cat(cat_r: ti.f32) -> cat.Cat:
    catInstance = cat.Cat()
    catInstance.init_cat(cat_r)
    return catInstance


def test_cat_initialization_with_env():
    cat.init_cat_env(
        move_radius=15.0,
        r0=5.0,
        r1=10.0,
        width=50,
        height=50,
        move_pattern=MOVE_PATTERN_LINE_ID,
        prob_inter=DISABLE_PROB_INTER,
    )
    catInstance = init_cat(3.0)
    for point in catInstance.point:
        assert 0 <= point <= cat._PLATE_WIDTH
    assert catInstance.radius == 3.0
    assert catInstance.move_pattern == MOVE_PATTERN_LINE_ID
