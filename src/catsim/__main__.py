import taichi as ti
import taichi.math as tm

import catsim.config as cfg
from catsim.cat import Cat, init_cat_env
from catsim.enums import (
    ALWAYS_VISIBLE,
    VISIBLE,
)
from catsim.grid import setup_grid, update_statuses
from catsim.spawner import Spawner

DCATS = Cat.field(shape=(cfg.CATS_N,))

POINTS = tm.vec2.field(shape=(cfg.CATS_N,))
COLORS = ti.field(ti.i32, shape=(cfg.CATS_N,))
RADIUSES = ti.field(ti.f32, shape=(cfg.CATS_N,))


@ti.kernel
def update_dcats(cats: ti.template(), favorite_cat_flag: ti.i32) -> ti.i32:
    counter: ti.i32 = 0
    for idx in range(cfg.CATS_N):
        cat = cats[idx]
        if cat.visibility_status == VISIBLE or cat.visibility_status == ALWAYS_VISIBLE:
            addr = ti.atomic_add(counter, 1)
            POINTS[addr] = cat.norm_point
            RADIUSES[addr] = cat.radius

            fav = cat.visibility_status == ALWAYS_VISIBLE
            COLORS[addr] = cfg.COLORS_FAV[cat.status] if fav else cfg.COLORS[cat.status]

    return counter


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].move()


def mainloop(cats: ti.template(), spawner: Spawner):
    GUI = ti.GUI("catsim", res=(cfg.PLATE_WIDTH, cfg.PLATE_HEIGHT))

    favorite_cat_flag = 0 <= cfg.FAVORITE_CAT_IDX < cfg.CATS_N
    spawner.set_cat_init_positions(cfg.CATS_N, 0)

    if favorite_cat_flag:
        spawner.set_always_visible_cat(cfg.FAVORITE_CAT_IDX, 0)

    while GUI.running:
        move_cats(cats)
        update_statuses(cats)

        counter = update_dcats(cats, favorite_cat_flag)
        if counter == 0:
            continue

        GUI.circles(
            pos=POINTS.to_numpy()[0:counter],
            radius=RADIUSES.to_numpy()[0:counter],
            color=COLORS.to_numpy()[0:counter],
        )

        GUI.show()


@ti.kernel
def set_cat_init_positions(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].init_cat(idx, cfg.CAT_RADIUS)


def validate_config():
    if cfg.PLATE_HEIGHT <= 0 or cfg.PLATE_WIDTH <= 0:
        raise ValueError("Plate height/width must be > 0")

    if cfg.CATS_N <= 0:
        raise ValueError("Number of cats must be > 0")

    if (
        cfg.CAT_RADIUS <= 0
        or cfg.MOVE_RADIUS <= 0
        or cfg.RADIUS_0 <= 0
        or cfg.RADIUS_1 <= 0
    ):
        raise ValueError("Radii must be > 0")

    if cfg.RADIUS_1 <= cfg.RADIUS_0:
        raise ValueError("Radius 1 must be > Radius 0")


def main():
    validate_config()

    init_cat_env(
        move_radius=cfg.MOVE_RADIUS,
        r0=cfg.RADIUS_0,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        move_pattern=cfg.MOVE_PATTERN_ID,
        prob_inter=cfg.PROB_INTERACTION,
        border_inter=cfg.BORDER_INTERACTION,
        distance_type=cfg.DISTANCE,
    )

    setup_grid(
        cat_n=cfg.CATS_N,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
    )

    cats = Cat.field(shape=(cfg.CATS_N,))
    spawner = Spawner(
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        cats_n=cfg.CATS_N,
        cat_radius=cfg.CAT_RADIUS,
        cats=cats,
        from_idx=0,
    )
    mainloop(cats, spawner)


if __name__ == "__main__":
    main()
