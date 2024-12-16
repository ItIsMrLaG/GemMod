import argparse
from pathlib import Path

import taichi as ti
import taichi.math as tm
from catsim.spawner import Spawner

from catsim.config import Config
from catsim.cat import Cat, init_cat_env
from catsim.constants import (
    COLOR_LEVEL_NO,
    COLOR_LEVEL_0,
    COLOR_LEVEL_1,
    FAV_COLOR_LEVEL_NO,
    FAV_COLOR_LEVEL_0,
    FAV_COLOR_LEVEL_1,
    INTERACTION_LEVEL_0,
    INTERACTION_LEVEL_1,
    INTERACTION_NO,
    VISIBLE,
    ALWAYS_VISIBLE,
)
from catsim.grid import setup_grid, update_statuses


@ti.func
def status_to_color(status: ti.i32) -> ti.i32:
    color = INTERACTION_NO

    if status == INTERACTION_NO:
        color = COLOR_LEVEL_NO

    if status == INTERACTION_LEVEL_0:
        color = COLOR_LEVEL_0

    if status == INTERACTION_LEVEL_1:
        color = COLOR_LEVEL_1

    return color


@ti.func
def fav_status_to_color(status: ti.i32) -> ti.i32:
    color = INTERACTION_NO

    if status == INTERACTION_NO:
        color = FAV_COLOR_LEVEL_NO

    if status == INTERACTION_LEVEL_0:
        color = FAV_COLOR_LEVEL_0

    if status == INTERACTION_LEVEL_1:
        color = FAV_COLOR_LEVEL_1

    return color


@ti.kernel
def update_dcats(cats_n: ti.i32, cats: ti.template(), points: ti.template(), radii: ti.template(), colors: ti.template()) -> ti.i32:
    counter: ti.i32 = 0
    for idx in range(cats_n):
        if (
            cats[idx].visibility_status == VISIBLE
            or cats[idx].visibility_status == ALWAYS_VISIBLE
        ):
            addr = ti.atomic_add(counter, 1)
            points[addr] = cats[idx].norm_point
            radii[addr] = cats[idx].radius

            if cats[idx].visibility_status == ALWAYS_VISIBLE:
                colors[addr] = fav_status_to_color(cats[idx].status)
            else:
                colors[addr] = status_to_color(cats[idx].status)

    return counter


@ti.kernel
def move_cats(CATS_N: ti.i32, cats: ti.template()):
    for idx in range(CATS_N):
        cats[idx].move()


def mainloop(cfg: Config, cats: ti.template(), spawner: Spawner, points: ti.template(), radii: ti. template(), colors: ti.template()):
    GUI = ti.GUI("catsim", res=(cfg.PLATE_WIDTH, cfg.PLATE_HEIGHT))

    favorite_cat_flag = 0 <= cfg.FAVORITE_CAT_IDX < cfg.CATS_N
    spawner.set_cat_init_positions(cfg.CATS_N, cfg.SPAWN_SEED)

    if favorite_cat_flag:
        spawner.set_always_visible_cat(cfg.FAVORITE_CAT_IDX, cfg.SPAWN_SEED)

    while GUI.running:
        move_cats(cfg.CATS_N, cats)

        update_statuses(cats)

        counter = update_dcats(cfg.CATS_N, cats, points, radii, colors)
        if counter == 0:
            continue

        GUI.circles(
            pos=points.to_numpy()[0:counter],
            radius=radii.to_numpy()[0:counter],
            color=colors.to_numpy()[0:counter],
        )

        GUI.show()


def validate_config(cfg):
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = Config.generate_from_json(Path(args.config_file))
    validate_config(cfg)

    points = tm.vec2.field(shape=(cfg.CATS_N,))
    colors = ti.field(ti.i32, shape=(cfg.CATS_N,))
    radii = ti.field(ti.f32, shape=(cfg.CATS_N,))

    init_cat_env(
        move_radius=cfg.MOVE_RADIUS,
        r0=cfg.RADIUS_0,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        move_pattern=cfg.MOVE_PATTERN_ID,
        prob_inter=cfg.PROB_INTERACTION,
        border_inter=cfg.BORDER_INTER,
        distance_type=cfg.DISTANCE,
    )

    setup_grid(
        cat_n=cfg.CATS_N,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        favorite_cat_idx=cfg.FAVORITE_CAT_IDX,
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
    mainloop(cfg, cats, spawner, points, radii, colors)


if __name__ == "__main__":
    main()
