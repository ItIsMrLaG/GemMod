import taichi as ti
from taichi.lang.argpack import np

import catsim.config as cfg
from catsim.cat import Cat, init_cat_env
from catsim.constants import INTERACTION_LEVEL_0, INTERACTION_LEVEL_1, INTERACTION_NO
from catsim.grid import setup_grid, update_statuses


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].move()


def status_to_color(status: ti.i32):
    if status == INTERACTION_NO:
        return cfg.COLOR_LEVEL_NO

    if status == INTERACTION_LEVEL_0:
        return cfg.COLOR_LEVEL_0

    if status == INTERACTION_LEVEL_1:
        return cfg.COLOR_LEVEL_1

    raise ValueError(status)


to_color_vf = np.vectorize(status_to_color)


def mainloop(cats: ti.template()):
    GUI = ti.GUI("cat simulation", res=(cfg.PLATE_WIDTH, cfg.PLATE_HEIGHT))

    while GUI.running:
        move_cats(cats)
        update_statuses(cats, cfg.DISTANCE)

        GUI.circles(
            pos=cats.norm_point.to_numpy(),
            radius=cats.radius.to_numpy(),
            color=to_color_vf(cats.status.to_numpy()),
        )
        GUI.show()


@ti.kernel
def set_cat_init_positions(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].init_cat(cfg.CAT_RADIUS)


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
    )

    setup_grid(
        cat_n=cfg.CATS_N,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
    )

    cats = Cat.field(shape=(cfg.CATS_N,))
    set_cat_init_positions(cats)

    mainloop(cats)


if __name__ == "__main__":
    main()
