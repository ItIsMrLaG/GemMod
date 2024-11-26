import taichi as ti

import catsim.config as cfg
from catsim.cat import Cat, init_cat_env
from catsim.grid import setup_grid, update_statuses

F_CATS = Cat.field(shape=(cfg.CATS_N,))
GUI = ti.GUI("cat simulation", res=(cfg.PLATE_WIDTH, cfg.PLATE_HEIGHT))


@ti.kernel
def set_cat_init_positions(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].init_cat(cfg.CAT_RADIUS)


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].move()


if __name__ == "__main__":
    init_cat_env(
        move_radius=cfg.MOVE_RADIUS,
        r0=cfg.RADIUS_0,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        move_pattern=cfg.MOVE_PATTERN_ID,
        l0_color=cfg.COLOR_LEVEL_0,
        l1_color=cfg.COLOR_LEVEL_1,
        lNO_color=cfg.COLOR_LEVEL_NO,
        prob_inter=cfg.PROB_INTERACTION,
    )
    setup_grid(
        cat_n=cfg.CATS_N,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
    )
    set_cat_init_positions(F_CATS)
    while GUI.running:
        move_cats(F_CATS)
        update_statuses(F_CATS, cfg.DISTANCE)
        pos = F_CATS.norm_point.to_numpy()
        r = F_CATS.radius.to_numpy()
        c = F_CATS.color.to_numpy()
        GUI.circles(pos, radius=r, color=c)
        GUI.show()
