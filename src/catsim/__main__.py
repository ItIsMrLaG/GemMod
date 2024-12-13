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

POINTS = tm.vec2.field(shape=(cfg.CATS_N,))
COLORS = ti.field(ti.i32, shape=(cfg.CATS_N,))
RADIUSES = ti.field(ti.f32, shape=(cfg.CATS_N,))

LINES1_BEGIN = tm.vec2.field(shape=(cfg.CATS_N,))
LINES1_END = tm.vec2.field(shape=(cfg.CATS_N,))

LINES2_BEGIN = tm.vec2.field(shape=(cfg.CATS_N,))
LINES2_END = tm.vec2.field(shape=(cfg.CATS_N,))

LINE_LENGTH = tm.vec2([cfg.RADIUS_1 / cfg.PLATE_WIDTH, cfg.RADIUS_1 / cfg.PLATE_HEIGHT])


@ti.kernel
def arrange_visuals(
    cats: ti.template(), observer_exists: bool
) -> tuple[ti.i32, ti.i32]:
    cat_idx: ti.i32 = 0
    line_idx: ti.i32 = 0

    for idx in range(cfg.CATS_N):
        cat = cats[idx]
        if cat.visibility_status == VISIBLE or cat.visibility_status == ALWAYS_VISIBLE:
            always_visible = cat.visibility_status == ALWAYS_VISIBLE

            if always_visible:
                LINES1_BEGIN[line_idx] = cat.norm_point
                LINES2_BEGIN[line_idx] = cat.norm_point

                LINES1_END[line_idx][0] = cat.norm_point[0] + LINE_LENGTH[0] * tm.cos(
                    cat.observable_angle[0]
                )
                LINES1_END[line_idx][1] = cat.norm_point[1] + LINE_LENGTH[1] * tm.sin(
                    cat.observable_angle[0]
                )

                LINES2_END[line_idx][0] = cat.norm_point[0] + LINE_LENGTH[0] * tm.cos(
                    cat.observable_angle[1]
                )
                LINES2_END[line_idx][1] = cat.norm_point[1] + LINE_LENGTH[1] * tm.sin(
                    cat.observable_angle[1]
                )

                ti.atomic_add(line_idx, 1)

            POINTS[cat_idx] = cat.norm_point
            RADIUSES[cat_idx] = cat.radius
            COLORS[cat_idx] = (
                cfg.COLORS_FAV[cat.status]
                if always_visible
                else cfg.COLORS[cat.status]
                if cat.observed or (not observer_exists)
                else cfg.COLORS_IGN[cat.status]
            )

            ti.atomic_add(cat_idx, 1)

    return cat_idx, line_idx


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].move()


def mainloop(cats: ti.template(), fav_cat_exists: bool, gui: ti.GUI):
    while gui.running:
        move_cats(cats)
        update_statuses(cats)

        count_cats, count_lines = arrange_visuals(cats, fav_cat_exists)
        if count_cats == 0:
            continue

        if count_lines != 0:
            gui.lines(
                begin=LINES1_BEGIN.to_numpy()[:count_lines],
                end=LINES1_END.to_numpy()[:count_lines],
            )
            gui.lines(
                begin=LINES2_BEGIN.to_numpy()[:count_lines],
                end=LINES2_END.to_numpy()[:count_lines],
            )

        gui.circles(
            pos=POINTS.to_numpy()[:count_cats],
            radius=RADIUSES.to_numpy()[:count_cats],
            color=COLORS.to_numpy()[:count_cats],
        )

        gui.show()


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
        raise ValueError("Radius must be > 0")

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

    gui = ti.GUI("catsim", res=(cfg.PLATE_WIDTH, cfg.PLATE_HEIGHT))

    spawner.set_cat_init_positions(cfg.CATS_N, 0)

    fav_cat_exists = False
    if 0 <= cfg.FAVORITE_CAT_IDX < cfg.CATS_N:
        spawner.set_always_visible_cat(cfg.FAVORITE_CAT_IDX, 0)
        fav_cat_exists = True

    mainloop(cats, fav_cat_exists, gui)


if __name__ == "__main__":
    main()
