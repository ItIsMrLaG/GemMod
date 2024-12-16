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
ANGLE_SHIFT = tm.vec2(
    [
        tm.asin(cfg.CAT_RADIUS / cfg.PLATE_WIDTH / LINE_LENGTH[0] * 1.5),
        tm.asin(cfg.CAT_RADIUS / cfg.PLATE_HEIGHT / LINE_LENGTH[1] * 1.5),
    ]
)


@ti.kernel
def arrange_visuals(cats: ti.template()) -> tuple[ti.i32, ti.i32]:
    render_idx: ti.i32 = 0

    for cat_idx in range(cfg.FAV_CATS_AMOUNT, cfg.CATS_N):
        # skip favorite cats for now to render them later
        # so they appear "above" others

        cat = cats[cat_idx]
        if cat.visibility_status == VISIBLE:
            POINTS[render_idx] = cat.norm_point
            RADIUSES[render_idx] = cat.radius
            COLORS[render_idx] = (
                cfg.COLORS[cat.status]
                if cat.observed or (not cfg.FAV_CATS_OBSERVING)
                else cfg.COLORS_IGN[cat.status]
            )

            ti.atomic_add(render_idx, 1)

    line_idx: ti.i32 = 0
    for cat_idx in range(cfg.FAV_CATS_AMOUNT):
        cat = cats[cat_idx]
        assert cat.visibility_status == ALWAYS_VISIBLE

        POINTS[render_idx] = cat.norm_point
        RADIUSES[render_idx] = cat.radius
        COLORS[render_idx] = cfg.COLORS_FAV[cat.status]

        ti.atomic_add(render_idx, 1)

        if cfg.FAV_CATS_OBSERVING:
            LINES1_BEGIN[line_idx] = cat.norm_point
            LINES2_BEGIN[line_idx] = cat.norm_point

            LINES1_END[line_idx][0] = cat.norm_point[0] + LINE_LENGTH[0] * tm.cos(
                cat.observable_angle[0] + ANGLE_SHIFT[0]
            )
            LINES1_END[line_idx][1] = cat.norm_point[1] + LINE_LENGTH[1] * tm.sin(
                cat.observable_angle[0] + ANGLE_SHIFT[1]
            )

            LINES2_END[line_idx][0] = cat.norm_point[0] + LINE_LENGTH[0] * tm.cos(
                cat.observable_angle[1] - ANGLE_SHIFT[0]
            )
            LINES2_END[line_idx][1] = cat.norm_point[1] + LINE_LENGTH[1] * tm.sin(
                cat.observable_angle[1] - ANGLE_SHIFT[1]
            )

            ti.atomic_add(line_idx, 1)

    return render_idx, line_idx


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(cfg.CATS_N):
        cats[idx].move()


def mainloop(cats: ti.template(), gui: ti.GUI):
    while gui.running:
        move_cats(cats)
        update_statuses(cats)

        count_cats, count_lines = arrange_visuals(cats)

        if count_lines != 0:
            gui.lines(
                begin=LINES1_BEGIN.to_numpy()[:count_lines],
                end=LINES1_END.to_numpy()[:count_lines],
                radius=cfg.LINES_RADIUS,
            )
            gui.lines(
                begin=LINES2_BEGIN.to_numpy()[:count_lines],
                end=LINES2_END.to_numpy()[:count_lines],
                radius=cfg.LINES_RADIUS,
            )

        if count_cats != 0:
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

    if not (0 <= cfg.FAV_CATS_AMOUNT <= cfg.CATS_N):
        raise ValueError("Invalid amount of favorite cats")

    if not (tm.pi / 8 <= cfg.OBSERVABLE_ANGLE_SPAN <= tm.pi / 2):
        raise ValueError("Invalid observable angle span")

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
    for idx in range(cfg.FAV_CATS_AMOUNT):
        spawner.set_always_visible_cat(idx, 0)

    mainloop(cats, gui)


if __name__ == "__main__":
    main()
