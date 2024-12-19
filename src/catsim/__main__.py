import argparse
from pathlib import Path

import taichi as ti
import taichi.math as tm

from catsim.config import Config
from catsim.cat import Cat, init_cat_env
from catsim.enums import ALWAYS_VISIBLE, VISIBLE, COLORS, COLORS_IGN, COLORS_FAV
from catsim.grid import setup_grid, update_statuses
from catsim.spawner import Spawner

POINTS: tm.vec2.field
CAT_COLORS: ti.field
RADII: ti.field

LINES1_BEGIN: tm.vec2.field
LINES1_END: tm.vec2.field

LINES2_BEGIN: tm.vec2.field
LINES2_END: tm.vec2.field

LINE_LENGTH: tm.vec2
ANGLE_SHIFT: tm.vec2


@ti.kernel
def arrange_visuals(
    cats: ti.template(),
    fav_cats_amount: ti.i8,
    fav_cats_observing: bool,
    cats_n: ti.i32,
) -> tuple[ti.i32, ti.i32]:
    render_idx: ti.i32 = 0

    for cat_idx in range(fav_cats_amount, cats_n):
        # skip favorite cats for now to render them later
        # so they appear "above" others

        cat = cats[cat_idx]
        if cat.visibility_status == VISIBLE:
            POINTS[render_idx] = cat.norm_point
            RADII[render_idx] = cat.radius
            CAT_COLORS[render_idx] = (
                COLORS[cat.status]
                if cat.observed or (not fav_cats_observing)
                else COLORS_IGN[cat.status]
            )

            ti.atomic_add(render_idx, 1)

    line_idx: ti.i32 = 0
    for cat_idx in range(fav_cats_amount):
        cat = cats[cat_idx]
        assert cat.visibility_status == ALWAYS_VISIBLE

        POINTS[render_idx] = cat.norm_point
        RADII[render_idx] = cat.radius
        CAT_COLORS[render_idx] = COLORS_FAV[cat.status]

        ti.atomic_add(render_idx, 1)

        if fav_cats_observing:
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
def move_cats(cats: ti.template(), cats_n: ti.i32):
    for idx in range(cats_n):
        cats[idx].move()


def mainloop(cfg: Config, cats: ti.template(), gui: ti.GUI):
    while gui.running:
        move_cats(cats, cfg.CATS_N)
        update_statuses(cats)

        count_cats, count_lines = arrange_visuals(
            cats, cfg.FAV_CATS_AMOUNT, cfg.FAV_CATS_OBSERVING, cfg.CATS_N
        )

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
                radius=RADII.to_numpy()[:count_cats],
                color=CAT_COLORS.to_numpy()[:count_cats],
            )

        gui.show()


def validate_config(cfg):
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


def init_env(cfg: Config):
    global POINTS, CAT_COLORS, RADII
    POINTS = tm.vec2.field(shape=(cfg.CATS_N,))
    CAT_COLORS = ti.field(ti.i32, shape=(cfg.CATS_N,))
    RADII = ti.field(ti.f32, shape=(cfg.CATS_N,))

    global LINES1_BEGIN, LINES1_END, LINES2_BEGIN, LINES2_END
    LINES1_BEGIN = tm.vec2.field(shape=(cfg.CATS_N,))
    LINES1_END = tm.vec2.field(shape=(cfg.CATS_N,))

    LINES2_BEGIN = tm.vec2.field(shape=(cfg.CATS_N,))
    LINES2_END = tm.vec2.field(shape=(cfg.CATS_N,))

    global LINE_LENGTH, ANGLE_SHIFT
    LINE_LENGTH = tm.vec2(
        [cfg.RADIUS_1 / cfg.PLATE_WIDTH, cfg.RADIUS_1 / cfg.PLATE_HEIGHT]
    )
    ANGLE_SHIFT = tm.vec2(
        [
            tm.asin(cfg.CAT_RADIUS / cfg.PLATE_WIDTH / LINE_LENGTH[0] * 1.5),
            tm.asin(cfg.CAT_RADIUS / cfg.PLATE_HEIGHT / LINE_LENGTH[1] * 1.5),
        ]
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = Config.generate_from_json(Path(args.config_file))
    validate_config(cfg)

    init_env(cfg)

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
        fav_cats_observing=cfg.FAV_CATS_OBSERVING,
        observable_angle_span=cfg.OBSERVABLE_ANGLE_SPAN,
    )

    setup_grid(
        cat_n=cfg.CATS_N,
        r1=cfg.RADIUS_1,
        width=cfg.PLATE_WIDTH,
        height=cfg.PLATE_HEIGHT,
        fav_cats_amount=cfg.FAV_CATS_AMOUNT,
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

    mainloop(cfg, cats, gui)


if __name__ == "__main__":
    main()
