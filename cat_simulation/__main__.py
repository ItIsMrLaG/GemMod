import taichi as ti

from cat_simulation.constants import *
from cat_simulation.cat import Cat, init_cat_env
from cat_simulation.grid import setup_grid, update_statuses

F_CATS = Cat.field(shape=(CATS_N,))
GUI = ti.GUI("cat simulation", res=(PLATE_WIDTH, PLATE_HEIGHT))


@ti.kernel
def set_cat_init_positions(cats: ti.template()):
    for idx in range(CATS_N):
        cats[idx].init_cat(CAT_RADIUS)


@ti.kernel
def move_cats(cats: ti.template()):
    for idx in range(CATS_N):
        cats[idx].move()


if __name__ == '__main__':
    init_cat_env(RADIUS_0, RADIUS_1, PLATE_WIDTH, PLATE_HEIGHT)
    setup_grid(CATS_N, RADIUS_1, PLATE_WIDTH, PLATE_HEIGHT)
    set_cat_init_positions(F_CATS)
    while GUI.running:
        move_cats(F_CATS)
        update_statuses(F_CATS, EUCLIDEAN_DISTANCE)
        pos = F_CATS.norm_point.to_numpy()
        r = F_CATS.radius.to_numpy()
        c = F_CATS.color.to_numpy()
        GUI.circles(pos, radius=r, color=c)
        GUI.show()
