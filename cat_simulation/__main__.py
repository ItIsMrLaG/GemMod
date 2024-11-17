import taichi as ti

from cat_simulation.constants import *
from cat_simulation.cat import Cat


F_CATS = Cat.field(shape=(N,))

PIXELS = ti.field(dtype=float, shape=(PLATE_WIDTH, PLATE_HEIGHT))
WHITE_COLOR = 0
BLACK_COLOR = 1
GREY_COLOR = 0.5

GUI = ti.GUI("Julia Set", res=(PLATE_WIDTH, PLATE_HEIGHT))


@ti.kernel
def set_cat_ini_positions():
    for idx in range(N):
        F_CATS[idx].init_pos()


@ti.kernel
def clear_pixels():
    for i, j in PIXELS:
        PIXELS[i, j] = 1


@ti.kernel
def paint_cats():
    for idx in range(N):
        F_CATS[idx].move()
        F_CATS[idx].update_pixel()
        F_CATS[idx].paint_pixel(PIXELS)


if __name__ == '__main__':
    set_cat_ini_positions()
    clear_pixels()
    while GUI.running:
        clear_pixels()
        paint_cats()
        GUI.set_image(PIXELS)
        GUI.show()
