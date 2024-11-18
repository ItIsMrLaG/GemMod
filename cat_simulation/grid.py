import math
from typing import Any

import taichi as ti
# from cat_simulation.constants import *

__all__ = [
    "setup_grid",
    "update_statuses",
]

# global settings
_CATS_N: ti.i32
_RADIUS_1: ti.i32
_PLATE_WIDTH: ti.i32
_PLATE_HEIGHT: ti.i32

_CELL_N: ti.i32
_CELL_SZ: ti.i32
_GRID_COL_N: ti.i32
_GRID_ROW_N: ti.i32

"""
contains Cats ids:
    - size := cats_n
    - each range contains cat_ids only from one Cell
        - Head_i and Tail_i of the range from F_CELL_HEADS
"""
_F_CELL_STORAGE: Any

""" 
contains Head for each cell:
    - size := cell_n + 1
    - if element_i is Head_i then element_{i+1} is Tail_i
    - { j } from [Head_i ; Tail_i) 
        ==> cat_id := F_CELL_STORAGE[j] 
        ==> cat := F_CATS[cat_id] 
        ==> cat in Cell_i   
"""
_F_CELL_HEADS: Any


# (internal) used to fill _F_CELL_STORAGE
_F_CELL_CUR: Any
_F_COLUMN_SUM: Any
_F_PREFIX_SUM: Any
_F_CAT_PER_CELL: Any


def setup_grid(
    cat_n: ti.i32,
    radius1: ti.i32,
    plate_width: ti.i32,
    plate_height: ti.i32
):
    global _CATS_N, _RADIUS_1, _PLATE_WIDTH, _PLATE_HEIGHT
    _CATS_N = cat_n
    _RADIUS_1 = radius1
    _PLATE_WIDTH = plate_width
    _PLATE_HEIGHT = plate_height

    global _CELL_N, _GRID_COL_N, _GRID_ROW_N, _CELL_SZ
    _CELL_SZ = _RADIUS_1
    _GRID_COL_N = math.ceil(_PLATE_WIDTH / _CELL_SZ)
    _GRID_ROW_N = math.ceil(_PLATE_HEIGHT / _CELL_SZ)
    _CELL_N = _GRID_COL_N * _GRID_ROW_N

    global _F_CELL_HEADS, _F_CELL_STORAGE
    _F_CELL_STORAGE = ti.field(dtype=ti.i32, shape=(_CATS_N,))
    _F_CELL_HEADS = ti.field(dtype=ti.i32, shape=(_CELL_N + 1,))

    global _F_CELL_CUR, _F_CAT_PER_CELL, _F_COLUMN_SUM, _F_PREFIX_SUM
    _F_CAT_PER_CELL = ti.field(dtype=ti.i32, shape=(_GRID_COL_N, _GRID_ROW_N))
    _F_COLUMN_SUM = ti.field(dtype=ti.i32, shape=(_GRID_COL_N,))
    _F_PREFIX_SUM = ti.field(dtype=ti.i32, shape=(_GRID_COL_N, _GRID_ROW_N))
    _F_CELL_CUR = ti.field(dtype=ti.i32, shape=(_CELL_N,))
