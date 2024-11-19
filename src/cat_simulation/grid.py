import math
from typing import Any

import taichi as ti

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


def setup_grid(cat_n: ti.i32, r1: ti.i32, width: ti.i32, height: ti.i32):
    global _CATS_N, _RADIUS_1, _PLATE_WIDTH, _PLATE_HEIGHT
    _CATS_N = cat_n
    _RADIUS_1 = r1
    _PLATE_WIDTH = width
    _PLATE_HEIGHT = height

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


@ti.func
def init_cell_storage(cats: ti.template()):
    _F_CAT_PER_CELL.fill(0)

    for idx in range(_CATS_N):
        cell_idx = ti.floor(cats[idx].point / _CELL_SZ, ti.i32)
        ti.atomic_add(_F_CAT_PER_CELL[cell_idx], 1)

    for col in range(_GRID_COL_N):
        _sum = 0
        for row in range(_GRID_ROW_N):
            _sum += _F_CAT_PER_CELL[col, row]
        _F_COLUMN_SUM[col] = _sum

    # Compute NOT in parallel
    _F_PREFIX_SUM[0, 0] = 0
    ti.loop_config(serialize=True)
    for col in range(1, _GRID_COL_N):
        _F_PREFIX_SUM[col, 0] = _F_PREFIX_SUM[col - 1, 0] + _F_COLUMN_SUM[col - 1]

    for col in range(_GRID_COL_N):
        for row in range(_GRID_ROW_N):
            if row == 0:
                _F_PREFIX_SUM[col, row] += _F_CAT_PER_CELL[col, row]
            else:
                _F_PREFIX_SUM[col, row] = (
                    _F_PREFIX_SUM[col, row - 1] + _F_CAT_PER_CELL[col, row]
                )

            cell_lin_idx = col * _GRID_ROW_N + row
            head = _F_PREFIX_SUM[col, row] - _F_CAT_PER_CELL[col, row]
            _F_CELL_HEADS[cell_lin_idx] = head
            _F_CELL_CUR[cell_lin_idx] = head
    _F_CELL_HEADS[_CELL_N] = _F_PREFIX_SUM[_GRID_COL_N - 1, _GRID_ROW_N - 1]

    for idx in range(_CATS_N):
        cell_idx = ti.floor(cats[idx].point / _CELL_SZ, ti.i32)
        cell_lin_idx = cell_idx[0] * _GRID_ROW_N + cell_idx[1]
        cat_cell_location = ti.atomic_add(_F_CELL_CUR[cell_lin_idx], 1)
        _F_CELL_STORAGE[cat_cell_location] = idx


@ti.kernel
def update_statuses(cats: ti.template(), distance_type: ti.i32):
    init_cell_storage(cats)

    for idx1 in range(_CATS_N):
        cell_idx = ti.floor(cats[idx1].point / _CELL_SZ, ti.i32)

        x_begin = max(cell_idx[0] - 1, 0)
        x_end = min(cell_idx[0] + 2, _GRID_COL_N)
        y_begin = max(cell_idx[1] - 1, 0)
        y_end = min(cell_idx[1] + 2, _GRID_ROW_N)

        for cell_col in range(x_begin, x_end):
            for cell_row in range(y_begin, y_end):
                neighbour_lin_idx = cell_col * _GRID_ROW_N + cell_row
                for _idx2 in range(
                    _F_CELL_HEADS[neighbour_lin_idx],
                    _F_CELL_HEADS[neighbour_lin_idx + 1],
                ):
                    idx2 = _F_CELL_STORAGE[_idx2]

                    if idx1 != idx2:
                        cats[idx1].fight_with(cats[idx2], distance_type)
