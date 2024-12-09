from typing import Any

import taichi as ti
import taichi.math as tm
from catsim.constants import (
    NEVER_APPEARED,
    VISIBLE,
    ALWAYS_VISIBLE,
)
from catsim.tools import init_cats


@ti.data_oriented
class Spawner:
    cats: Any  # Cat.field(shape=(cfg.CATS_N,))
    cats_n: ti.i32
    cat_radius: ti.f32
    _from_idx: ti.i32

    def __init__(
        self,
        width: ti.i32,
        height: ti.i32,
        cats_n: ti.i32,
        cat_radius: ti.f32,
        cats: Any,
        from_idx: ti.i32 = 0,
    ):
        self.cats_n = cats_n
        self.cats = cats
        self.cat_radius = cat_radius
        self.max_x = width
        self.max_y = height
        self._from_idx = from_idx
        init_cats(cats_n, cat_radius, cats, NEVER_APPEARED)

    @ti.func
    def _random_spawn_seed(self, idx: ti.i32):
        point = tm.vec2([ti.random() * self.max_x, ti.random() * self.max_y])
        self.cats[idx].set_point(point)

    @ti.func
    def _spawn_seed(self, idx: ti.i32, spawn_tp: ti.i32):
        self._random_spawn_seed(idx)

    @ti.kernel
    def set_always_visible_cat(self, idx: ti.i32, spawn_tp: ti.i32):
        self._spawn_seed(idx, spawn_tp)
        self.cats[idx].visibility_status = ALWAYS_VISIBLE

    @ti.kernel
    def set_cat_init_positions(self, to_idx: ti.i32, spawn_tp: ti.i32):
        _to_idx = tm.min(to_idx, self.cats_n)
        _update_cnt = self._from_idx

        for idx in range(self._from_idx, _to_idx):
            _update_cnt += 1
            self.cats[idx].visibility_status = VISIBLE
            self._spawn_seed(idx, spawn_tp)
