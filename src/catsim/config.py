from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import taichi as ti
import taichi.math as tm

enums = importlib.import_module("enums")


@dataclass
class Config:
    # ----- GENERAL -----
    PLATE_WIDTH: int = 1500
    PLATE_HEIGHT: int = 1000
    CATS_N: int = 150

    # ----- CAT -----
    CAT_RADIUS: float = 0.02 * PLATE_HEIGHT
    MOVE_RADIUS: float = CAT_RADIUS * 2
    ACT_MIN_RADIUS: float = CAT_RADIUS * 2
    ACT_MAX_RADIUS: float = ACT_MIN_RADIUS * 3

    # ----- PATTERNS -----
    MOVE_PATTERN_ID: int = enums.MOVE_PATTERN_PHIS
    DISTANCE: int = enums.EUCLIDEAN_DISTANCE

    # ----- INTERACTIONS -----
    PROB_INTERACTION: bool = False
    BORDER_INTERACTION: bool = True

    # ----- FAVORITE CATS ----- #
    # 0 <= amount <= CATS_N
    FAV_CATS_AMOUNT: int = 1
    FAV_CATS_OBSERVING: bool = True
    FAV_CATS_LOGGING: bool = True

    # pi / 8 <= angle <= pi / 2
    OBSERVABLE_ANGLE_SPAN: float = ti.math.pi / 4

    # ----- VISUALISATION ----- #
    LINES_RADIUS: ti.i32 = CAT_RADIUS // 10

    @staticmethod
    def generate_from_json(json_name: Path) -> Config:
        data: dict
        with open(json_name, "r") as file:
            data = json.load(file)

        _cfg_data = {}

        keys_with_const = {"MOVE_PATTERN_ID", "DISTANCE"}

        for key, value in data.items():
            if key in keys_with_const:
                try:
                    _cfg_data[key] = getattr(enums, value)
                except AttributeError:
                    print(
                        f"WARNING: Attribute '{value}' not found in 'const'. "
                        f"Therefore the default value for field {key} was used."
                    )
            else:
                _cfg_data[key] = value

        return Config(**_cfg_data)

    def validate(self):
        if self.PLATE_HEIGHT <= 0 or self.PLATE_WIDTH <= 0:
            raise ValueError("Plate height/width must be > 0")

        if self.CATS_N <= 0:
            raise ValueError("Number of cats must be > 0")

        if not (0 <= self.FAV_CATS_AMOUNT <= self.CATS_N):
            raise ValueError("Invalid amount of favorite cats")

        if not (tm.pi / 8 <= self.OBSERVABLE_ANGLE_SPAN <= tm.pi / 2):
            raise ValueError("Invalid observable angle span")

        if (
            self.CAT_RADIUS <= 0
            or self.MOVE_RADIUS <= 0
            or self.ACT_MIN_RADIUS <= 0
            or self.ACT_MAX_RADIUS <= 0
        ):
            raise ValueError("Radius must be > 0")

        if self.ACT_MAX_RADIUS <= self.ACT_MIN_RADIUS:
            raise ValueError("Radius 1 must be > Radius 0")

        if self.FAV_CATS_AMOUNT > 5:
            raise ValueError("Favorite cats cannot be more than 5")
