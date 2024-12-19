from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import taichi as ti

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
    RADIUS_0: float = CAT_RADIUS * 2
    RADIUS_1: float = RADIUS_0 * 3

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
