from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

const = importlib.import_module("constants")


@dataclass
class Config:
    # ----- GENERAL -----
    PLATE_WIDTH: int = 1500
    PLATE_HEIGHT: int = 1000
    PLATE_H_MIN: int = 0
    PLATE_W_MIN: int = 0
    CATS_N: int = 500000

    # ----- CAT -----
    CAT_RADIUS: float = 0.4
    MOVE_RADIUS: float = CAT_RADIUS * 1.5
    RADIUS_0: float = CAT_RADIUS * 1.5
    RADIUS_1: float = RADIUS_0 * 1.5

    # ----- PATTERNS -----
    MOVE_PATTERN_ID: int = const.MOVE_PATTERN_PHIS_ID
    DISTANCE: int = const.EUCLIDEAN_DISTANCE

    # ----- INTERACTIONS -----
    PROB_INTERACTION: int = const.DISABLE_PROB_INTER
    BORDER_INTER: int = const.ENABLE_BORDER_INTER

    # FAVORITE CAT SETTINGS
    FAVORITE_CAT_IDX: int = 1
    SPAWN_SEED: int = const.RANDOM_SEED

    @staticmethod
    def generate_from_json(json_name: Path) -> Config:
        data: dict
        with open(json_name, "r") as file:
            data = json.load(file)

        _cfg_data = {}

        keys_with_const = {"MOVE_PATTERN_ID", "DISTANCE", "PROB_INTERACTION", "BORDER_INTER", "SPAWN_SEED"}

        for key, value in data.items():
            if key in keys_with_const:
                try:
                    _cfg_data[key] = getattr(const, value)
                except AttributeError:
                    print(f"WARNING: Attribute '{value}' not found in 'const'. "
                          f"Therefore the default value for field {key} was used.")
            else:
                _cfg_data[key] = value

        return Config(**_cfg_data)

