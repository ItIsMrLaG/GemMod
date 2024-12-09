import taichi as ti

EPS = 1

# ----- PATTERNS SETTINGS ----- #
MOVE_PATTERN_RANDOM_ID = 0
MOVE_PATTERN_LINE_ID = 1
MOVE_PATTERN_PHIS_ID = 2

EUCLIDEAN_DISTANCE = 0
MANHATTAN_DISTANCE = 1
CHEBYSHEV_DISTANCE = 2

# ----- PROBABILISTIC INTERACTION ----- #
ENABLE_PROB_INTER = 1
DISABLE_PROB_INTER = 0

# ----- BORDER INTERACTION ----- #
ENABLE_BORDER_INTER = 1
DISABLE_BORDER_INTER = 0

# ----- INTERACTION LEVELS ----- #
INTERACTION_LEVEL_0 = 2
INTERACTION_LEVEL_1 = 1
INTERACTION_NO = 0

# ----- COLORS ----- #
RED_COLOR = 0xED553B
YELLOW_COLOR = 0xFFFF00
GREEN_COLOR = 0x34C924

S_RED_COLOR = 0xF1948A
S_YELLOW_COLOR = 0xFAD7A0
S_GREEN_COLOR = 0x45B39D

EMPTY_COLOR = 0x90D8D0
COLOR_1 = 0xF5B7B1
COLOR_2 = 0xF9E79F
COLOR_3 = 0xA9DFBF

# ----- VISIBILITY STATUSES ----- #
NEVER_APPEARED: ti.i8 = 0
VISIBLE: ti.i8 = 1
INVISIBLE: ti.i8 = 2
ALWAYS_VISIBLE: ti.i8 = 3
ALWAYS_INVISIBLE: ti.i8 = 4

# ----- SPAWN SEED ----- #
RANDOM_SEED = 0
