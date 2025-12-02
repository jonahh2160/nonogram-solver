import math
from pathlib import Path

# Square board configuration
BOARD_SIZE = 10
BOARD_AREA = BOARD_SIZE **2
CLUE_LENGTH = math.ceil(BOARD_SIZE / 2) # Maximum number of clues per row/column
# |--> 5 in the case of a 10x10 board

# File pathing
PUZZLE_FOLDER = Path("..") / "puzzles"
X_TRAIN = PUZZLE_FOLDER / "x_train_dataset.npz"
Y_TRAIN = PUZZLE_FOLDER / "y_train_dataset.npz"
X_TEST = PUZZLE_FOLDER / "x_test_dataset.npz"
Y_TEST = PUZZLE_FOLDER / "y_test_dataset.npz"