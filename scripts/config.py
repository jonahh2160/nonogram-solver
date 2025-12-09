import math
from pathlib import Path

# Square board configuration
BOARD_SIZE = 10 # Can be assigned values of either 10 or 15
FOLDER_NAME = str(BOARD_SIZE) + "x" + str(BOARD_SIZE)
BOARD_AREA = BOARD_SIZE ** 2
CLUE_LENGTH = math.ceil(BOARD_SIZE / 2) # Maximum number of clues per row/column
# |--> 5 in the case of a 10x10 board
CLUE_COUNT = BOARD_SIZE * CLUE_LENGTH * 2 # Total number of clues
BATCH_SIZE = 64
EPOCHS = 24

# File pathing
PUZZLE_FOLDER = Path("..") / "puzzles"  / FOLDER_NAME
X_TRAIN = PUZZLE_FOLDER / "x_train_dataset.npz"
Y_TRAIN = PUZZLE_FOLDER / "y_train_dataset.npz"
X_TEST = PUZZLE_FOLDER / "x_test_dataset.npz"
Y_TEST = PUZZLE_FOLDER / "y_test_dataset.npz"