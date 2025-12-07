# nonogram-solver

Various notes

Original dataset by Jose Maria Buades:
https://github.com/josebambu/NonoDataset

WARNING: You MUST unzip the zip files in the 15x15 puzzle folder before training the 15x15 model.

There are as many clues stored in each row/column as there can be clues possible in that size grid (e.g., a 10x10 can store at maximum 5 clues per row: [1,1,1,1,1], so there are 5 clues per row and column in the dataset).
Additionally, these clues are "right-padded", which is to say stored at the TAIL-END of each row/column, NOT at the head (e.g., a 10x10 grid will store a clue of 10 as [0,0,0,0,10]).
These clues are stored first by row (top row to bottom) and then by column (left column to right).

Datasets labeled with the prefix "x" contain only clues (the numbers above and to the side of the grid).
Datasets labeled with the prefix "y" contain the value of each individual square of the grid (0 or 1), left to right, top to bottom.

Baseline model: random (each cell randomly assigned 0 or 1 without regard for clues)
Oracle model: supervised learning using relu and sigmoid

10x10 Total puzzles: 376,368
Training puzzles: 361,094 (~96%)
Testing puzzles: 15,274 (~4%)

~88% accuracy for 10x10
~81% accuracy for 15x15

Run the training programs through a shell with ```python baseline_model_training.py``` or ```python oracle_model_training.py```

Steps to run code:

1. Ensure NumPy and TensorFlow are installed using ```pip install```
2. Unzip ```x_train_dataset.7z``` and ```y_train_dataset.7z``` under ```puzzles/15x15```
3. Set BOARD_SIZE in scripts/config.py to either 10 or 15 depending on the size of puzzle you wish to use
4. cd down to scripts and type ```python ``` and then the name of the model you wish to run (either ```baseline_model_training.py``` or ```oracle_model_trainig.py```)

Files:
puzzles: 10x10, 15x15
x_training_dataset: the clues for the training dataset
y_training_dataset: the answer key for the training dataset
x_test_dataset: the clues for the test dataset
y_test_dataset: the answer key for the test dataset
scripts:
config.py: the settings for the model
baseline_model.py: the baseline model
baseline_model_training.py: the training process for the baseline model
oracle_model.py: the oracle model
oracle_model_training.py: the training process for the oracle_model