# nonogram-solver

Various notes

Original dataset by Jose Maria Buades:
https://github.com/josebambu/NonoDataset

There are as many clues stored in each row/column as there can be clues possible in that size grid (e.g., a 10x10 can store at maximum 5 clues per row: [1,1,1,1,1], so there are 5 clues per row and column in the dataset).
Additionally, these clues are "right-padded", which is to say stored at the TAIL-END of each row/column, NOT at the head (e.g., a 10x10 grid will store a clue of 10 as [0,0,0,0,10]).
These clues are stored first by row (top row to bottom) and then by column (left column to right).

Datasets labeled with the prefix "x" contain only clues (the numbers above and to the side of the grid).
Datasets labeled with the prefix "y" contain the value of each individual square of the grid (0 or 1), left to right, top to bottom.

Baseline model: random (each cell randomly assigned 0 or 1 without regard for clues)
Oracle model: heuristics

Total puzzles: 376,368
Training puzzles: 361,094 (~96%)
Testing puzzles: 15,274 (~4%)