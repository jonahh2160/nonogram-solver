import config
import tensorflow as tf
from tensorflow import keras
from keras import layers

class BaselineModel(keras.Model):
    """Our baseline nonogram solution model, which fills in cells randomly."""

    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(config.BOARD_AREA) # Sets dimensionality of output to board area
    
    def call(self, inputs):
        """
        Generates random nonogram solutions.
        Args:
            inputs: The clues for the puzzle (not used here).
        Returns:
            Random binary solution of shape (batch_size, BOARD_AREA).
        """
        batch_size = tf.shape(inputs)[0] # Number of puzzles in the batch
        random_solution = tf.random.uniform(
            shape = (batch_size, config.BOARD_AREA), # Number of puzzles x number of cells
            maxval = 2,
            dtype = tf.int32
        )
        return tf.cast(random_solution, tf.float32)