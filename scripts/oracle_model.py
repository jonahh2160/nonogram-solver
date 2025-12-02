import config
import tensorflow as tf
from tensorflow import keras
from keras import layers

OracleModel = keras.Sequential(
    [
        tf.keras.layers.Input(shape = (config.CLUE_COUNT,)), # Based on number of clues
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(config.BOARD_AREA, activation='sigmoid') # Binary classification of each cell
    ]
)