import config
from oracle_model import OracleModel as model
import numpy as np
import tensorflow as tf

x_train = np.load(config.X_TRAIN)['arr_0']
y_train = np.load(config.Y_TRAIN)['arr_0']
x_test = np.load(config.X_TEST)['arr_0']
y_test = np.load(config.Y_TEST)['arr_0']

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy', # Binary classification
    metrics = ['accuracy']
)

X = x_train
Y = y_train

model.fit(
    X,
    Y,
    epochs = 32,
    batch_size = 64,
    validation_split = 0.1,
    shuffle = True    
)

X = x_test
Y = y_test

model.evaluate(
    X,
    Y
)

prediction = model.predict(x_train[:1])[0]
binary_grid = (prediction > 0.5).astype(int)

print(binary_grid.reshape(config.BOARD_SIZE, config.BOARD_SIZE))