import config
from oracle_model import OracleModel as model
import numpy as np
import tensorflow as tf

x_train = np.load(config.X_TRAIN)['arr_0'].astype(np.float32)
y_train = np.load(config.Y_TRAIN)['arr_0'].astype(np.float32)
x_test = np.load(config.X_TEST)['arr_0'].astype(np.float32)
y_test = np.load(config.Y_TEST)['arr_0'].astype(np.float32)

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy', # Binary classification
    metrics = ['binary_accuracy'] # Checks the average accuracy of predictions by cell
)

model.fit(
    x_train,
    y_train,
    epochs = config.EPOCHS,
    batch_size = config.BATCH_SIZE,
    validation_split = 0.1,
    shuffle = True    
)

model.evaluate(
    x_test,
    y_test
)

prediction = model.predict(x_train[:1])[0]
binary_grid = (prediction > 0.5).astype(int)

print(binary_grid.reshape(config.BOARD_SIZE, config.BOARD_SIZE))