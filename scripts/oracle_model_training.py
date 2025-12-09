import config
from oracle_model import OracleModel as model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Assign datasets to variables
x_train = np.load(config.X_TRAIN)['arr_0'].astype(np.float32)
y_train = np.load(config.Y_TRAIN)['arr_0'].astype(np.float32)
x_test = np.load(config.X_TEST)['arr_0'].astype(np.float32)
y_test = np.load(config.Y_TEST)['arr_0'].astype(np.float32)

# Compile model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy', # Binary classification
    metrics = ['binary_accuracy'] # Checks the average accuracy of predictions by cell
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs = config.EPOCHS,
    batch_size = config.BATCH_SIZE,
    validation_split = 0.1,
    shuffle = True    
)

# Evaluate model
model.evaluate(
    x_test,
    y_test
)

# Predict with model and calculate percent of perfectly solved puzzles
prediction = model.predict(x_test)
binary_prediction = (prediction > 0.5).astype(np.float32)
correct = np.all(binary_prediction == y_test, axis = 1)
percent_correct = correct.mean()
print("Percent of puzzles solved: ", percent_correct)

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Overall Cell Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()