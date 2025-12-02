import config
import baseline_model
import oracle_model
import numpy as np
import tensorflow as tf

# Load test datasets
x_test = np.load(config.X_TEST)['arr_0']
y_test = np.load(config.Y_TEST)['arr_0']

# Initialize models
randomModel = baseline_model.BaselineModel()
heuristicModel = oracle_model.OracleModel()

def evaluateModel(model, x_test, y_test):
    """Evaluates the percent of cells filled in accurately by the given model."""

    # Get predictions
    predictions = model(x_test)
    solutions = tf.cast(predictions > 0.5, tf.float32).numpy() # Convert predictions to boolean, then to binary

    # Calculate puzzle accuracy
    cells_correct = solutions == y_test
    cells_per_puzzle = np.sum(cells_correct, axis = 1)
    puzzle_accuracy_percent = (cells_per_puzzle / config.BOARD_AREA) * 100
    return puzzle_accuracy_percent

baselineResults = evaluateModel(randomModel, x_test, y_test)
oracleResults = evaluateModel(heuristicModel, x_test, y_test)

# Print results
print("Baseline model cell accuracy: " + baselineResults + "%")
print("Oracle model cell accuracy: " + oracleResults + "%")