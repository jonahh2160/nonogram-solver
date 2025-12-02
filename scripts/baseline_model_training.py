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
# heuristicModel = oracle_model.OracleModel

def evaluateBaselineModel(model, x_test, y_test):
    """Evaluates the percent of cells filled in accurately by the baseline model."""

    # Get predictions
    predictions = model(x_test)
    solutions = tf.cast(predictions > 0.5, tf.float32).numpy() # Convert predictions to boolean, then to binary

    # Calculate puzzle accuracy
    cells_correct = solutions == y_test
    cells_per_puzzle = np.sum(cells_correct, axis = 1)
    return cells_per_puzzle.mean()

#def evaluateOracleModel(model, x_test, y_test):
    """Evaluates the percent of cells filled in accurately by the oracle model."""

baselineResults = evaluateBaselineModel(randomModel, x_test, y_test)
#oracleResults = evaluateOracleModel(heuristicModel, x_test, y_test)

# Print results
print("Baseline model cell accuracy: " + str(baselineResults) + "%")
#print("Oracle model cell accuracy: " + oracleResults + "%")