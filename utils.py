import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss.
    """
    return np.mean((y_true - y_pred) ** 2)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))


def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add bias (intercept) term to feature matrix.
    """
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))
