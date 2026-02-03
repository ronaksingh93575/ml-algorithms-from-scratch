import numpy as np
from utils import mean_squared_error, add_bias


class LinearRegression:
    """
    Linear Regression implemented using Gradient Descent.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model.
        """
        X = add_bias(X)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = X @ self.weights
            gradient = (2 / n_samples) * (X.T @ (y_pred - y))
            self.weights -= self.learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        """
        X = add_bias(X)
        return X @ self.weights


if __name__ == "__main__":
    # Example usage
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("MSE:", mean_squared_error(y, predictions))
