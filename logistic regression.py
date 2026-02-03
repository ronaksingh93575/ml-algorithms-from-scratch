import numpy as np
from utils import sigmoid, add_bias


class LogisticRegression:
    """
    Logistic Regression implemented using Gradient Descent.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.
        """
        X = add_bias(X)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            linear_output = X @ self.weights
            y_pred = sigmoid(linear_output)

            gradient = (1 / n_samples) * (X.T @ (y_pred - y))
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates.
        """
        X = add_bias(X)
        return sigmoid(X @ self.weights)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":
    # Example usage
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print("Predictions:", predictions)
