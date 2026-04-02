import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def train(X, y_true, lr, iterations):
    w = np.zeros(X.shape[1])
    b = 0.0
    
    for i in range(iterations):

        y_pred = predict(X, w, b)
        error = y_pred - y_true

        dw = X.T @ error / len(X)
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            loss = np.mean((y_pred - y_true) ** 2)
            print(f"Iteration {i}: loss={loss:.6f}, w={w}, b={b}")

    return w, b

def predict_labels(X, w, b, threshold=0.5):
    final_pred = predict(X, w, b)
    return (final_pred >= threshold).astype(int)