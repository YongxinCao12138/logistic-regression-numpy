import numpy as np
from model import sigmoid, predict, train, predict_labels

np.set_printoptions(suppress=True)

# Data for logistic regression
# Two features: [score, study_hours]
X = np.array([
    [50, 1],
    [60, 2],
    [70, 3],
    [80, 6],
    [90, 8]
], dtype=float)

y_true = np.array([0, 0, 0, 1, 1], dtype=float)

# Train the model
lr = 0.001
iterations = 1000
w, b = train(X, y_true, lr, iterations)
print("\nTraining finished")
print("w =", w)
print("b =", b)

# Predict probabilities and labels
final_pred = predict(X, w, b)
print("final prediction:", final_pred)
print("predicted labels:", predict_labels(X, w, b))
