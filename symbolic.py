import numpy as np
from pysr import PySRRegressor
from data import freefall_g
import matplotlib.pyplot as plt


X, y = freefall_g(10_000)

model = PySRRegressor(
    niterations=5,
    unary_operators=["sin", "cos", "sqrt", "exp"],
    binary_operators=["+", "-", "*", "/"],
)

model.fit(X, y)
best_eq = model.get_best()
print(best_eq)  # e.g., x0^2 + sin(x0)

# Evaluate model prediction
x0 = np.linspace(-5, 5, 200)
x1 = np.ones(200)

X_test = np.array([x0, x1]).T
y_pred = model.predict(X_test)

# Plot
plt.plot(x0, y_pred, "--", label="Symbolic Fit", linewidth=2)
plt.legend()
plt.show()
