import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import freefall, freefall_g
from invariant import InvariantModel, train_step
from pysr import PySRRegressor


# Usage
model = InvariantModel()
optimizer = tf.keras.optimizers.Adam()

N = 5_000
# vrednosti podobne velikosti, brez večjih lukenj
gs = [1.62, 2.11, 3.33, 3.71, 4.9, 7.11, 8.3, 9.81]
x_train = freefall(gs, N=N)

fig, axes = plt.subplots(3, 1, figsize=(24, 8))
epochs = 100  # dobro bi bilo imeti to veliko večje recimo 100
losses = []
predict = []
for epoch in range(epochs):
    loss = train_step(model, optimizer, x_train)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")
    losses.append(loss)

# narisana grafa g_est od g in sqrt(g_est) od g
g_test = np.linspace(min(gs), max(gs))
g_ests = []
for g in g_test:
    # izračunamo estimirano konstanto, kot povprečje večih outputov
    g_est = np.average(model(freefall([g], N=5000)[0, :, :]))
    g_ests.append(g_est)

# axes[0].plot(losses)
# axes[1].plot(g_test, g_ests)
# axes[2].plot(g_test, np.sqrt(g_ests - min(g_ests)))
# plt.show()

X, _ = freefall_g(10_000)
y = model(X)

model = PySRRegressor(
    niterations=40,
    unary_operators=["sin", "cos", "sqrt", "exp"],
    binary_operators=["+", "-", "*", "/"],
)

model.fit(X, y)
best_eq = model.get_best()
print(best_eq)

x0 = np.linspace(0, 5, 200)
x1 = np.ones(200)

X_test = np.array([x0, x1]).T
y_pred = model.predict(X_test)
