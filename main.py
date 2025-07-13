import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import freefall
from invariant import InvariantModel, train_step


# Usage
model = InvariantModel()
optimizer = tf.keras.optimizers.Adam()


# Generate random regression data: y = 3x + noise
N = 10_000
gs = [9.81, 1.62, 3.71, 24.79, 274.0]
x_train = freefall(gs, N=N)
x_train = tf.convert_to_tensor(x_train)


fig, axes = plt.subplots(3, 1, figsize=(24, 8))
epochs = 30
losses = []
predict = []
for epoch in range(epochs):
    loss = train_step(model, optimizer, x_train)
    losses.append(loss)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")
    losses.append(loss.numpy())


h = np.linspace(0, 3, 1000)
x_test = np.array([h, np.ones(1000)]).T
x_test = tf.convert_to_tensor(x_test)
out_gsh = []
for x in x_test:
    out_gsh.append(model.call(np.array([x]))[0])

t = np.linspace(0, 3, 1000)
x_test = np.array([np.ones(1000), t]).T
x_test = tf.convert_to_tensor(x_test)
out_gst = []
for x in x_test:
    out_gst.append(model.call(np.array([x]))[0])

print(model.call(np.array([[4.905, 1]]))[0])
axes[0].plot(losses)
axes[1].plot(h, out_gsh)
axes[2].plot(t, out_gst)
plt.show()
