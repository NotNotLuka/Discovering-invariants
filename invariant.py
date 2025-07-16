import tensorflow as tf

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class InvariantModel(tf.keras.Model):
    def __init__(self, hidden_units=320):
        super().__init__()
        self.layers_ = [
            tf.keras.layers.Dense(hidden_units, activation=mish),
            tf.keras.layers.Dense(hidden_units, activation=mish),
            tf.keras.layers.Dense(hidden_units, activation=mish),
            tf.keras.layers.Dense(hidden_units, activation=mish),
            tf.keras.layers.Dense(1, activation=mish),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        return x

    def compute_loss(self, x, epsilon):
        loss = 0.0
        Q = 1
        for group in x:
            #print("group", group.shape)
            Ftheta = self.call(group)
            Ftheta_epsilon = self.call(group + epsilon)
            #print("ftheta", Ftheta.shape)
            loss_group = tf.math.reduce_std(Ftheta) + tf.math.abs(
                Q - tf.math.reduce_std(Ftheta_epsilon)
            )
            loss += loss_group

        return loss


@tf.function
def train_step(model, optimizer, x):
    with tf.GradientTape() as tape:
        epsilon = tf.random.normal(shape=x.shape[1:], mean=0.0, stddev=1.0, dtype=tf.float64)
        loss = model.compute_loss(x, epsilon)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    return loss
