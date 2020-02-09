import tensorflow as tf

W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def model(x):
    return W * x + b

m = model([0,1])

print(m)

