import tensorflow as tf
import matplotlib.pyplot as plt

# y = x
x_train = [1,2,3]
y_train = [1,2,3]

# Variable is trainable variable for tensorflow.
# It will be upated automatically by tensorflow.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# [1] means its rank is 1, and the number of data is 1.

# Hypothesis: W * x + b
hypothesis = x_train * W + b

# reduce_mean calculates average.
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Training system
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Until this, we build graph.

# Launch the graph in a session
sess = tf.Session()
# Initialize global variables in the grpah.
sess.run(tf.global_variables_initializer())

# For graph
steps = []
outputs = {"cost" : [], "weight" : [], "bias" : []}


# Train
for step in range(1001):
    sess.run(train)

    _cost = sess.run(cost)
    _weight = sess.run(W)
    _bias = sess.run(b)

    steps.append(step)
    outputs["cost"].append(_cost)
    outputs["weight"].append(_weight[0])
    outputs["bias"].append(_bias[0])

    if step in (1, 10, 100, 1000):
        print("Step: {0:4d}, Cost: {1:13.10f}, Weight: {2:13.10f}, Bias: {3:13.10f}".\
              format(step, _cost, _weight[0], _bias[0]))

# Draw graph
for k, v in outputs.items():
    plt.plot(steps, v)
    plt.title(k)
    plt.xlabel("step")
    plt.ylabel(k)
    plt.show()
