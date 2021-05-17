import tensorflow as tf
import matplotlib.pyplot as plt

# Placeholder can be used as input data
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
# [None] means it is 1 deimensional tensor
#  and its number of values is not determined.

# Variable is trainable variable for tensorflow.
# It will be upated automatically by tensorflow.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# [1] means its rank is 1, and its total count is 1.

# Hypothesis: W * x + b
hypothesis = X * W + b

# reduce_mean calculates average.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

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
    # Return order is the same as the input order of run
    _cost, _weight, _bias, _ = sess.run([cost, W, b, train],\
                                        feed_dict={X:[1,2,3], Y:[1,2,3]})

    steps.append(step)
    outputs["cost"].append(_cost)
    outputs["weight"].append(_weight[0])
    outputs["bias"].append(_bias[0])

    if step in (1, 10, 100, 1000):
        print(\
            "Step: {0:4d}, Cost: {1:13.10f}, Weight: {2:13.10f}, Bias: {3:13.10f}".\
              format(step, _cost, _weight[0], _bias[0]))

# After training, we can test hypothesis with new input
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[1.8, 3.2]}))

# Draw graph
for k, v in outputs.items():
    plt.plot(steps, v)
    plt.title(k)
    plt.xlabel("step")
    plt.ylabel(k)
    plt.show()