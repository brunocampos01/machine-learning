import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data from file
xy = np.loadtxt("../data/diabetes.csv", delimiter=",", dtype=np.float32)
# Input, all rows and n-1 columns
x_data = xy[:, 0:-1]
# Answer, all rows and last column
y_data = xy[:, [-1]]

# Placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# Weight, 2 x 1
W = tf.Variable(tf.random_normal([8,1]), name='weight')
# Bias, 1
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Cost function
cost = -tf.reduce_mean(\
        Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# Gradient descent optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Output function: True if hypthesis>.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# Accuracy computation
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

maxTrials = 20001
trials = [i for i in range(maxTrials)]
costs = []
# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(maxTrials):
        # Training
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        costs.append(cost_val)

    # Accuracy check for trained input data
    h, p, a = sess.run([hypothesis, predicted, accuracy],\
                      feed_dict={X:x_data, Y:y_data})

    print("Acuracy: {0}".format(a))

    plt.plot(trials, costs)
    plt.title("Costs")
    plt.xlabel("trial")
    plt.ylabel("costs")
    plt.grid()
    plt.show()
