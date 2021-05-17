import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# From reproducibility
tf.set_random_seed(777)

# Learning rate
learning_rate = 0.1

# Inputs data
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
# Labels
y_data = [[0],
          [1],
          [1],
          [0]]

# Inputs array
x_data = np.array(x_data, dtype=np.float32)
# Labels array
y_data = np.array(y_data, dtype=np.float32)

# Placeholder for Inputs and Labels
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# Weights for each layers
W_i = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight_input')
W_h1 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_1')
W_h2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_2')
W_h3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_3')
W_h4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_4')
W_h5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_5')
W_h6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_6')
W_h7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_7')
W_h8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_8')
W_h9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight_hidden_9')
W_o = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight_output')

# Biases for each layers
b_i = tf.Variable(tf.zeros([5]), name='bias_input')
b_h1 = tf.Variable(tf.zeros([5]), name='bias_hidden_1')
b_h2 = tf.Variable(tf.zeros([5]), name='bias_hidden_2')
b_h3 = tf.Variable(tf.zeros([5]), name='bias_hidden_3')
b_h4 = tf.Variable(tf.zeros([5]), name='bias_hidden_4')
b_h5 = tf.Variable(tf.zeros([5]), name='bias_hidden_5')
b_h6 = tf.Variable(tf.zeros([5]), name='bias_hidden_6')
b_h7 = tf.Variable(tf.zeros([5]), name='bias_hidden_7')
b_h8 = tf.Variable(tf.zeros([5]), name='bias_hidden_8')
b_h9 = tf.Variable(tf.zeros([5]), name='bias_hidden_9')
b_o = tf.Variable(tf.zeros([1]), name='bias_output')

# Layers
L_i = tf.sigmoid(tf.matmul(X, W_i) + b_i)
L_h1 = tf.sigmoid(tf.matmul(L_i, W_h1) + b_h1)
L_h2 = tf.sigmoid(tf.matmul(L_h1, W_h2) + b_h2)
L_h3 = tf.sigmoid(tf.matmul(L_h2, W_h3) + b_h3)
L_h4 = tf.sigmoid(tf.matmul(L_h3, W_h4) + b_h4)
L_h5 = tf.sigmoid(tf.matmul(L_h4, W_h5) + b_h5)
L_h6 = tf.sigmoid(tf.matmul(L_h5, W_h6) + b_h6)
L_h7 = tf.sigmoid(tf.matmul(L_h6, W_h7) + b_h7)
L_h8 = tf.sigmoid(tf.matmul(L_h7, W_h8) + b_h8)
L_h9 = tf.sigmoid(tf.matmul(L_h8, W_h9) + b_h9)
hypothesis = tf.sigmoid(tf.matmul(L_h9, W_o) + b_o)

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# Optimizer
train = tf.train.\
            GradientDescentOptimizer(learning_rate=learning_rate).\
            minimize(cost)

# Set threshold.
#  True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y),\
            dtype=tf.float32))

costs= []
accs = []

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        # Train
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        _cost = sess.run(cost, feed_dict={
                X: x_data, Y: y_data})
        costs.append(_cost)
        _acc = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
        accs.append(_acc)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

steps = [i for i in range(len(accs))]

plt.plot(steps, costs)
plt.title("Costs")
plt.xlabel("Steps")
plt.ylabel("Cost")
plt.show()

plt.plot(steps, accs)
plt.title("Accuracies")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.show()
