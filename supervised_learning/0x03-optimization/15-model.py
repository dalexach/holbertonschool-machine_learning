#!/usr/bin/env python3
"""
Put it all together
"""


import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way:
    Arguments:
     - X: is the first numpy.ndarray of shape (m, nx) to shuffle
        * m is the number of data points
        * nx is the number of features in X
     - Y: is the second numpy.ndarray of shape (m, ny) to shuffle
        * m is the same number of data points as in X
        * ny is the number of features in Y
    Returns:
     The shuffled X and Y matrices
    """
    perm = np.random.permutation(len(X))

    shuffled_X = X[perm]
    shuffled_Y = Y[perm]

    return shuffled_X, shuffled_Y


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross-entropy loss of a prediction
    Arguments:
     - y placeholder for the labels of the input data
     - y_pred tensor that containins the network’s predictions
    Return:
    A tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction
    Arguments:
     - y placeholder for the labels of the input data
     - y_pred tensor that containins the network’s predictions
    Return:
    A tensor containing the decimal accuracy of the prediction
    """
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a NN in tensorflow
    using the Adam optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the weight used for the first moment
     - beta2 is the weight used for the second moment
     - epsilon is a small number to avoid division by zero
    Returns:
    The Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    train = optimizer.minimize(loss)

    return train


def create_layer(prev, n, activation):
    """
    Function that creates the layer
    Arguments:
     - prev: tensor output of the previus layer
     - n: numer of nodes in the new layer
     - activation: activation function to use
    Return:
     The new layer
    """

    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel_ini, name='layer')

    return layer(prev)


def create_batch_norm_layer(prev, n, activation, epsilon):
    """
    Function that creates a batch normalization layer for a NN in tensorflow:
    Arguments:
     - prev is the activated output of the previous layer
     - n is the number of nodes in the layer to be created
     - activation is the activation function that should be used on
        the output of the layer
    Returns:
     A tensor of the activated output for the layer
    """

    kernel_in = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=kernel_in)

    z = layer(prev)

    m, v = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    z_n = tf.nn.batch_normalization(z, m, v, beta, gamma, epsilon)
    y_pred = activation(z_n)

    return y_pred


def create_placeholders(nx, classes):
    """
    Function to create the placeholders
    Arguments:
     - nx is the data
     - classes is the classes for the data
    Return:
    Two placeholders for the NN
    """
    x = tf.placeholder('float', [None, nx], name='x')
    y = tf.placeholder('float', [None, classes], name='y')

    return x, y


def forward_prop(x, layer_sizes, activations, epsilon=1e-8):
    """
    Function that creates the forward propagation graph for the NN
    Arguments:
     - x placeholder for the input data
     - layer list containing the number of nodes in each layer of the network
     - param activations list that containins the activation functions
        for each layer of the network
    Return:
     A tensor with the prediction of the network
    """
    prev = x
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        if i == len(layer_sizes) - 1:
            layer = create_layer(prev, n, activation)
        else:
            layer = create_batch_norm_layer(prev, n, activation, epsilon)
        prev = layer
    return layer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    using inverse time decay
    Arguments:
     - alpha is the original learning rate
     - decay_rate is the weight used to determine the rate at which alpha
        will decay
     - global_step is the number of passes of gradient descent that
        have elapsed
     - decay_step is the number of passes of GD that should occur before
        alpha is decayed further
    Returns:
     The learning rate decay operation
    """

    lr = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                     decay_rate, staircase=True)

    return lr


def get_batch(t, batch_size):
    """
    Helper function to divide data in batches
    """

    batch_list = []
    i = 0
    m = t.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(t[i:(i + batch_size)])
        else:
            batch_list.append(t[i:])
        i += batch_size

    return batch_list


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a NN model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization:
    Arguments:
     - Data_train is a tuple containing the training inputs and training
        labels, respectively
     - Data_valid is a tuple containing the validation inputs and
        validation labels, respectively
     - layers is a list containing the number of nodes in each layer of
        the network
     - activation is a list containing the activation functions used
        for each layer of the network
     - alpha is the learning rate
     - beta1 is the weight for the first moment of Adam Optimization
     - beta2 is the weight for the second moment of Adam Optimization
     - epsilon is a small number used to avoid division by zero
     - decay_rate is the decay rate for inverse time decay of the
        learning rate (the corresponding decay step should be 1)
     - batch_size is the number of data points that should be in a mini-batch
     - epochs is the number of times the training should pass through
        the whole dataset
     - save_path is the path where the model should be saved to
    Returns:
     The path where the model was saved
    """

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    m = Data_train[0].shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign_add(global_step, 1,
                                          name='increment_global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs + 1):
            x_t, y_t = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run((loss, accuracy),
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if e < epochs:
                X_batch_t = get_batch(x_t, batch_size)
                Y_batch_t = get_batch(y_t, batch_size)
                for b in range(1, len(X_batch_t) + 1):
                    sess.run((increment_global_step, train_op),
                             feed_dict={x: X_batch_t[b - 1],
                             y: Y_batch_t[b - 1]})
                    loss_t, acc_t = sess.run((loss, accuracy),
                                             feed_dict={x: X_batch_t[b - 1],
                                                        y: Y_batch_t[b - 1]})
                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_t))
                        print('\t\tAccuracy: {}'.format(acc_t))
    save_path = saver.save(sess, save_path)
    return save_path
