#!/usr/bin/env python3
"""
Class DeepNeuralNetwork
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
         - layers (list): representing the number of nodes in each layer of
                          the network
        Public instance attributes:
         - L: The number of layers in the neural network.
         - cache: A dictionary to hold all intermediary values of the network.
         - weights: A dictionary to hold all weights and biased of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Private intance attributes
        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """
        getter function for L
        Returns the number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        getter gunction for cache
        Returns a dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter function for weights
        Returns a dictionary to hold all weights and biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
        """
        self.cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            Z = np.matmul(W, Aprev) + b

            if i is self.__L - 1:
                t = np.exp(Z)
                self.cache[Akey] = t / np.sum(t, axis=0, keepdims=True)
            else:
                self.cache[Akey] = 1 / (1 + np.exp(-Z))

        return (self.__cache[Akey], self.__cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
                              labels for the input data
         - A (numpy.ndarray): with shape (1, m) containing the activated output
                              of the neuron for each example
        Returns:
         The cost
        """
        y1 = 1 - Y
        # y2 = 1.0000001 - A
        m = Y.shape[1]
        # cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))
        cost = -1 * (1 / m) * np.sum(Y * np.log(A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Arguments:
         - X is a numpy.ndarray with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
             labels for the input data
        Returns:
         The neuron’s prediction and the cost of the network, respectively
        """
        self.forward_prop(X)[0]
        tmp = np.amax(self.__cache["A{}".format(self.__L)], axis=0)
        cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
        aux = np.where(self.__cache["A{}".format(self.__L)] == temp, 1, 0)

        return (aux, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Arguments:
         - Y (numpy.ndarray) with shape (1, m) that contains the correct
           labels for the input data
         - cache (dictionary): containing all the intermediary values of
           the network
         - alpha (float): is the learning rate
        """
        m = Y.shape[1]
        Al = cache["A{}".format(self.__L)]
        dAl = (-1 * (Y / Al)) + (1 - Y)/(1 - Al)

        for i in reversed(range(1, self.__L + 1)):
            wkey = "W{}".format(i)
            bkey = "b{}".format(i)
            Al = cache["A{}".format(i)]
            Al1 = cache["A{}".format(i - 1)]
            g = Al * (1 - Al)
            dZ = np.multiply(dAl, g)
            dW = (1 / m) * np.matmul(dZ, Al1.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            W = self.__weights["W{}".format(i)]
            dAl = np.matmul(W.T, dZ)

            self.__weights[wkey] = self.__weights[wkey] - alpha * dW
            self.__weights[bkey] = self.__weights[bkey] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network by updating the private attributes
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray):  with shape (1, m) that contains the correct
              labels for the input data
         - iterations (int): is the number of iterations to train over
         - alpha (float): is the learning rate
         - varbose (boolean): that defines whether or not to print
              information about the training
         - graph (boolean): that defines whether or not to graph information
              about the training once the training has completed
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            # cost = self.cost(Y, A)
            cost = self.cost(Y, self.__cache["A{}".format(self.L)])
            cost_list.append(cost)

            if verbose:
                if i % step == 0 or step == iterations:
                    step_list.append(i)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        Arguments:
         - filename (str): is the file to which the object should be saved
        """
        ext = filename.split('.')
        if len(ext) == 1:
            filename = filename + '.pkl'

        # Writing the fine in binary 'wb'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        Arguments:
         - filename (str): is the file from which the object should be loaded
        Returns:
         The loaded object, or None if filename doesn’t exist
        """
        try:
            # Open the file in binary 'rb'
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None