#!/usr/bin/env python3
"""
Class Neuron
"""


import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class Neuron that defines a simple neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
        Private instance attributes:
         - W: The weights vector for the neuron. Upon instantiation, it should
              be initialized using a random normal distribution.
         - b: The bias for the neuron. Upon instantiation, it should be
              initialized to 0.
         - A: The activated output of the neuron (prediction). Upon
              instantiation, it should be initialized to 0.

        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private instance attributes
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for W
        Returns weights
        """
        return self.__W

    @property
    def b(self):
        """
        getter gunction for b
        Returns bias
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for A
        Returns activation values
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Arguments:
        - X (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute __A
        """
        z = np.dot(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))

        self.__A = sigmoid

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
           contains the correct labels for the input data
         - A is a numpy.ndarray with shape (1, m) containing the activated
           output of the neuron for each example
        Returns:
         The cost
        """
        y1 = 1 - Y
        y2 = 1.0000001 - A
        m = Y.shape[1]
        cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
        Returns:
         The neuron’s prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return (np.round(A).astype(int), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
         - A (numpy.ndarray): with shape (1, m) containing the activated output
            of the neuron for each example
         - alpha (float): is the learning rate
        Updates the private attributes __W & __b
        """
        dZ = A - Y
        m = Y.shape[1]
        dW = (1 / m) * np.matmul(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
         - iterations (int): number of iterations to train over
         - alpha (float): is the learning rate
         - verbose (boolean): that defines whether or not to print information
           about the training. If True, print
            Cost after {iteration} iterations: {cost} every step iterations
         - graph (boolean): that defines whether or not to graph information
           about the training once the training has completed.
        Updates the private attributes __W, __b & __A
        Returns:
         The evaluation of the training data after iterations of training
         have occurred
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
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)

            if verbose:
                if i % step == 0 or step == iterations:
                    cost_list.append(cost)
                    step_list.append(i)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)
