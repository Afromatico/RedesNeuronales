import random as rand
import numpy as np


class AbstractPerseptron(object):
    def __init__(self, lr):
        self.b = 0
        self.m = []
        self.lr = lr
        self.results = []

    def train(self, ntrains, x, y):

        self.initVariables(len(x[0]),len(x))

        self.results = np.zeros((ntrains,len(y)))

        for train in range(ntrains):

            for i in range(len(y)):
                self.results[train][i] = self.fit(x[i])
                for j in range(len(x[i])):
                    self.m[j] = self.m[j] + (self.lr * x[i][j] * (y[i] - self.results[train][i]))
                self.b = self.b + (self.lr * (y[i] - self.results[train][i]))

    def initVariables(self, size, n_elements_dataset):
        self.b = rand.random() * 4.0 - 2.0
        self.m = np.zeros(size)
        for j in range(size):
            self.m[j] = rand.random() * 4.0 - 2.0

    def fit(self, x, i=0):
        value = 0
        for j in range(len(x)):
            value = value + x[j] * self.m[j]
        value = value + self.b
        return self.activationFuntion(value)

    def predict(self, x):
        values = np.zeros(len(x))
        for i in range(len(x)):
            values[i] = self.fit(x[i])
        return values

    def activationFuntion(self, value):
        return value

    def results(self):
        return self.results