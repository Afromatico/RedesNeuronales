import numpy as np
import abstractPerseptron


class MLPPerseptronUnit(abstractPerseptron.AbstractPerseptron):
    def __init__(self, lr):
        abstractPerseptron.AbstractPerseptron.__init__(self, lr)
        self.result = []
        self.error = []

    def initVariables(self, size, n_elements_dataset, weights=None, bias=None):
        super(MLPPerseptronUnit,self).initVariables(size,n_elements_dataset, weights=weights, bias=bias)

        self.result = np.zeros(n_elements_dataset)
        self.error = np.zeros(n_elements_dataset)
        self.delta = np.zeros(n_elements_dataset)



    def activationFuntion(self, value):
        if -value > np.log(np.finfo(type(value)).max):
            return 0.0
        return 1/(1+np.exp(-value))

    def train(self, ntrains, x, y):
        pass

    def fit(self, x, i=0):
        value = 0
        for j in range(len(x)):
            value = value + x[j] * self.m[j]
        value = value + self.b
        self.result[i] = self.activationFuntion(value)
        return self.result[i]

    def lastResult(self):
        return self.result

    def cal_error(self, delta, m, expected=None, first=False):
        for var in range(len(delta[0])):
            if first:
                self.error[var] = expected[var] - self.result[var]
            else:
                summ = 0
                for per in range(len(delta)):
                    summ += delta[per][var] * m[per]
                self.error[var] = summ
            self.delta[var] = self.error[var]*self.transfer_derivative_output(self.result[var])
        return [[elm for elm in self.delta], [elm for elm in self.m]]


    def transfer_derivative_output(self, output):
        return output*(1-output)

    def update_weight(self, input):
        for i in range(len(input)):
            for j in range(len(self.m)):
                self.m[j] = self.m[j] + self.lr*self.delta[i]*input[i][j]
            self.b = self.b + (self.lr*self.delta[i])
        ret = [_ for _ in self.result]
        self.result = np.zeros(len(self.result))
        return ret