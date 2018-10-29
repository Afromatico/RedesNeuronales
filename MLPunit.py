import numpy as np
import abstractPerseptron


class MLPPerseptronUnit(abstractPerseptron.AbstractPerseptron):
    def __init__(self, lr):
        abstractPerseptron.AbstractPerseptron.__init__(self, lr)
        self.result = []
        self.error = []

    def initVariables(self, size, n_elements_dataset, weights=None, bias=None):
        super(MLPPerseptronUnit,self).initVariables(size,n_elements_dataset, weights=weights, bias=bias)

        self.result = [0 for _ in range(n_elements_dataset)]
        self.error = [0 for _ in range(n_elements_dataset)]
        self.delta = [0 for _ in range(n_elements_dataset)]


    def activationFuntion(self, value):
        if -value > np.log(np.finfo(type(value)).max):
            return 0.0
        return 1/(1+np.exp(-value))

    def train(self, ntrains, x, y):
        pass

    def fit(self, x, i=0):
        self.result[i] = self.activationFuntion(sum(x[j]*self.m[j] for j in range(len(x))) + self.b)
        return self.result[i]

    def lastResult(self):
        return self.result

    def cal_error(self, delta, m, expected=None, first=False):
        for var in range(len(delta[0])):
            if first:
                self.error[var] = expected[var] - self.result[var]
            else:
                self.error[var] = sum(delta[per][var]*m[per] for per in range(len(delta)))
            self.delta[var] = self.error[var]*self.transfer_derivative_output(self.result[var])
        return [self.delta, self.m]


    def transfer_derivative_output(self, output):
        return output*(1-output)

    def update_weight(self, input):
        for i in range(len(input)):
            for j in range(len(self.m)):
                self.m[j] = self.m[j] + self.lr*self.delta[i]*input[i][j]
            self.b = self.b + (self.lr*self.delta[i])
        ret = self.result
        self.clean_memory()

        return ret

    def clean_memory(self):
        self.result = [0 for _ in range(len(self.result))]
        self.error = [0 for _ in range(len(self.error))]
        self.delta = [0 for _ in range(len(self.delta))]