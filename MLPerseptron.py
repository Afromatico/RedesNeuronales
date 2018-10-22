import numpy as np
import abstractPerseptron
import MLPunit


class MLPerseptron(abstractPerseptron.AbstractPerseptron):
    def __init__(self, lr, capas):
        abstractPerseptron.AbstractPerseptron.__init__(self, lr)
        self.capas = [[MLPunit.MLPPerseptronUnit(lr) for _ in range(capas[val])] for val in range(len(capas))]
        self.IO = np.zeros(len(capas))
        for i in range(len(capas) - 1):
            self.IO[i + 1] = capas[i]

        self.out = capas[len(capas)-1]

    def activationFuntion(self, value):
        return 1 / (1 + np.exp(-value))

    def train(self, ntrains, x, y):

        self.initVariables(len(x[0]), len(x))

        self.results = np.zeros((ntrains, len(y), self.out))

        for train in range(ntrains):

            for i in range(len(y)):
                j = 0
                for ele in self.fit(x[i], i=i):
                    self.results[train][i][j] = ele
                    j += 1
            self.cal_error(y)
            input = x
            for array in self.capas:
                raw_input = []
                for per in array:
                    raw_input.append(per.update_weight(input))
                input = raw_input


    def initVariables(self, size, n_elements_data_set):
        self.IO[0] = size
        i = 0
        for array in self.capas:
            for per in array:
                per.initVariables(int(self.IO[i]), n_elements_data_set)
            i += 1

    def fit(self, x, i=0):
        resultado_capa_anterior = [x[ele] for ele in range(len(x))]
        resultado_capa_actual = []
        for layer in self.capas:
            for per in layer:
                resultado_capa_actual.append(per.fit(resultado_capa_anterior, i=i))
            resultado_capa_anterior = resultado_capa_actual
            resultado_capa_actual = []

        return [self.activationFuntion(ele) for ele in resultado_capa_anterior]

    def predict(self, x):
        values = np.zeros((len(x), self.out))
        for i in range(len(x)):
            temp = self.fit(x[i])
            for j in range(len(temp)):
                values[i][j] = temp[j]
        return values

    def cal_error(self, y):
        j = len(self.capas)-1
        first = True
        expected = [[y[l][k] for l in range(len(y))] for k in range(len(y[0]))]
        delta = [[0 for _ in range(len(y))]]
        m = [0 for _ in range(len(y))]
        for _ in range(len(self.capas)):
            i = 0
            delta_ret = []
            m_ret = []
            for per in self.capas[j]:
                if first:
                    ret = per.cal_error(delta, m[i], expected=expected[i],first=first)
                else:
                    ret = per.cal_error(delta, m[i])
                delta_ret.append(ret[0])
                m_ret.append(ret[1])
                i+= 1
            j = j-1
            delta = delta_ret
            m = [[m_ret[l][k] for l in range(len(m_ret))] for k in range(len(m_ret[0]))]
            first=False
            expected = None