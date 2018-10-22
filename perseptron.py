import numpy as np
import abstractPerseptron


class Perseptron(abstractPerseptron.AbstractPerseptron):
    def __init__(self, lr):
        abstractPerseptron.AbstractPerseptron.__init__(self, lr)

    def activationFuntion(self, value):
        return 1/(1+np.exp(-value))