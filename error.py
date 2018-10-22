import numpy as np
from perseptron import Perseptron
import matplotlib.pyplot as plt
import random as rand
import pandas as pd

class errorDefinitions:

    def __init__(self):
        self.diff = 0.5

    def ECM(self, expected, real):

        summ = 0
        for elm in range(len(expected)):
            for var in range(len(expected[elm])):
                summ += pow(real[elm][var] - expected[elm][var], 2)

        return summ / len(real)

    def error(self, expected, real):

        summ = 0
        for elm in range(len(expected)):
            for var in range(len(expected[elm])):
                summ += abs(real[elm] - expected[elm])

        return summ / len(real)

    def presicion(self, expected, real):

        summ = 0
        for elm in range(len(expected)):
            for var in range(len(expected[elm])):
                if abs(real[elm][var] - expected[elm][var]) < self.diff:
                    summ += 1

        return summ * 100 / len(real)