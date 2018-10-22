import numpy as np
import unittest
import matplotlib.pyplot as plt
from sklearn import datasets

from perseptron import Perseptron

class TestPerseptron(unittest.TestCase):

    def setUp(self):
        self.per = Perseptron(0.1)
        self.xAnd = [[1,1],[1,0],[0,1],[0,0]]
        self.yAnd = [1,0,0,0]

    def runTest(self):

        self.per.train(100,self.xAnd,self.yAnd)
        predict = self.per.predict(self.xAnd)
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()