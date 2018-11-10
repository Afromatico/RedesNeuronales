import numpy as np
from MLPerseptron import MLPerseptron
import matplotlib.pyplot as plt
import error

from sklearn.model_selection import train_test_split

import pandas as pd
data = pd.read_csv("samples/iris.csv")

data_x = data.ix[:,[0,1,2,3]]
data_y = data.ix[:,4]

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

per = MLPerseptron(0.1,[4, 7, 1])
# xAnd = [[1, 1], [1, 0], [0, 1], [0, 0]]
# yAnd = [[1], [0], [0], [0]]
#
# nValues = 200
# xrand = np.random.rand(nValues,2)
#
# yvals = [[0] for _ in range(len(xrand))]
# i = 0
# for elem in xrand:
#     if elem[0] > 0.5 and elem[1] < 0.5:
#         yvals[i] = [1]
#     if elem[0] < 0.5 and elem[1] > 0.5:
#         yvals[i] = [1]
#     i+=1

# per.train(1, [[1, 1]], [[1, 1]], weights=[[[0.7, 0.3], [0.3, 0.7]], [[0.2, 0.3],[0.4, 0.2]]], bias=[[0.5, 0.4], [0.3, 0.6]])
#
# print(per.results)
#
# X_train, X_test, y_train, y_test = train_test_split(xrand, yvals, test_size=0.3)

exper = 100
per.train(exper, X_train.values, [y_train.values])
predict = per.predict(X_test.values)
#
#
plotValues = np.zeros(len(per.results))
err = error.errorDefinitions()

# print(err.ECM(labels_test, predict))
# print(err.presicion(labels_test, predict))
# print(err.error(labels_test, predict))


for element in range(len(per.results)):
   plotValues[element] = err.presicion([y_train.values], per.results[element])

plt.plot(range(0,exper),plotValues)

plt.show()

print(err.presicion([y_test.values],predict))

