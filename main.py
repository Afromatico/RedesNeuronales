import numpy as np
from MLPerseptron import MLPerseptron
import matplotlib.pyplot as plt
import error

from sklearn.model_selection import train_test_split


#
# from mnist import MNIST
#
# mndata = MNIST('samples')
#
# images, labels = mndata.load_training()
# # or
# images_test, labels_test = mndata.load_testing()




per = MLPerseptron(0.1,[2, 1])
# xAnd = [[1, 1], [1, 0], [0, 1], [0, 0]]
# yAnd = [[1], [0], [0], [0]]
#
nValues = 200
xrand = np.random.rand(nValues,2)
#
yvals = [[0] for _ in range(len(xrand))]
i = 0
for elem in xrand:
    if elem[0] > 0.5 and elem[1] < 0.5:
        yvals[i] = [1]
    if elem[0] < 0.5 and elem[1] > 0.5:
        yvals[i] = [1]
    i+=1

# per.train(1, [[1, 1]], [[1, 1]], weights=[[[0.7, 0.3], [0.3, 0.7]], [[0.2, 0.3],[0.4, 0.2]]], bias=[[0.5, 0.4], [0.3, 0.6]])
#
# print(per.results)
#
X_train, X_test, y_train, y_test = train_test_split(xrand, yvals, test_size=0.3)

exper = 2000
per.train(exper, X_train, y_train)
predict = per.predict(X_test)
#
#
plotValues = np.zeros(len(per.results))
err = error.errorDefinitions()

# print(err.ECM(labels_test, predict))
# print(err.presicion(labels_test, predict))
# print(err.error(labels_test, predict))


for element in range(len(per.results)):
   plotValues[element] = err.presicion(y_train, per.results[element])

plt.plot(range(0,exper),plotValues)

plt.show()

print(err.presicion(y_test,predict))

