import numpy as np
from MLPerseptron import MLPerseptron
import matplotlib.pyplot as plt
import error

#
# from mnist import MNIST
#
# mndata = MNIST('samples')
#
# images, labels = mndata.load_training()
# # or
# images_test, labels_test = mndata.load_testing()




per = MLPerseptron(0.5,[1, 1])
xAnd = [[1, 1], [1, 0], [0, 1], [0, 0]]
yAnd = [[1], [0], [0], [0]]

nValues = 200
xrand = np.random.rand(nValues,2)

yvals = [[0] for _ in range(len(xrand))]
i = 0
for elem in xrand:
    if elem[0] > 0.5 and elem[1] < 0.5:
        yvals[i] = [1]
    if elem[0] < 0.5 and elem[1] > 0.5:
        yvals[i] = [1]
    i+=1

per.train(1, [[1, 1]], [[1]], weights=[[[0.4, 0.3]], [[0.3]]], bias=[[0.5], [0.4]])

print(per.results)

# exper = 200
# per.train(exper, images, labels)
# predict = per.predict(images_test)
#
#
# #plotValues = np.zeros(len(per.results))
# err = error.errorDefinitions()
#
# print(err.ECM(labels_test, predict))
# print(err.presicion(labels_test, predict))
# print(err.error(labels_test, predict))
#

#for element in range(len(per.results)):
#    plotValues[element] = err.ECM(yvals,per.results[element])

#plt.plot(range(0,exper),plotValues)

#plt.show()

#print(plotValues[len(plotValues)-1])

