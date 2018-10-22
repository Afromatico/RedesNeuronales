import numpy as np
from MLPerseptron import MLPerseptron
import matplotlib.pyplot as plt
import error


per = MLPerseptron(0.5,[3, 2, 1])
xAnd = [[1, 1], [1, 0], [0, 1], [0, 0]]
yAnd = [[1], [0], [0], [0]]

nValues = 20
xrand = np.random.rand(nValues,2)

yvals = [[0] for _ in range(len(xrand))]
i = 0
for elem in xrand:
    if elem[0] > 0.5 or elem[1] > 0.5:
        yvals[i] = [1]
    ##if elem[0] < 0.5 and elem[1] > 0.5:
    ##    yvals[i] = 1
    i+=1

exper = 200
per.train(exper, xrand, yvals)
predict = per.predict(xrand)

plotValues = np.zeros(len(per.results))
err = error.errorDefinitions()

for element in range(len(per.results)):
    plotValues[element] = err.presicion(yvals,per.results[element])

plt.plot(range(0,exper),plotValues,0,1)

plt.show()

print(plotValues[len(plotValues)-1])