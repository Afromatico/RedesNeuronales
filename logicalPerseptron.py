import abstractPerseptron

class logicalPerseptron(abstractPerseptron.AbstractPerseptron):
    def __init__(self, lr):
        abstractPerseptron.AbstractPerseptron.__init__(self, lr)

    def activationFuntion(self, value):
        if value > 0:
            return 1
        else:
            return 0