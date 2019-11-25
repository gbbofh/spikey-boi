import numpy

from entity import Entity

from network import Layer
from network import Network


class Agent(Entity):

    MOTORWINDOW = 100

    def __init__(self):
        super().__init__()

        self.shape('square')
        self.color('red')
        self.pendown()
        self.showturtle()

        inputLayer = Layer(2, 0)
        outputLayer = Layer(2, 0)

        self.net = Network(10, 0, inputLayer, outputLayer)

        self.motorFrequency = numpy.zeros(2)
        self.motorClock = 0


    def update(self):
        self.net.update([2 * numpy.pi, -2 * numpy.pi])
        outputs = self.net.motorSpikes

        self.motorFrequency[outputs] += 1.0 / Agent.MOTORWINDOW
        self.motorClock += 1

        if self.motorClock >= Agent.MOTORWINDOW:
            self.motorFrequency *= 0
            self.motorClock = 0

        self.left(numpy.diff(self.motorFrequency) * 2.0)
        self.forward(numpy.sum(self.motorFrequency))

