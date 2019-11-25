import numpy

from entity import Entity

from network import Layer
from network import Network


class Agent(Entity):

    MOTORWINDOW = 100

    def __init__(self):
        super().__init__()

        # self.shape('square')
        self.shape('arrow')
        self.color('red')
        self.pendown()
        self.showturtle()

        inputLayer = Layer(2, 0)
        outputLayer = Layer(2, 0)

        self.net = Network(2, 1, inputLayer, outputLayer)

        self.motorFrequency = numpy.zeros(2)
        self.motorClock = 0


    def update(self):
        angle = self.towards(self.target)
        heading = self.heading()

        angle = angle - heading + numpy.pi
        angle = angle if angle <= numpy.pi else angle - 2 * numpy.pi

        # self.net.update((angle, -angle))
        self.net.update((2 * angle, -2 * angle))
        outputs = self.net.motorSpikes

        self.motorFrequency[outputs] += 1.0 / Agent.MOTORWINDOW
        self.motorClock += 1

        if self.motorClock >= Agent.MOTORWINDOW:
            self.motorFrequency *= 0
            self.motorClock = 0

        self.left(numpy.diff(self.motorFrequency) * 2.0)
        self.forward(numpy.sum(self.motorFrequency))


    def setTarget(self, target=None):
        self.target = target

