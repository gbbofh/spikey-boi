import numpy

from entity import Entity

from network import Layer
from network import Network


class Agent(Entity):

    MOTORWINDOW = 100

    def __init__(self, numIn=0, numEx=0):
        super().__init__()

        self.shape('arrow')
        self.color('red')
        self.pendown()
        self.showturtle()

        # inputLayer = Layer(3, 2)
        inputLayer = Layer(3, 0)
        outputLayer = Layer(2, 0)

        self.net = Network(numIn, numEx, inputLayer, outputLayer)

        self.motorAccum = numpy.zeros((Agent.MOTORWINDOW, 2))
        self.motorFrequency = numpy.zeros(2)
        self.motorClock = 0


    def update(self):
        angle = self.towards(self.target)
        heading = self.heading()

        angle = angle - heading + numpy.pi
        angle = angle if angle <= numpy.pi else angle - 2 * numpy.pi
        iAngle = numpy.pi * numpy.sign(angle) - angle
        frontAngle = abs(angle) if abs(angle) >= numpy.pi * (1 - 1 / 6) else 0

        # self.net.update((5 * angle, -5 * angle, 5 * frontAngle, 5 * iAngle, -5 * iAngle))
        self.net.update((5 * angle, -5 * angle, 5 * frontAngle))
        outputs = self.net.motorSpikes

        self.motorAccum[self.motorClock % Agent.MOTORWINDOW] = 0
        self.motorAccum[self.motorClock % Agent.MOTORWINDOW][outputs] = 1
        self.motorClock += 1
        self.motorFrequency = numpy.sum(self.motorAccum, axis=0) / self.MOTORWINDOW
        # self.motorFrequency[outputs] += 1.0 / Agent.MOTORWINDOW
        # self.motorClock += 1

        # if self.motorClock >= Agent.MOTORWINDOW:
        #     self.motorFrequency *= 0
        #     self.motorClock = 0

        self.left(numpy.diff(self.motorFrequency) * 2.0)
        self.forward(numpy.sum(self.motorFrequency) * 20.0)


    def setTarget(self, target=None):
        self.target = target


    def reward(self, delta):
        sensorySyn = self.net.inputSynapses
        recurrentSyn = self.net.synapses
        motorSyn = self.net.outputSynapses

        self.net.inputSynapses = numpy.clip(sensorySyn * delta, -1.0, 1.0)
        self.net.synapses = numpy.clip(recurrentSyn * delta, -1.0, 1.0)
        self.net.outputSynapses = numpy.clip(motorSyn * delta, -1.0, 1.0)
