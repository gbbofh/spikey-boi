import numpy

from entity import Entity

from network import Layer
from network import Network


class Agent(Entity):

    MOTORWINDOW = 500

    def __init__(self, numIn=0, numEx=0):
        super().__init__()

        self.shape('arrow')
        self.color('red')
        self.pendown()
        self.showturtle()

        inputLayer = Layer(3, 2)
        outputLayer = Layer(2, 0)

        self.net = Network(numIn, numEx, inputLayer, outputLayer)

        self.motorFrequency = numpy.zeros(2)
        self.motorClock = 0


    def update(self):
        angle = self.towards(self.target)
        heading = self.heading()

        angle = angle - heading + numpy.pi
        angle = angle if angle <= numpy.pi else angle - 2 * numpy.pi
        iAngle = numpy.pi * numpy.sign(angle) - angle
        frontAngle = abs(angle) if abs(angle) >= numpy.pi * (1 - 1 / 6) else 0

        self.net.update((5 * angle, -5 * angle, 5 * frontAngle, 5 * iAngle, -5 * iAngle))
        outputs = self.net.motorSpikes

        self.motorFrequency[outputs] += 1.0 / Agent.MOTORWINDOW
        self.motorClock += 1

        if self.motorClock >= Agent.MOTORWINDOW:
            self.motorFrequency *= 0
            self.motorClock = 0

        self.left(numpy.diff(self.motorFrequency) * 4.0)
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
