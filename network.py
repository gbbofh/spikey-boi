import numpy
import scipy.stats as stats

def isin(el, te, au=False, invert=False):
    el = numpy.asarray(el)
    return numpy.in1d(el, te, au, invert=invert).reshape(el.shape)


class Network():

    STDP_INIT = 0.2
    STDP_DECAY = 0.2
    SYNAPSE_DECAY = 0.0001

    """
    Implements the simple spiking model by Eugene Izhikevich

    Generates a randomly connected network of numEx + numIn neurons with
    random properties, as described in the paper
    'A Simple Model of Spiking Neurons' (2003).
    """
    def __init__(self, numEx, numIn, inputs=0, outputs=0):
        """
        Constructs a randomly connected spiking neural network

        Parameters:
        numEx (int): The number of excitatory neurons to generate
        numIn (int): The number of inhibitory neurons to generate
        inputs (int): The number of input neurons
        outputs (int): The number of output neurons
        """

        self.numEx = numEx
        self.numIn = numIn
        totalNum = numEx + numIn

        self.totalNum = totalNum

        r = stats.uniform.rvs(size=(totalNum))

        self.scale = numpy.ones(totalNum)
        self.uSens = numpy.ones(totalNum)
        self.reset = numpy.ones(totalNum)
        self.uReset = numpy.ones(totalNum)

        self.scale[0 : numEx] *= 0.02
        self.scale[numEx : ] *= 0.02 + 0.08 * r[numEx : ]
        self.uSens[0 : numEx] *= 0.2
        self.uSens[numEx : ] *= 0.25 - 0.05 * r[numEx : ]
        self.reset[0 : numEx] *= -65 + 15 * r[0 : numEx] ** 2
        self.reset[numEx : ] *= -65
        self.uReset[0 : numEx] *= 8 - 6 * r[0 : numEx] ** 2
        self.uReset[numEx : ] *= 2

        self.synapses = 0.5 * stats.uniform.rvs(size=(totalNum, totalNum))
        self.synapses[numEx : ] *= -2.0

        self.voltage = numpy.full(totalNum, -65.0)
        self.recovery = numpy.multiply(self.voltage, self.uSens)
        self.input = numpy.zeros(totalNum)

        self.sensoryInput = numpy.ones(inputs)
        self.motorOutput = numpy.zeros(outputs)
        tmp = numpy.random.choice(numEx, inputs + outputs, replace=False)
        self.sensorySyn = tmp[0 : inputs]
        self.motorSyn = tmp[inputs : inputs + outputs]

        self.stdp = numpy.zeros(totalNum)

        self.pSpikes = numpy.where(self.voltage >= 30.0)[0]


    def update(self):
        """
        Updates the membrane potential for all neurons in the network

        Returns:
        list: The list of spikes which were generated in the previous timestep
        """

        spikes = numpy.where(self.voltage >= 30.0)[0]

        self.voltage[spikes] = self.reset[spikes]
        self.recovery[spikes] += self.uReset[spikes]

        self.input[self.sensorySyn] += self.sensoryInput
        #self.input[self.sensorySyn] = self.sensoryInput
        self.input += self.synapses[spikes, :].sum(axis=0)

        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.recovery += self.scale * (self.voltage * self.uSens - self.recovery)

        self.voltage[numpy.where(self.voltage >= 30.0)] = 30.0

        mo = numpy.where(self.voltage[self.motorSyn] >= 30.0, True, False)

        self.motorOutput = numpy.clip(mo - 0.1 * self.motorOutput, 0.0, 1.0)

        self.stdp *= Network.STDP_DECAY
        self.stdp[spikes] = Network.STDP_INIT

        for s in range(self.totalNum):
            for ps in range(self.totalNum):
                # We have to check if it is less than STDP_INIT in order to
                # prevent neurons that just fired in this time-step from
                # stimulating themselves. Without this check,
                # the network develops some very odd symmetric qualities
                if s in spikes and self.stdp[ps] < Network.STDP_INIT:
                    # w' = w + p
                    # w = e^(nt) - p
                    self.synapses[ps, s] += self.stdp[ps] * numpy.sign(self.synapses[ps, s])
                else:
                    # w' = w - kw
                    # w' = w(1 - k)
                    # w = e^(t - kt)
                    self.synapses[ps, s] -= Network.SYNAPSE_DECAY * self.synapses[ps, s]
                # if s in spikes and self.synapses[ps, s] != 0.0:
                    #self.synapses[ps, s] += self.stdp[ps] * 0.9
                    #self.stdp[ps] = 0.0
                #else:
                    #self.synapses[ps, s] -= 0.0005 * self.synapses[ps, s]

        # TODO: FIX THIS BULLSHIT. There has to be a better way.
        # for s in range(self.totalNum):
        #     for ps in range(self.totalNum):
        #         if s in spikes:
        #             self.stdp[ps] = 0.0
        # Lock weights to be between -1.0 and 1.0
        self.synapses = numpy.clip(self.synapses, -1.0, 1.0)

        self.pSpikes = spikes
        return spikes
