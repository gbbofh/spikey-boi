import numpy
import scipy.stats as stats

# TODO: Consider making a base class since this and Network
# share so much code in common -- the only real difference is
# the lack of synapses for the layer.
# NOTE: Perhaps just merge this and Network back together
# and provide a method to easily set synaptic connections?
class Layer():

    STDP_INIT = 0.2
    STDP_DECAY = 0.3
    SYNAPSE_DECAY = 0.0001

    """
    Implements a single layer of Izhikevich spiking neurons

    The neurons in this layer are disjoint from one another with no
    synaptic connections between them. This is useful for representing
    things like sensory input and motor output.
    In short, this is basically Network with the synapses and associated
    STDP portions gutten. We keep the STDP values for the neurons so that
    connections can form autonomously with the recurrent network that this
    feeds into, or feeds from. Synapses will be managed by that layer, though
    so that all the STDP code can stay in one place.
    In the future this may be replaced with a Network that simply
    does not allow for connections between neurons in the same layer
    """
    def __init__(self, numEx = 0, numIn = 0):
        """
        Constructs an unconnected layer of spiking neurons

        Parameters:
        numEx (int): The number of excitatory neurons to generate
        numIn (int): The number of inhibitory neurons to generate
        """

        self.numEx = numEx
        self.numIn = numIn

        self.totalNum = numEx + numIn

        # For brevity
        tNum = self.totalNum

        r = stats.uniform.rvs(size=tNum)

        self.scale = numpy.ones(tNum)
        self.uSens = numpy.ones(tNum)
        self.reset = numpy.ones(tNum)
        self.uReset = numpy.ones(tNum)

        self.scale[0 : numEx] *= 0.02
        self.scale[numEx : ] *= 0.02 + 0.08 * r[numEx : ]
        self.uSens[0 : numEx] *= 0.2
        self.uSens[numEx : ] *= 0.25 - 0.05 * r[numEx : ]
        self.reset[0 : numEx] *= -65 + 15 * r[0 : numEx] ** 2
        self.reset[numEx : ] *= -65
        self.uReset[0 : numEx] *= 8 - 6 * r[0 : numEx] ** 2
        self.uReset[numEx : ] *= 2

        self.voltage = numpy.full(tNum, -65.0)
        self.recovery = numpy.multiply(self.voltage, self.uSens)
        self.input = numpy.zeros(tNum)

        self.stdp = numpy.zeros(tNum)

        self.pSpikes = numpy.where(self.voltage >= 30.0)[0]

        self.noisy = True


    def update(self, inputs = None):

        if inputs is not None:
            self.input = numpy.array(inputs)

        if self.noisy:
            self.input[0 : self.numEx] += 5.0 * stats.norm.rvs(size=self.numEx)
            self.input[self.numEx : ] += 2.0 * stats.norm.rvs(size=self.numIn)

        spikes = numpy.where(self.voltage >= 30.0)[0]

        self.voltage[spikes] = self.reset[spikes]
        self.recovery[spikes] += self.uReset[spikes]

        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.recovery += self.scale * (self.voltage * self.uSens - self.recovery)

        self.voltage[numpy.where(self.voltage >= 30.0)] = 30.0

        self.stdp *= Network.STDP_DECAY
        self.stdp[spikes] = Network.STDP_INIT

        self.pSpikes = spikes
        return spikes


class Network():

    # Constants used for STDP
    STDP_INIT = 0.2
    STDP_DECAY = 0.3
    SYNAPSE_DECAY = 0.0001

    """
    Implements the simple spiking model by Eugene Izhikevich

    Generates a randomly connected network of numEx + numIn neurons with
    random properties, as described in the paper
    'A Simple Model of Spiking Neurons' (2003).
    """
    def __init__(self, numEx, numIn, sensory=Layer(), motor=Layer()):
        """
        Constructs a randomly connected spiking neural network

        Parameters:
        numEx (int): The number of excitatory neurons to generate
        numIn (int): The number of inhibitory neurons to generate
        sensory (Layer): A layer of sensory neurons
        motor (Layer): A layer of motor neurons
        """

        self.sensory = sensory
        self.motor = motor

        self.numEx = numEx
        self.numIn = numIn
        tNum = numEx + numIn

        self.totalNum = tNum

        r = stats.uniform.rvs(size=(tNum))

        self.scale = numpy.ones(tNum)
        self.uSens = numpy.ones(tNum)
        self.reset = numpy.ones(tNum)
        self.uReset = numpy.ones(tNum)

        # Initialize neuron parameters.
        # This will create both excitatory and inhibitory neurons
        # as well as neurons with various types of behaviours. Neuron
        # behaviours are biased towards regular spiking, but allow for
        # chattering, fast spiking, and other behaviours
        self.scale[0 : numEx] *= 0.02
        self.scale[numEx : ] *= 0.02 + 0.08 * r[numEx : ]
        self.uSens[0 : numEx] *= 0.2
        self.uSens[numEx : ] *= 0.25 - 0.05 * r[numEx : ]
        self.reset[0 : numEx] *= -65 + 15 * r[0 : numEx] ** 2
        self.reset[numEx : ] *= -65
        self.uReset[0 : numEx] *= 8 - 6 * r[0 : numEx] ** 2
        self.uReset[numEx : ] *= 2

        self.inputSynapses = 0.5 * stats.uniform.rvs(size=(sensory.totalNum, tNum))
        self.inputSynapses[sensory.numEx :, :] *= -2.0

        self.outputSynapses = 0.5 * stats.uniform.rvs(size=(tNum, motor.totalNum))
        self.outputSynapses[numEx : ] *= -2.0

        # Recurrent network synapses
        self.synapses = 0.5 * stats.uniform.rvs(size=(tNum, tNum))
        self.synapses[numEx : ] *= -2.0

        self.voltage = numpy.full(tNum, -65.0)
        self.recovery = numpy.multiply(self.voltage, self.uSens)
        self.input = numpy.zeros(tNum)

        self.stdp = numpy.zeros(tNum)

        self.pSpikes = numpy.where(self.voltage >= 30.0)[0]
        self.noisy = True
        self.sensorySpikes = numpy.where(self.sensory.voltage >= 30.0)[0]
        self.motorSpikes = numpy.where(self.motor.voltage >= 30.0)[0]


    def update(self, inputs = None):
        """
        Updates the membrane potential for all neurons in the network

        Returns:
        list: The list of spikes which were generated in the previous timestep
        """

        spikes = numpy.where(self.voltage >= 30.0)[0]

        self.voltage[spikes] = self.reset[spikes]
        self.recovery[spikes] += self.uReset[spikes]

        # Add the sensory input to the input vector
        # It is assumed that the input vector is cleared/reset
        # OUTSIDE of the network. This may change in the future
        # but right now stochastic inputs are generated in agent.py
        if inputs is not None:
            inputs = numpy.array(inputs)
        inputSpikes = self.sensory.update(inputs)
        self.sensorySpikes = inputSpikes

        self.input = self.inputSynapses[inputSpikes, :].sum(axis=0)

        outputs = self.outputSynapses[spikes].sum(axis=0)
        motorSpikes = self.motor.update(outputs)
        self.motorSpikes = motorSpikes

        if self.noisy:
            self.input[0 : self.numEx] += 5.0 * stats.norm.rvs(size=self.numEx)
            self.input[self.numEx : ] += 2.0 * stats.norm.rvs(size=self.numIn)

        self.input += self.synapses[spikes, :].sum(axis=0)

        # Calculate two iterations for stability
        # This equation comes from Izhikevich "Simple Model of Spiking Neurons"
        # published in 2003.
        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.voltage += 0.5 * (0.04 * self.voltage ** 2 +
                       5.0 * self.voltage + 140 - self.recovery + self.input)
        self.recovery += self.scale * (self.voltage * self.uSens - self.recovery)

        self.voltage[numpy.where(self.voltage >= 30.0)] = 30.0


        # Calculate the next iteration of the decaying STDP constants for
        # neurons which have already fired
        self.stdp *= Network.STDP_DECAY

        # For any neurons that just spiked, setup the STDP values
        self.stdp[spikes] = Network.STDP_INIT

        for s in range(self.totalNum):
            for ps in range(self.sensory.totalNum):
                if s in spikes and self.sensory.stdp[ps]:
                    self.inputSynapses[ps, s] += self.sensory.stdp[ps] * numpy.sign(self.inputSynapses[ps, s])
                else:
                    self.inputSynapses[ps, s] -= Network.SYNAPSE_DECAY * self.inputSynapses[ps, s]


        # Iterate over all synaptic weights in order to update the weight
        # matrix. I feel like there is a better way to do this through numpy,
        # but I couldn't find anything online about it and most suggestions were
        # to either do what I am doing here, or to generate a list of all
        # possible indices... This seemed like the faster solution to get this
        # prototype running.
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

        for ps in range(self.totalNum):
            for s in range(self.motor.totalNum):
                if s in motorSpikes and self.stdp[ps]:
                    self.outputSynapses[ps, s] += self.stdp[ps] * numpy.sign(self.outputSynapses[ps, s])
                else:
                    self.outputSynapses[ps, s] -= Network.SYNAPSE_DECAY * self.outputSynapses[ps, s]


        self.synapses = numpy.clip(self.synapses, -1.0, 1.0)

        self.pSpikes = spikes
        return spikes
