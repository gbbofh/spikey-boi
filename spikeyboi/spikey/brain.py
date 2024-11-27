import spikeyboi.snn.network


class Brain():

    params = {
        'I_ext_adaptive': False,
        'd_max': 6,
        'A_plus': 0.01,
        'A_minus': 0.01 * 1.1,
        'r_min': -1,
        'r_max': 1,
        'F_t': 15,
    }

    def __init__(self, num_neurons: int, num_inputs: int, num_outputs: int):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.net = spikeyboi.snn.network.Network(num_neurons, params=Brain.params)
        self.inputs = self.net.I_inj[:num_inputs]
        self.outputs = self.net.firing_rates[num_inputs:num_inputs + num_outputs]
        self.rewards = self.net.reward

        self.input_first = 0
        self.input_last = num_inputs - 1
        self.output_first = num_inputs
        self.output_last = num_inputs + num_outputs - 1

    def update(self, delta_time):
        self.net.update()

    def reset(self):
        self.net.__init__(self.num_neurons)
