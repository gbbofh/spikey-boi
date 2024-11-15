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
        self.net = spikeyboi.snn.network.Network(num_neurons, params=Brain.params)
        self.inputs = self.net.I_inj[:num_inputs + 1]
        self.outputs = self.net.firing_rates[num_inputs:num_inputs + num_outputs + 1]

    def update(self, delta_time):
        self.net.update()
