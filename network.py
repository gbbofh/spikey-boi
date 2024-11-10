import numpy as np
import scipy.ndimage as ndi


from types import SimpleNamespace


import util


class Network():

    default_params = {
        # Neuron parameters
        'tau_m': 20.0,
        'v_rest': -65.0,
        'v_reset': -70.0,
        'v_threshold': -50.0,
        'R_m': 10.0,
        'F_t': 15.0,

        # Synapse parameters
        'P_syn': 0.4,
        'P_syn_gen': 0.00005,
        'tau_s': 5.0,
        'w_I': 0.05, # Orig - 0.05
        'S_decay': 0.00001,
        'tau_scale': 50.0,
        'eta_s': 0.05,

        # Conductance delay parameters
        'd_min': 1,
        'd_max': 10,

        # STDP parameters
        'A_plus': 0.01, # orig 0.01
        'A_minus': 0.012, # orig 0.012
        'tau_plus': 10.0,
        'tau_minus': 20.0,

        # Reward parameters
        'r_scale_factor': 1.0,
        'r_asymmetry': 0.8,
        'tau_r': 200.0,
        'r_min': -1.5,
        'r_max': 1.5,

        # Eligibility trace parameters
        'tau_e': 500.0,

        # Subthreshold activity parameters
        'sigmoid_grad': 3,

        # Gaussian noise parameters
        'I_ext_mean': 1.5,
        'I_ext_std': 0.5, # orig 0.5
        'I_ext_enable': True,
        'I_ext_adaptive': True,
        'I_ext_adaptive_scale': 0.05,

        # Spike correlation parameters
        'min_correlation': 0.7,

        # Time parameters
        'dt': 0.1,
    }

    SPIKE_WINDOW = 1000

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.I_total = np.zeros(self.num_neurons)

    def __init__(self, n_neurons=10, ratio=0.8, params=default_params):

        self.num_neurons = n_neurons
        self.num_exc = int(ratio * n_neurons)
        self.num_inh = n_neurons - self.num_exc

        # SimpleNamespace acts as a dictionary with a member access operator.
        self.params = SimpleNamespace(**Network.default_params)

        for k, v in params.items():
            setattr(self.params, k, v)

        p = self.params

        # membrane potential
        self.v_m = np.full(n_neurons, p.v_rest)

        # manage excitatory / inhibitory neurons
        # without having to calculate using
        # indices all the time
        neuron_type = np.ones(n_neurons)
        neuron_type[self.num_exc:] = -1
        self.neuron_type = neuron_type

        # random probability matrix for sparse connectivity
        # wp = np.random.random(size=(n_neurons, n_neurons))
        wp = util.random.random(size=(n_neurons, n_neurons))

        # generate synaptic matrix
        w = np.random.uniform(0.1, 0.5, (n_neurons, n_neurons))

        # Create inhibitory synapses
        # w[self.num_exc:, :] *= -1

        # prune synapses
        w[wp < (1 - p.P_syn)] = 0

        self.w = w
        # print(self.w[:,-2])

        # Synaptic conductance delays
        # delays = np.array(np.random.randint(p.d_min, p.d_max,
        #                             (n_neurons, n_neurons)) / p.dt,
        #                             dtype=np.int16)
        delays = np.array(util.random.integers(p.d_min, p.d_max,
                                    (n_neurons, n_neurons)) / p.dt,
                                    dtype=np.int16)
        max_delay = int(np.max(delays))
        spike_buffer = np.zeros((n_neurons, max_delay))

        self.delays = delays
        self.max_delay = max_delay
        self.spike_buffer = spike_buffer

        # Spike record
        spikes = np.zeros(n_neurons, dtype=bool)
        self.spikes = spikes

        # Trace of the last 1000 cycles of spikes
        # Each cycle is 0.1 ms of simulated time, so this trace is over
        # 0.1 seconds of simulated time
        self.spike_trace = np.zeros((n_neurons, Network.SPIKE_WINDOW))
        self.spike_weight = np.tile(np.linspace(0, 1, Network.SPIKE_WINDOW), (n_neurons, 1))
        self.firing_rates = np.zeros(n_neurons)

        # Hebbian plasticity / Structural pasticity
        spike_pairs = np.zeros((n_neurons, n_neurons))
        self.spike_pairs = spike_pairs

        # Neuron input current
        self.I_ext = np.zeros(n_neurons)
        self.I_syn = np.zeros(n_neurons)
        self.I_inj = np.zeros(n_neurons)
        self.I_total = np.zeros(n_neurons)

        # Traces for LTD and LTP
        self.P_pre = np.zeros(n_neurons)
        self.P_post = np.zeros(n_neurons)

        self.V_pre = np.zeros((n_neurons, n_neurons))
        self.V_post = np.zeros((n_neurons, n_neurons))

        # Reward modulation
        self.reward = np.zeros((n_neurons, n_neurons))
        self.last_reward = np.zeros((n_neurons, n_neurons))

        # Reward eligibility
        self.E_syn = np.zeros((n_neurons, n_neurons))
        self.T_syn = np.zeros((n_neurons, n_neurons), dtype=np.int16)

        self.dw = np.zeros((n_neurons, n_neurons))

        self.time_acc = 0

    def update(self, input=None):
        p = self.params
        nN = self.num_neurons
        nE = self.num_exc
        nI = self.num_inh

        self.time_acc += p.dt

        # self.I_ext[:] = (p.I_ext_mean + np.random.normal(0, p.I_ext_std, nN)) * int(p.I_ext_enable)
        self.I_ext[:] = (p.I_ext_mean + util.random.normal(0, p.I_ext_std, nN)) * int(p.I_ext_enable)
        self.I_syn[:] *= np.exp(-p.dt / p.tau_s)

        self.I_ext[:] += p.I_ext_adaptive_scale * (p.F_t - self.firing_rates) / p.F_t * int(p.I_ext_enable) * int(p.I_ext_adaptive)

        # Roll spike buffer to advance spikes for synaptic conductance
        self.spike_buffer[:, :] = np.roll(self.spike_buffer, -1)
        self.spike_buffer[:, -1:] = 0

        # Skipping on computing delay properly for now
        # Need to figure out a good implementation
        # self.spike_buffer[:, self.max_delay - 1] = self.spikes[:]

        # I think this works correctly?
        for i in range(nN):
            self.spike_buffer[i, self.delays[i] - 1] = self.spikes[i]

        # for i in range(nN): # Post
        #     for j in range(nN): # Pre
        #         self.I_syn[i] += self.w[j, i] * self.spike_buffer[j, 0] * self.neuron_type[j]

        # arrived = self.spike_buffer[:, 0]
        # self.I_syn += self.w.sum(axis=0) * arrived * self.neuron_type

        mod = self.spike_buffer[:, 0] * self.neuron_type
        self.I_syn += np.dot(mod, self.w)

        # r = np.random.random() > (1 - p.P_syn_gen)
        r = util.random.random() > (1 - p.P_syn_gen)

        # i = np.random.randint(0, self.num_neurons) * int(r)
        i = util.random.integers(0, self.num_neurons) * int(r)
        # j = np.random.randint(0, self.num_neurons) * int(r)
        j = util.random.integers(0, self.num_neurons) * int(r)

        c = (i != j) and (self.w[i, j] == 0)

        self.w[i,j] += p.w_I if c else 0
        if c:
            print(f'New Synapse (Random): {i} -> {j}')

        dt_factor = p.dt / Network.default_params['dt']
        dir_mask = self.P_pre[:, np.newaxis] > self.P_pre

        mask = np.logical_and(self.w == 0, self.w.T == 0)
        mask = np.logical_and(mask, np.eye(self.num_neurons) == 0)

        self.spike_pairs[mask] *= 0.99 ** dt_factor
        spike_mask = np.outer(self.spikes, self.spikes)
        spike_update_mask = mask & spike_mask

        self.spike_pairs[spike_update_mask] += 0.5 * dt_factor

        corr_mask = self.spike_pairs >= p.min_correlation
        final_mask = mask & corr_mask

        unidirectional_mask = np.logical_and(final_mask, dir_mask)

        self.w[unidirectional_mask] += p.w_I
        # self.w[final_mask] = p.w_I
        # if final_mask.any():
        if unidirectional_mask.any():
            print(f'New Synapse (Correlation): {np.where(unidirectional_mask)}')

        self.spikes[:] = 0

        self.I_total[:] = self.I_ext + self.I_syn + self.I_inj
        I_total = self.I_total
        dv_m = (p.v_rest - self.v_m + p.R_m * I_total) * (p.dt / p.tau_m)
        self.v_m += dv_m

        pspike = self.v_m >= p.v_threshold
        self.v_m[pspike] = p.v_reset
        self.spikes[pspike] = 1

        self.spike_trace[:] = np.roll(self.spike_trace, -1)
        self.spike_trace[:, -1:] = self.spikes[:, np.newaxis]

        self.firing_rates[:] = 1000 * (self.spike_trace * self.spike_weight).sum(axis=1) / Network.SPIKE_WINDOW / p.dt

        # Keep track of when spikes occurred for STDP
        self.P_pre[:] *= np.exp(-p.dt / p.tau_plus)
        self.P_post[:] *= np.exp(-p.dt / p.tau_minus)

        self.dw[:] = 0

        # decay synaptic connections
        # self.w *= (1 - p.S_decay)

        self.P_pre[self.spikes] += 1
        self.P_post[self.spikes] += 1

        self.V_pre[:] = np.outer(self.v_m, self.spikes)
        self.V_post[:] = np.outer(self.spikes, self.v_m)

        V_pre = self.V_pre
        V_post = self.V_post

        self.E_syn += np.outer(self.P_pre, self.P_post)

        sigmoid_pre = 1 / (1 + np.exp(p.sigmoid_grad * (V_pre - p.v_threshold)))
        sigmoid_post = 1 / (1 + np.exp(p.sigmoid_grad * (V_post - p.v_threshold)))
        sigmoid_eligibility = 2 / (1 + np.exp(-p.sigmoid_grad * self.E_syn)) - 1

        ltp_dw = (1 + p.r_scale_factor * self.reward * sigmoid_eligibility) * sigmoid_pre * sigmoid_post * p.A_plus * np.outer(self.P_pre, self.spikes)
        ltp_mask = np.outer(np.ones(nN, dtype=bool), self.spikes) & (~np.eye(nN, dtype=bool))
        self.dw += ltp_dw * ltp_mask

        ltd_dw = -(1 + p.r_asymmetry * p.r_scale_factor * self.reward * sigmoid_eligibility) * sigmoid_pre * sigmoid_post * p.A_minus * np.outer(self.spikes, self.P_post)
        ltd_mask = np.outer(self.spikes, np.ones(nN, dtype=bool)) & (~np.eye(nN, dtype=bool))
        self.dw += ltd_dw * ltd_mask

        self.w += self.dw

        self.reward[:] = np.clip(self.reward, p.r_min, p.r_max)

        self.last_reward[:] = self.reward
        self.reward[:] *= np.exp(-p.dt / p.tau_r)

        self.E_syn[:] *= np.exp(-p.dt / p.tau_e)

        # Synaptic scaling for homeostasis
        if self.time_acc > p.tau_scale:
            self.time_acc = 0
            scale = 1 + p.eta_s * (p.F_t - self.firing_rates) / p.F_t
            self.w *= scale[np.newaxis, :]
            # Didn't work quite as well..
            # norm = np.linalg.norm(self.w, axis=0)
            # self.w[:, norm > 0] *= 2 / norm[norm > 0]

        self.w = np.clip(self.w, 0, 1)

    def toggle_noise(self):
        self.params.I_ext_enable = not self.params.I_ext_enable

    def toggle_adaptive_noise(self):
        self.params.I_ext_adaptive = not self.params.I_ext_adaptive

