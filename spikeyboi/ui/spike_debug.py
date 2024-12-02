import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UISpikeDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = pg.Surface((self.net.num_neurons, self.net.num_neurons), pg.SRCALPHA)
        # self.kernel = np.array([
        #     [ 0.25, 0.25, ],
        #     [ 0.25, 0.25, ],
        # ])
        self.kernel = 1/16 * np.array([
            [ 1,2,1 ],
            [ 2,4,2 ],
            [ 1,2,1 ],
        ])

        self.kernel_enabled = True
        self.data = np.zeros((self.net.num_neurons, self.net.num_neurons), dtype=np.float64)
        self.alpha = 1.0

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):
        # super().update(delta_time)

        # w = self.net.w * self.net.neuron_type[:, np.newaxis]
        # w_min = np.abs(np.min(w))
        # w = w + w_min
        # w_max = np.max(w)
        # w = w / w_max if w_max != 0 else w
        # w = w.T

        s = self.net.spikes
        s = np.outer(s[:, np.newaxis], s[np.newaxis, :]).astype(np.int32) * np.eye(self.net.num_neurons)

        # values = w
        self.data[:] = self.alpha * s + (1 - self.alpha) * self.data
        values = self.data
        if not (self.kernel is None) and self.kernel_enabled:
            values = sp.ndimage.convolve(s, self.kernel)

        rgba = spikeyboi.ui.colormaps['zebra'](values)

        self.buffer.fill((0,0,0,255))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        # rgba[mask,:] = 0

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

    def on_load_completed(self):
        self.net = self.sim.agent.brain.net
        self.data[:] = 0

    def on_agent_selected(self, agent):
        self.net = agent.brain.net
        self.data[:] = 0
