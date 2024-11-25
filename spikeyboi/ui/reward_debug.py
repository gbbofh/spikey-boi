import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi
import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIRewardDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = pg.Surface((self.net.num_neurons, self.net.num_neurons), pg.SRCALPHA)
        self.kernel = np.array([
            [ 0.25, 0.25, ],
            [ 0.25, 0.25, ],
        ])

        self.kernel_enabled = True

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):
        # super().update(delta_time)

        r = self.net.reward.copy().T
        r_min = self.net.params.r_min

        r += np.abs(r_min)
        r /= 2 * self.net.params.r_max

        # w = (self.net.w * self.net.neuron_type[:, np.newaxis] + 1) / 2
        # w = w.T

        values = r
        if not (self.kernel is None) and self.kernel_enabled:
            values = sp.ndimage.convolve(r, self.kernel)

        rgba = spikeyboi.ui.colormaps['rdbu'](values)

        self.buffer.fill((0,0,0,0))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

    def on_load_completed(self):
        self.net = self.sim.agent.brain.net

    def on_agent_selected(self, agent):
        self.net = agent.brain.net
