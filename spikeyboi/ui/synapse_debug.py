import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UISynapseDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = pg.Surface((self.net.num_neurons, self.net.num_neurons), pg.SRCALPHA)
        self.kernel = np.array([
            [ 0.25, 0.25, ],
            [ 0.25, 0.25, ],
        ])

        self.kernel_enabled = True

    def update(self, delta_time):
        super().update(delta_time)

        w = (self.net.w * self.net.neuron_type[:, np.newaxis] + 1) / 2
        w = w.T

        values = w
        if not (self.kernel is None) and self.kernel_enabled:
            values = sp.ndimage.convolve(w, self.kernel)

        rgba = spikeyboi.ui.colormaps['rdgr'](values)

        self.buffer.fill((0,0,0,0))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)


