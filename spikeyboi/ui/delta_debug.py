import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIDeltaDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = pg.Surface((self.net.num_neurons, self.net.num_neurons), pg.SRCALPHA)
        self.kernel = np.array([
            [ 0.25, 0.25, ],
            [ 0.25, 0.25, ],
        ])

        self.kernel_enabled = True
        self.data = np.zeros_like(self.net.dw)
        self.alpha = 0.01

    def update(self, delta_time):
        super().update(delta_time)

        # dw = (self.net.dw + 1) / 2
        dw = self.net.dw.copy().T
        dw_min = np.abs(np.min(dw))
        dw = dw + dw_min
        dw_max = np.max(dw)
        dw = dw / dw_max if dw_max > 0 else dw

        # E_syn = E_syn / E_max if E_max > 0 else E_syn

        self.data[:] = self.alpha * dw + (1 - self.alpha) * self.data
        values = self.data
        if not (self.kernel is None) and self.kernel_enabled:
            values = sp.ndimage.convolve(self.data, self.kernel)

        rgba = spikeyboi.ui.colormaps['magma'](values)

        self.buffer.fill((0,0,0,0))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)


