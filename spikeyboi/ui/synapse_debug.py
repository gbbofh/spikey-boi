import numpy as np
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UISynapseDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = pg.Surface((self.net.num_neurons, self.net.num_neurons), pg.SRCALPHA)

    def update(self, delta_time):

        w = (self.net.w * self.net.neuron_type[:, np.newaxis] + 1) / 2
        rgba = spikeyboi.ui.colormaps['rdgr'](w)

        if self.kernel:
            np.convolve(

        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha()
        alpha[:] = rgba[:,:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)


