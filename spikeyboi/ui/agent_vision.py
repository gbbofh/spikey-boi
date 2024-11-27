import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIAgentVisionDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = self.sim.agent.brain

        # First 7 inputs are raycast results
        self.buffer = pg.Surface((self.brain.num_inputs - 3, self.brain.num_inputs - 3), pg.SRCALPHA)
        self.kernel = np.array([
            [ 0.25, 0.25, ],
            [ 0.25, 0.25, ],
        ])

        self.kernel_enabled = True
        self.data = np.zeros((self.brain.num_inputs - 3, self.brain.num_inputs - 3), dtype=np.float64)
        self.alpha = 1.0

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):
        self.data[:] = self.brain.inputs[:7, np.newaxis]
        values = self.data

        rgba = spikeyboi.ui.colormaps['zebra'](values)

        self.buffer.fill((0,0,0,255))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

    def on_load_completed(self):
        self.brain = self.sim.agent.brain
        self.data[:] = 0

    def on_agent_selected(self, agent):
        self.brain = agent.brain
        self.data[:] = 0
