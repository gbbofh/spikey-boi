import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIHistogramDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.index = 0
        self.tick_count = 0
        self.max_index = self.net.SPIKE_WINDOW

        self.buffer = pg.Surface((self.max_index, self.net.num_neurons), pg.SRCALPHA)
        self.data = np.zeros((self.max_index, self.net.num_neurons), dtype=np.float64)

        self.original_rect = self.rect
        self.aspect_ratio = self.rect.width / self.rect.height

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):
        agent = spikeyboi.spikey.sim_instance.agent

        self.tick_count = (self.tick_count + 1) % agent.UPDATE_TICKS_MAX

        if self.tick_count % agent.UPDATE_TICKS_MAX:
            return

        s = self.net.spikes

        self.data[self.index] = s

        rgba = spikeyboi.ui.colormaps['zebra'](self.data)

        rgba[self.index,:,:-1] = (255,100,100) + (255 - rgba[self.index,:,:-1])

        self.buffer.fill((0,0,0,255))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

        self.index = (self.index + 1) % self.max_index

    # def fill_past_data(self):
    #     for i in range(self.index):
    #         trace_ind = (self.index - self.max_index + i) % self.max_index
    #         self.data[i] = self.net.spike_trace[:, trace_ind]
    #         # n = self.net.SPIKE_WINDOW
    #         # self.data[i] = self.net.spike_trace[:, n - i - 1]
    #     # self.data[:self.index] = self.net.spike_trace[:,-self.index:].T

    # def process_event(self, e):
    #     if e.type == gui.UI_WINDOW_RESIZED:
    #         if e.ui_element == self:
    #             w, h = self.get_abs_rect().size
    #             if w / h > self.aspect_ratio:
    #                 h = w / self.aspect_ratio
    #             else:
    #                 w = h * self.aspect_ratio
    #             self.set_dimensions((w,h))

    #             self.on_update(0)

    #             return True
    #     return super().process_event(e)

    def on_load_completed(self):
        self.net = self.sim.agent.brain.net
        self.data[:] = 0
        self.index = 0
        # self.fill_past_data()

    def on_agent_selected(self, agent):
        self.net = agent.brain.net
        self.data[:] = 0
        self.index = 0
        # self.fill_past_data()
        self.on_update(0)
