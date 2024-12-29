import io


import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import matplotlib.pyplot as plt
plt.style.use('dark_background')


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIGraphDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.time_accum = 10.0 # refresh now

        self.original_rect = self.rect
        self.aspect_ratio = self.rect.width / self.rect.height
        self.buffer = pg.Surface(self.rect, pg.SRCALPHA)

        self.REBUILD_EVERY = 0.2 # seconds

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):

        self.time_accum += delta_time

        self.disp_surf.image.fill((0,0,0))
        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

        if self.time_accum >= self.REBUILD_EVERY:
            self.time_accum -= self.REBUILD_EVERY

    def process_event(self, e):
        if e.type == gui.UI_WINDOW_RESIZED:
            if e.ui_element == self:
                w, h = self.get_abs_rect().size
                if w / h > self.aspect_ratio:
                    h = w / self.aspect_ratio
                else:
                    w = h * self.aspect_ratio
                self.set_dimensions((w,h))

                self.disp_surf.image.fill((0,0,0))
                self.time_accum = 10.0

                return True
        return super().process_event(e)

    def on_load_completed(self):
        self.net = self.sim.agent.brain.net
        self.time_accum = 10.0

    def on_agent_selected(self, agent):
        self.net = agent.brain.net
        self.time_accum = 10.0
