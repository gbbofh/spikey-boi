import io


import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import matplotlib.pyplot as plt
plt.style.use('dark_background')


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIDendrogramDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        self.buffer = None

        fig, ax = plt.subplots(1,1)

        self.fig = fig
        self.ax = ax
        self.time_accum = 10.0 # refresh now

        self.img = None
        self.original_rect = self.rect
        self.aspect_ratio = self.rect.width / self.rect.height

        self.REBUILD_EVERY = 10.0 # seconds

    def on_update(self, delta_time):
        # super().update(delta_time)

        self.time_accum += delta_time

        if self.time_accum >= self.REBUILD_EVERY:
            self.time_accum -= self.REBUILD_EVERY
            self.ax.clear()

            lm = sp.cluster.hierarchy.linkage(self.net.w, method='ward')
            sp.cluster.hierarchy.dendrogram(lm, ax=self.ax)

            buf = io.BytesIO()
            self.fig.savefig(buf, format='jpg')

            buf.seek(0)

            self.img = pg.image.load(buf, '.jpg')
            self.img = self.img.convert()

            self.disp_surf.image.fill((0,0,0))

            pg.transform.scale(self.img, self.disp_surf.image.size, self.disp_surf.image)

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
                pg.transform.scale(self.img, self.disp_surf.image.size, self.disp_surf.image)

                return True
        return super().process_event(e)

    def on_load_completed(self):
        self.net = self.sim.agent.brain.net
