import numpy as np
import pygame as pg
import pygame_gui as gui


import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.brain


import spikeyboi.snn.network


class UIDebugWindow(gui.elements.UIWindow):

    def __init__(self, title, rect, manager):
        rect = pg.Rect(rect)
        super().__init__(rect, manager, title, visible=False, always_on_top=True)

        surf_size = self.get_container().get_size()
        surf_rect = pg.Rect((0,0), surf_size)
        surf_buffer = pg.Surface(surf_size, pg.SRCALPHA)
        self.disp_surf = gui.elements.UIImage(surf_rect, surf_buffer,
                                            manager=manager,
                                            container=self,
                                            parent_element=self)

        self.sim = spikeyboi.spikey.sim_instance
        self.kernel = None

    def update(self, delta_time):
        super().update(delta_time)

    def on_close_window_button_pressed(self):
        self.hide()

