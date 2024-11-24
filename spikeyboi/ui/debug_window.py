import numpy as np
import pygame as pg
import pygame_gui as gui


import spikeyboi
import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.brain


import spikeyboi.snn.network


class UIDebugWindow(gui.elements.UIWindow):

    def __init__(self, title, rect, manager):
        # print(title, rect, manager)
        rect = pg.Rect(rect)
        super().__init__(rect, manager, title, visible=False, always_on_top=False, resizable=True)

        surf_size = self.get_container().get_size()
        surf_rect = pg.Rect((10,10), (surf_size[0] - 20, surf_size[1] - 20))
        surf_buffer = pg.Surface(surf_size, pg.SRCALPHA)
        self.disp_surf = gui.elements.UIImage(surf_rect, surf_buffer,
                                            manager=manager,
                                            container=self.get_container(),
                                            parent_element=self,
                                            anchors={'left': 'left', 'right': 'right', 'top': 'top', 'bottom': 'bottom'})

        self.sim = spikeyboi.spikey.sim_instance
        self.kernel = None

        spikeyboi.app_instance.on_load_completed_event.append(self.on_load_completed)

    def update(self, delta_time):
        super().update(delta_time)

        if self.visible:
            self.on_update(delta_time)

    def on_update(self, delta_time):
        pass

    def on_close_window_button_pressed(self):
        self.hide()

    def process_event(self, e):
        if e.type == gui.UI_WINDOW_RESIZED:
            if e.ui_element == self:
                nw, nh = self.get_abs_rect().size
                ns = max(nw, nh)
                self.set_dimensions((ns, ns))
        return super().process_event(e)

    def on_load_completed(self):
        pass
