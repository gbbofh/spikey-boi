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
        super().__init__(rect, manager, title)

        surf_size = self.get_container().get_size()
        surf_rect = pg.Rect((0,0), surf_size)
        surf_buffer = pg.Surface(surf_size, pg.SRCALPHA)
        self.disp_surf = gui.elements.UIImage(surf_rect, surf_buffer,
                                            manager=manager,
                                            container=self,
                                            parent_element=self)

        self.sim = spikeyboi.spikey.sim_instance
        self.kernel = None

    def process_event(self, e):
        handled = super().process_event(e)
        return handled

    def update(self, delta_time):
        pass

        # net : spikeyboi.snn.network.Network = self.sim.agent.brain.net

        # weights = net.w
        # types = net.neuron_type

        # weights = weights * types[:, np.newaxis]
