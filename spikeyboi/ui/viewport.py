import pygame as pg
import pygame_gui as gui

from typing import Union, Tuple


import spikeyboi.spikey.sim


class UIViewport(gui.elements.UIPanel):

    def __init__(self, relative_rect: pg.Rect, manager: gui.UIManager, *args, **kwargs):
        super().__init__(relative_rect, manager=manager, *args, **kwargs)

        self.sim = spikeyboi.spikey.sim.Simulation(self.get_container().get_size())

        surf_size = self.get_container().get_size()
        surf_rect = pg.Rect((0,0),surf_size)
        surf_buffer = pg.Surface(surf_size)
        self.sim_surf = gui.elements.UIImage(surf_rect,surf_buffer,
                                            manager=manager,container=self,
                                            parent_element=self)

        self.buffer = pg.Surface(relative_rect.size, pg.SRCALPHA)

    def update(self, delta_time):
        super().update(delta_time)

        # self.sim.update(delta_time)
        self.sim.draw(self.sim_surf.image)

    def fixed_update(self, fixed_delta):
        self.sim.update(delta_time)

    # def draw(self, surf):
    #     self.buffer.fill((0,0,0,0))

    #     self.sim.draw(self.buffer)

    #     surf.blit(self.buffer, self.relative_rect.topleft)

