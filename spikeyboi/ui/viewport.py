import pygame as pg
import pygame_gui as gui

from typing import Union, Tuple


import spikeyboi.spikey.sim
import spikeyboi.spikey.agent


UI_AGENT_SELECTED = pg.event.custom_type()


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

        self.buffer = pg.Surface(self.get_container().get_size(), pg.SRCALPHA)

    def update(self, delta_time):
        super().update(delta_time)

        self.buffer.fill((0,0,0,0))
        self.sim.draw(self.buffer)

        self.sim_surf.image.fill((0,0,0,255))
        self.sim_surf.image.blit(self.buffer, (0,0))

    def fixed_update(self, fixed_delta):
        self.sim.update(delta_time)

    def _accept_click(self, e):
        if self.sim_surf.rect.collidepoint(e.pos):
            x = e.pos[0] - self.sim_surf.rect.left
            y = e.pos[1] - self.sim_surf.rect.top
            x = x - 5
            y = y - 5
            w = 10
            h = 10
            rect = pg.Rect(x,y,w,h)
            objects = self.sim.quadtree.hit(rect)
            for obj in objects:
                if type(obj) is spikeyboi.spikey.agent.Agent:
                    event_data = {'pos': obj.rect.center, 'agent': obj}
                    pg.event.post(pg.event.Event(UI_AGENT_SELECTED, event_data))
                    return True
        return False

    def process_event(self, e):
        if e.type == pg.MOUSEBUTTONUP and e.button == 1:
            return self._accept_click(e)

