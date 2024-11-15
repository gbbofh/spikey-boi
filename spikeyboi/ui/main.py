import pygame as pg
import pygame_gui as gui


class MainUI(gui.elements.UIPanel):

    def __init__(self, rel_rect, manager):
        super().__init__(rel_rect, manager)

        main_surf_size = self.get_container().get_size()
        self.main_surf = gui.elements.UIImage(pg.Rect((0,0), main_surf_size)
                                                pg.Surface(main_surf_size).convert(),
                                                manager=manager,
                                                container=self,
                                                parent_element=self)
        self.sim = None

    def process_event(self, e):
        handled = super().process_event(event)
        return handled

    def update(self, delta_time):
        if self.alive():
            # TODO: Update sim
            pass
        super().update(delta_time)
        self.sim.draw(self.main_surf.image)

