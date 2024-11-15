import pygame as pg
import pygame_gui as gui


class UIMenuBar(gui.elements.UIPanel):

    def __init__(self, rel_rect, manager, data):
        super().__init__(rel_rect, manager=manager)

        self.menu_buttons = {}
        self.dropdowns = {}

        button_x = 10
        for name, options in data.items():
            rect = pg.Rect((button_x,0), (80,rel_rect.height))
            btn = gui.elements.UIButton(rect, name, manager=manager, container=self)
            self.menu_buttons[name] = btn

            rect = pg.Rect((button_x,rel_rect.height), (120,30*len(options)))
            panel = gui.elements.UIPanel(rect, manager=manager, visible=False)
            self.dropdowns[name] = panel
            for i, opt in enumerate(options):
                pass
