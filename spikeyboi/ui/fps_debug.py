import pygame as pg
import pygame_gui as gui


class UIFPSDebugger(gui.elements.UILabel):

    def __init__(self, rect, manager, anchors={'right': 'right'}):
        super().__init__(rect, '', manager, anchors=anchors)
        self.show()

    def update(self, delta_time):
        super().update(delta_time)

        fps = 1 / delta_time if delta_time != 0 else 999

        self.set_text(f'{fps=:.0f}')
