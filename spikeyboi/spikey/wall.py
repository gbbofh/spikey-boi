import numpy as np
import pygame as pg


class Wall(pg.sprite.Sprite):

    def __init__(self, rect, color=(100,100,100)):
        super().__init__()

        rect = pg.Rect(rect)
        self.rect = rect
        self.image = pg.Surface(rect.size)
        self.image.fill(color)
        self.mask = pg.mask.from_surface(self.image)
        self.color = np.array(color)
        self.is_static = True
