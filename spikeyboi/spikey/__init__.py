import numpy as np
import pygame as pg

random = np.random.default_rng()

def fixed_update(self, *args, **kwargs):
    for sprite in self.sprites():
        if hasattr(sprite, 'fixed_update'):
            sprite.fixed_update(*args, **kwargs)

pg.sprite.Group.fixed_update = fixed_update
