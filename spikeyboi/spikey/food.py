import numpy as np
import pygame as pg


import spikeyboi.spikey


class Food(pg.sprite.Sprite):

    def __init__(self, max_life=10.0, *group):
        super().__init__(group)
        self.color = np.array((50, 200, 100))

        self.image = pg.Surface((15,15), pg.SRCALPHA)
        self.rect = self.image.get_rect()

        self.color[1] = spikeyboi.spikey.random.integers(120, 200)

        pg.draw.circle(self.image,self.color,self.rect.center,7.5)

        self.lifetime = 0.0
        self.max_life = max_life
        self.on_collision_event = []
        self.on_lifetime_exceeded_event = []

    def update(self, delta_time):
        self.lifetime += delta_time
        render_list = spikeyboi.spikey.sim_instance.render_list
        # render_list = self.get_render_group()
        if self.lifetime >= self.max_life and render_list is not None:
            self.on_lifetime_exceeded()

    def on_collision(self, other):
        if type(other) == type(self):
            return

        # render_list = self.get_render_group()
        # render_list.remove(self)
        render_list = spikeyboi.spikey.sim_instance.render_list
        render_list.remove(self)
        spikeyboi.spikey.sim_instance.physics_group.remove(self)

        for e in self.on_collision_event:
            e(self)

    def on_lifetime_exceeded(self):
        render_list = self.get_render_group()
        render_list.remove(self)
        self.lifetime = 0.0
        for e in self.on_lifetime_exceeded_event:
            e(self)

        # for e in self.on_collision_event:
        #     self.on_collision_event(self)

    def get_render_group(self):
        for group in self.groups():
            if isinstance(group, pg.sprite.RenderUpdates):
                return group
        return None
