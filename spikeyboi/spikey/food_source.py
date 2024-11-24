import numpy as np
import pygame as pg


import spikeyboi.spikey
import spikeyboi.spikey.food


class FoodSource():

    def __init__(self, pos=(0,0), radius=50, max_food=50, max_life=100.0, *groups: pg.sprite.Group):

        self.pos = pos
        self.max_food = max_food
        self.lifetime = 0.0
        self.max_life = max_life
        self.radius = radius
        self.pool = []
        self.on_lifetime_exceeded_event = []
        self.on_object_moved_event = []

        r = spikeyboi.spikey.random.uniform(0, self.radius, max_food)
        a = spikeyboi.spikey.random.uniform(0, 2 * np.pi, max_food)
        x = self.pos[0] + r * np.cos(a)
        y = self.pos[1] + r * np.sin(a)
        # lifetime = spikeyboi.spikey.random.uniform(30.0, 45.0, max_food)
        lifetime = np.ones(max_food) * -1

        for i in range(max_food):
            food = spikeyboi.spikey.food.Food(lifetime[i], groups)
            food.rect.x, food.rect.y = x[i], y[i]
            food.on_collision_event.append(self.on_food_collision)
            food.on_lifetime_exceeded_event.append(self.on_food_collision)
            print(food.rect)

        self.groups = groups

    def spawn_food(self, f):
        r = spikeyboi.spikey.random.uniform(0, self.radius)
        a = spikeyboi.spikey.random.uniform(0, 2 * np.pi)
        x = self.pos[0] + r * np.cos(a)
        y = self.pos[1] + r * np.sin(a)

        lifetime = spikeyboi.spikey.random.uniform(1.0, 5.0)
        f.rect.x = x
        f.rect.y = y
        f.max_life = lifetime if f.max_life > -1 else -1

        str = f'spawning food @ {f.rect.topleft}'
        print(str)

        spikeyboi.spikey.sim_instance.render_list.add(f)
        spikeyboi.spikey.sim_instance.physics_group.add(f)

    def update(self, delta_time):
        self.lifetime += delta_time
        if self.max_life < 0:
            self.lifetime = self.max_life - 1

        if self.lifetime >= self.max_life:
            self.lifetime = 0
            for e in self.on_lifetime_exceeded_event:
                e(self)

        while len(self.pool) > 0:
            f = self.pool.pop()
            self.spawn_food(f)

    def on_food_collision(self, food):
        self.pool.append(food)
        print(self.pool)
        for e in self.on_object_moved_event:
            e(food)
