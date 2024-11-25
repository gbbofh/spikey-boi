import numpy as np
import pygame as pg


import spikeyboi.spikey
import spikeyboi.spikey.food


class FoodSource():

    def __init__(self, rect, max_food=50, *groups: pg.sprite.Group):

        self.rect = rect
        self.max_food = max_food

        x = spikeyboi.spikey.random.uniform(self.rect.x, self.rect.width, max_food)
        y = spikeyboi.spikey.random.uniform(self.rect.y, self.rect.height, max_food)

        for i in range(max_food):
            food = spikeyboi.spikey.food.Food(lifetime[i], groups)

            food.on_collision_event.append(self.on_food_collision)
            food.on_lifetime_exceeded_event.append(self.on_food_expired)

        self.groups = groups

    def update(self, delta_time):
        while len(self.pool) > 0:
            f = self.pool.pop()
            self.spawn_food(f)

    def spawn_food(self):
        food.rect.x = x[i]
        food.rect.y = y[i]

    def on_food_collision(self, food):
        pass

    def on_food_expired(self, food):
        pass
