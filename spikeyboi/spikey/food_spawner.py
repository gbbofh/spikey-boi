import numpy as np
import pygame as pg

import spikeyboi.spikey
import spikeyboi.spikey.food


class FoodSpawner(pg.sprite.Sprite):

    def __init__(self, area : pg.Rect, max_food = 25, spawn_rate = 5.0):
        super().__init__()

        self.area = area
        self.max_food = max_food
        self.spawn_rate = spawn_rate
        self.food_lifetime_min = 30
        self.food_lifetime_max = 120

        self.pool = []

        lifetime = spikeyboi.spikey.random.uniform(self.food_lifetime_min, self.food_lifetime_max, max_food)

        for i in range(max_food):
            f = spikeyboi.spikey.food.Food(lifetime[i])
            f.on_lifetime_exceeded_event.append(self.on_food_expired)
            f.on_collision_event.append(self.on_food_eaten)
            f.id = i

            self.pool.append(f)

        self.time_accum = spawn_rate

        self.sim = spikeyboi.spikey.sim_instance

    def spawn_food(self):
        if len(self.pool):
            x = spikeyboi.spikey.random.integers(self.area.x, self.area.w)
            y = spikeyboi.spikey.random.integers(self.area.y, self.area.h)
            lifetime = spikeyboi.spikey.random.uniform(self.food_lifetime_min, self.food_lifetime_max)

            f : spikeyboi.spikey.food.Food = self.pool.pop()
            print(f'Spawning food ({f.id})')

            f.rect.x = x
            f.rect.y = y
            f.max_life = lifetime
            f.lifetime = 0.0

            food_group = self.sim.food_group
            entities = self.sim.all_entities
            physics_group = self.sim.physics_group
            render_list = self.sim.render_list

            f.add((food_group, entities, physics_group, render_list))


    def update(self, delta_time):
        self.time_accum += delta_time

        if self.time_accum >= self.spawn_rate:
            self.time_accum -= self.spawn_rate
            self.spawn_food()

    def on_food_eaten(self, food, agent):
        print(f'Agent ({agent.id}) ate food ({food.id})')
        self.sim.food_group.remove(food)
        self.sim.all_entities.remove(food)
        self.sim.render_list.remove(food)
        self.sim.physics_group.remove(food)

        self.pool.append(food)

    def on_food_expired(self, food):
        print(f'Food ({food.id}) lifetime exceeded')
        self.sim.food_group.remove(food)
        self.sim.all_entities.remove(food)
        self.sim.render_list.remove(food)
        self.sim.physics_group.remove(food)

        self.pool.append(food)

