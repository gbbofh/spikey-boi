import numpy as np
import pygame as pg


import time
import pickle


import spikeyboi
import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.food
import spikeyboi.spikey.food_source
import spikeyboi.spikey.physics
import spikeyboi.spikey.quadtree
import spikeyboi.spikey.wall


class Simulation():

    def __init__(self, size=(800,600)):
        spikeyboi.spikey.sim_instance = self
        self.app = spikeyboi.app_instance

        self.rng_seed = int(time.time())
        spikeyboi.spikey.random = np.random.default_rng(self.rng_seed)

        self.render_list = pg.sprite.RenderUpdates()
        self.agent_group = pg.sprite.Group()
        self.food_group = pg.sprite.Group()
        self.physics_group = pg.sprite.Group()
        self.all_entities = pg.sprite.Group()

        agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
        agent.x, agent.y = 100, 200
        agent.rect.x, agent.rect.y = agent.x, agent.y
        self.agent = agent

        # agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
        # agent.x, agent.y = 200, 200
        # agent.rect.x, agent.rect.y = agent.x, agent.y
        # self.agent2 = agent

        w,h = size
        center = w // 2,h // 2
        self.food_source = spikeyboi.spikey.food_source.FoodSource(center, 200, 50, -1.0, self.food_group, self.all_entities, self.physics_group, self.render_list)
        self.food_source.on_lifetime_exceeded_event.append(self.on_food_source_lifetime_exceeded)

        wall_thickness = 50

        north = spikeyboi.spikey.wall.Wall((0,0,800,wall_thickness))
        east = spikeyboi.spikey.wall.Wall((800 - wall_thickness,0,wall_thickness,600))
        south = spikeyboi.spikey.wall.Wall((0,600 - 1.5 * wall_thickness,800,wall_thickness))
        west = spikeyboi.spikey.wall.Wall((0,0,wall_thickness,600))
        self.walls = [north, east, south, west]

        self.render_list.add(self.walls)
        self.all_entities.add(self.walls)
        self.physics_group.add(self.walls)

        self.quadtree = spikeyboi.spikey.quadtree.QuadTree(list(self.physics_group), (0,0,800,600), 3)
        self.physics = spikeyboi.spikey.physics.Physics(self.physics_group)

        self.time = 0.0
        self.fixed_delta_time = 0.03
        self.background_color = np.zeros(3)
        self.rect = pg.Rect((0,0),size)
        self.size = size

        self.app.on_save_event.append(self.on_save)
        self.app.on_load_event.append(self.on_load)

    def update(self, time_delta):
        self.time += time_delta

        self.food_source.update(time_delta)
        self.agent_group.update(time_delta)
        self.food_group.update(time_delta)

        if self.time >= self.fixed_delta_time:
            self.fixed_update(self.fixed_delta_time)
            self.time -= self.fixed_delta_time

    def fixed_update(self, fixed_delta):
        self.all_entities.fixed_update(fixed_delta)
        self.physics.fixed_update(fixed_delta)

    def draw(self, surface):
        surface.fill(self.background_color)

        self.rect = surface.get_rect()
        self.size = self.rect.size
        self.render_list.draw(surface)

        # self.physics.debug_draw(surface)
        # self.quadtree.debug_draw(surface)
        # self.agent.debug_draw(surface)

    def on_food_source_lifetime_exceeded(self, food_source):
        x = spikeyboi.spikey.random.uniform(0, 1) * self.size[0]
        y = spikeyboi.spikey.random.uniform(0, 1) * self.size[1]

        food_source.pos = (x,y)

    def on_save(self):
        with open('sim.pickle', 'wb') as fp:
            pickle.dump(self.rng_seed, fp)
            pickle.dump(self.agent.brain, fp)

    def on_load(self):
        with open('sim.pickle', 'rb') as fp:
            rng_seed = pickle.load(fp)
            spikeyboi.spikey.random = np.random.default_rng(rng_seed)

            self.agent.brain = pickle.load(fp)

if __name__ == '__main__':
    sim = Simulation()
