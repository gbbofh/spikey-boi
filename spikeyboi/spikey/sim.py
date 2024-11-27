import numpy as np
import pygame as pg


import time
import pickle


import spikeyboi
import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.food
# import spikeyboi.spikey.food_source
import spikeyboi.spikey.food_spawner
import spikeyboi.spikey.physics
import spikeyboi.spikey.quadtree
import spikeyboi.spikey.wall


class Simulation():

    def __init__(self, size=(800,600), num_agents=5):
        spikeyboi.spikey.sim_instance = self
        self.app = spikeyboi.app_instance

        self.time = 0.0
        self.fixed_delta_time = 0.03
        self.background_color = np.zeros(4)
        self.rect = pg.Rect((0,0),size)
        self.size = size

        self.rng_seed = int(time.time())
        spikeyboi.spikey.random = np.random.default_rng(self.rng_seed)

        w,h = size
        center = (w // 2,h // 2)

        self.render_list = pg.sprite.RenderUpdates()
        self.agent_group = pg.sprite.Group()
        self.food_group = pg.sprite.Group()
        self.physics_group = pg.sprite.Group()
        self.all_entities = pg.sprite.Group()

        for i in range(num_agents):
            a = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
            a.id = i
            a.x = spikeyboi.spikey.random.integers(70, w - 70)
            a.y = spikeyboi.spikey.random.integers(70, h - 70)
            a.rect.topleft = a.x, a.y
            a.angle = spikeyboi.spikey.random.uniform(0, 2 * np.pi)
            self.agent = a

        # agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
        # agent.x, agent.y = (100,200)
        # agent.rect.x, agent.rect.y = agent.x, agent.y
        # self.agent = agent

        # agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
        # agent.x, agent.y = 200, 200
        # agent.rect.x, agent.rect.y = agent.x, agent.y
        # self.agent2 = agent

        # food_source_pos = center

        # self.food_source = spikeyboi.spikey.food_source.FoodSource(food_source_pos, 50, 50, -1.0, self.food_group, self.all_entities, self.physics_group, self.render_list)
        # self.food_source.on_lifetime_exceeded_event.append(self.on_food_source_lifetime_exceeded)

        spawn_rect = pg.Rect((70, 70, w - 70, h - 70))
        self.food_spawner = spikeyboi.spikey.food_spawner.FoodSpawner(spawn_rect, spawn_rate=3)
        self.all_entities.add(self.food_spawner)

        wall_thickness = 50

        w,h = size

        north = spikeyboi.spikey.wall.Wall((5,5,w - 10,wall_thickness))
        east = spikeyboi.spikey.wall.Wall((w - wall_thickness - 5,5,wall_thickness,h - 25))
        south = spikeyboi.spikey.wall.Wall((5,h - wall_thickness - 15,w - 10,wall_thickness - 5))
        west = spikeyboi.spikey.wall.Wall((5,5,wall_thickness,h - 25))
        self.walls = [north, east, south, west]

        self.render_list.add(self.walls)
        self.all_entities.add(self.walls)
        self.physics_group.add(self.walls)

        self.quadtree = spikeyboi.spikey.quadtree.QuadTree(list(self.physics_group), self.rect, 3)
        self.physics = spikeyboi.spikey.physics.Physics(self.physics_group)

        self.app.on_save_brain_event.append(self.on_save_brain)
        self.app.on_load_brain_event.append(self.on_load_brain)

        self.debug_physics = False
        self.debug_quadtree = False
        self.debug_agents = False

    def update(self, time_delta):
        # self.time += time_delta

        # self.food_source.update(time_delta)
        # self.agent_group.update(time_delta)
        # self.food_group.update(time_delta)

        # if self.time >= self.fixed_delta_time:
        #     self.fixed_update(self.fixed_delta_time)
        #     self.time -= self.fixed_delta_time
        pass

    def fixed_update(self, fixed_delta):
        # self.food_source.update(fixed_delta)

        self.all_entities.update(fixed_delta)
        self.all_entities.fixed_update(fixed_delta)

        self.physics.fixed_update(fixed_delta)

    def draw(self, surface):
        surface.fill(self.background_color)

        self.rect = surface.get_rect()
        self.size = self.rect.size

        pos = self.agent.rect.center
        pg.draw.circle(surface, (200,50,100,100), pos, 20)
 
        self.render_list.draw(surface)

        if self.debug_physics:
            self.physics.debug_draw(surface)

        if self.debug_quadtree:
            self.quadtree.debug_draw(surface)

        if self.debug_agents:
            for agent in self.agent_group:
                agent.debug_draw(surface)

    def on_food_source_lifetime_exceeded(self, food_source):
        x = spikeyboi.spikey.random.uniform(0, 1) * self.size[0]
        y = spikeyboi.spikey.random.uniform(0, 1) * self.size[1]

        food_source.pos = (x,y)

    def on_save_brain(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.rng_seed, fp)
            pickle.dump(self.agent.brain, fp)

    def on_load_brain(self, path):
        with open(path, 'rb') as fp:
            rng_seed = pickle.load(fp)
            spikeyboi.spikey.random = np.random.default_rng(rng_seed)

            self.agent.brain = pickle.load(fp)

    def select_agent(self, agent):
        self.agent = agent


if __name__ == '__main__':
    sim = Simulation()
