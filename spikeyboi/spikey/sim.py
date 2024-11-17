import numpy as np
import pygame as pg

import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.food
import spikeyboi.spikey.food_source
import spikeyboi.spikey.physics
import spikeyboi.spikey.quadtree
import spikeyboi.spikey.wall


class Simulation():

    def __init__(self):
        spikeyboi.spikey.sim_instance = self
        self.render_list = pg.sprite.RenderUpdates()
        self.agent_group = pg.sprite.Group()
        self.food_group = pg.sprite.Group()
        self.physics_group = pg.sprite.Group()
        self.all_entities = pg.sprite.Group()

        agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.physics_group, self.all_entities, self.render_list)
        agent.x, agent.y = 100, 100
        # agent.rect.x, agent.rect.y = 100, 100
        # agent.on_agent_moved_event.append(self.on_object_moved)

        self.food_source = spikeyboi.spikey.food_source.FoodSource((200,200), 50, 25, 5.0, self.food_group, self.all_entities, self.physics_group, self.render_list)
        self.food_source.on_lifetime_exceeded_event.append(self.on_food_source_lifetime_exceeded)
        # self.food_source.on_object_moved_event.append(self.on_object_moved)

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
        self.rect = pg.Rect(0,0,0,0)
        self.size = (0,0)

    def update(self, time_delta):
        self.time += time_delta

        self.food_source.update(time_delta)
        self.agent_group.update(time_delta)
        self.food_group.update(time_delta)

        if self.time >= self.fixed_delta_time:
            self.fixed_update(self.fixed_delta_time)
            self.time -= self.fixed_delta_time

    def fixed_update(self, fixed_delta):
        # self.agent_group.fixed_update(fixed_delta)
        # self.food_group.fixed_update(fixed_delta)
        self.all_entities.fixed_update(fixed_delta)
        self.physics.fixed_update(fixed_delta)

    def draw(self, surface):
        surface.fill(self.background_color)

        self.rect = surface.get_rect()
        self.size = self.rect.size
        self.render_list.draw(surface)
        self.quadtree.debug_draw(surface)

    def on_food_source_lifetime_exceeded(self, food_source):
        x = spikeyboi.spikey.random.uniform(0, 1) * self.size[0]
        y = spikeyboi.spikey.random.uniform(0, 1) * self.size[1]

        food_source.pos = (x,y)

    # def on_object_moved(self, object):
    #     self.quadtree.rebuild(list(self.all_entities), (0,0,800,600), 3)

if __name__ == '__main__':
    sim = Simulation()
