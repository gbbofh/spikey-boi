import numpy as np
import pygame as pg

import spikeyboi.spikey
import spikeyboi.spikey.food
import spikeyboi.spikey.food_source
import spikeyboi.spikey.agent


class Simulation():

    def __init__(self):
        self.render_list = pg.sprite.RenderUpdates()
        self.agent_group = pg.sprite.Group()
        self.food_group = pg.sprite.Group()

        agent = spikeyboi.spikey.agent.Agent(self.agent_group, self.render_list)
        agent.x, agent.y = 100, 100
        self.food_source = spikeyboi.spikey.food_source.FoodSource((200,200), 50, 50, 5.0, self.food_group, self.render_list)
        self.food_source.on_lifetime_exceeded_event.append(self.on_food_source_lifetime_exceeded)

        self.time = 0.0
        self.fixed_delta_time = 0.03
        self.background_color = np.zeros(3)
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
        self.agent_group.fixed_update(fixed_delta)
        self.food_group.fixed_update(fixed_delta)

    def draw(self, surface):
        surface.fill(self.background_color)

        self.size = surface.get_rect().size
        self.render_list.draw(surface)

    def on_food_source_lifetime_exceeded(self, food_source):
        x = spikeyboi.spikey.random.uniform(0, 1) * self.size[0]
        y = spikeyboi.spikey.random.uniform(0, 1) * self.size[1]

        food_source.pos = (x,y)

if __name__ == '__main__':
    sim = Simulation()
