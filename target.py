import numpy as np

import pygame


import util


class Target():

    def __init__(self, x=0, y=0, radius=10):
        self.screen = pygame.display.get_surface()
        # rand = np.random.randint
        rand = util.random.integers
        self.x = x or rand(radius, self.screen.get_width() - radius)
        self.y = y or rand(radius, self.screen.get_height() - radius)
        self.pos = (self.x,self.y)

        self.radius = radius
        self.color = (0, 128, 64)

        hr = radius // 2
        self.bounds = pygame.Rect(self.x - hr, self.y - hr, radius, radius)

    def draw(self):
        center = (self.x, self.y)
        r = pygame.draw.circle(self.screen, self.color, center, self.radius)
        self.bounds = r

    def on_collision(self):
        radius = self.radius
        hr = radius // 2

        rand = np.random.randint

        x = rand(radius, self.screen.get_width() - radius)
        y = rand(radius, self.screen.get_height() - radius)

        # self.x = np.random.randint(radius, self.screen.get_width() - radius)
        # self.y = np.random.randint(radius, self.screen.get_height()- radius)
        self.set_pos((x,y))

        self.bounds = pygame.Rect(self.x - hr, self.y - hr, radius, radius)

    # def set_pos(self, x, y):
    #     self.x = x
    #     self.y = y

    def set_pos(self, pos):
        self.x, self.y = pos
        self.pos = pos
