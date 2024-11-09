import pygame
import numpy as np

class Time():

    cur_time = 0
    prev_time = 0
    delta_time = 0

    def update():
        Time.cur_time = pygame.time.get_ticks()
        Time.delta_time = Time.cur_time - Time.prev_time
        Time.prev_time = Time.cur_time

random = np.random.default_rng()
