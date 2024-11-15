import spikeyboi.ui.viewport


import pygame as pg
import pygame_gui as gui


class App():
    def __init__(self, title='spikeyboi', size=(800,600)):
        pg.init()

        self.display = pg.display.set_mode(size)
        pg.display.set_caption(title)

        self.run = True
        self.time_accum = 0.0
        self.fixed_delta_time = 0.03

        self.manager = gui.UIManager(size)

        self.viewport = spikeyboi.ui.viewport.UIViewport(pg.Rect((0,0),size), self.manager)

        self.buffer = pg.Surface(self.display.size, pg.SRCALPHA)

    def process_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.run = False

    def update(self, delta_time):
        self.manager.update(delta_time)

    def fixed_update(self, fixed_delta):
        # self.manager.fixed_update(fixed_delta)
        pass

    def draw(self):
        self.buffer.fill((50,100,200))

        # Rendering code here
        self.manager.draw_ui(self.buffer)

        self.display.blit(self.buffer,(0,0))
        pg.display.update()

    def main_loop(self):
        cur_time = pg.time.get_ticks()
        prev_time = pg.time.get_ticks()

        while self.run:
            prev_time = cur_time

            self.process_events()

            cur_time = pg.time.get_ticks()
            delta_time = (cur_time - prev_time) / 1000.0

            self.time_accum += delta_time

            self.update(delta_time)

            if self.time_accum >= self.fixed_delta_time:
                self.fixed_update(self.fixed_delta_time)
                self.time_accum -= self.fixed_delta_time

            self.draw()

