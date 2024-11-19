import spikeyboi
import spikeyboi.ui.menubar
import spikeyboi.ui.viewport
import spikeyboi.ui.debug_window
import spikeyboi.ui.delta_debug
import spikeyboi.ui.synapse_debug
import spikeyboi.ui.reward_debug
import spikeyboi.ui.eligibility_debug
import spikeyboi.ui.fps_debug


import pickle


import pygame as pg
import pygame_gui as gui


class App():
    def __init__(self, title='spikeyboi', size=(800,600)):
        spikeyboi.app_instance = self

        self.on_save_event = []
        self.on_load_event = []
        self.on_load_completed_event = []

        self.run = True
        self.time_accum = 0.0
        self.fixed_delta_time = 0.03
        self.size = size

        pg.init()

        self.display = pg.display.set_mode(size)
        pg.display.set_caption(title)

        self.manager = gui.UIManager(size)

        menubar_data = {
            'File': ['Save Brain', 'Load Brain'],
            'View': ['Synapses', 'Deltas', 'Rewards', 'Eligibility', 'Toggle FPS', 'Toggle Blur'],
        }

        self.menubar = spikeyboi.ui.menubar.UIMenuBar(pg.Rect((0,0),(size[0],30)), self.manager, menubar_data)
        # self.menubar.bind_action('Save Brain', self.save_brain)
        # self.menubar.bind_action('Load Brain', self.load_brain)
        self.menubar.bind_action('Save Brain', self.on_save)
        self.menubar.bind_action('Load Brain', self.on_load)
        self.menubar.bind_action('Synapses', lambda: self.toggle_window(self.sd_view))
        self.menubar.bind_action('Deltas', lambda: self.toggle_window(self.dd_view))
        self.menubar.bind_action('Rewards', lambda: self.toggle_window(self.rd_view))
        self.menubar.bind_action('Eligibility', lambda: self.toggle_window(self.ed_view))
        self.menubar.bind_action('FPS', lambda: self.toggle_window(self.fps_debug))
        self.menubar.bind_action('Toggle Blur', self.toggle_kernels)

        self.viewport = spikeyboi.ui.viewport.UIViewport(pg.Rect((0,0),(size[0], size[1] - 30)), self.manager, anchors={'top_target': self.menubar})
        self.dd_view = spikeyboi.ui.delta_debug.UIDeltaDebugger('Synaptic Deltas', (100,100,300,300), self.manager)
        self.sd_view = spikeyboi.ui.synapse_debug.UISynapseDebugger('Synaptic Weights', (100,100,300,300), self.manager)
        self.rd_view = spikeyboi.ui.reward_debug.UIRewardDebugger('Synaptic Rewards', (100,100,300,300), self.manager)
        self.ed_view = spikeyboi.ui.eligibility_debug.UIEligibilityDebugger('Reward Eligibility', (100,100,300,300), self.manager)
        self.fps_debug = spikeyboi.ui.fps_debug.UIFPSDebugger((-100,5), self.manager)

        self.buffer = pg.Surface(self.display.size, pg.SRCALPHA)

    def process_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.run = False
            self.manager.process_events(e)

    def update(self, delta_time):
        self.manager.update(delta_time)

    def toggle_window(self, window):
        if window.visible:
            window.hide()
        else:
            window.show()

    def toggle_kernels(self):
        self.sd_view.kernel_enabled = not self.sd_view.kernel_enabled
        self.rd_view.kernel_enabled = not self.rd_view.kernel_enabled
        self.ed_view.kernel_enabled = not self.ed_view.kernel_enabled
        self.dd_view.kernel_enabled = not self.dd_view.kernel_enabled

    # TODO: Add event for on_save with listener in brain
    # TODO: Save rng seed
    # def save_brain(self):
    #     with open('brain.pickle', 'wb') as fp:
    #         pickle.dump(self.agent.brain, fp)

    # # TODO: Add event for on_load, with listeners in UI and brain
    # # TODO: Load rng seed
    # def load_brain(self):
    #     with open('brain.pickle', 'rb') as fp:
    #         brain = pickle.load(fp)
    #         self.agent.brain = brain
    #         self.sd_view.net = brain.net
    #         self.dd_view.net = brain.net
    #         self.rd_view.net = brain.net
    #         self.ed_view.net = brain.net

    def on_load(self):
        for e in self.on_load_event:
            e()

        for e in self.on_load_completed_event:
            e()

    def on_save(self):
        for e in self.on_save_event:
            e()

    def fixed_update(self, fixed_delta):
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

