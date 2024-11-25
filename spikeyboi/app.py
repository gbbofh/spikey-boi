import spikeyboi
import spikeyboi.ui.menubar
import spikeyboi.ui.viewport
import spikeyboi.ui.debug_window
import spikeyboi.ui.delta_debug
import spikeyboi.ui.synapse_debug
import spikeyboi.ui.reward_debug
import spikeyboi.ui.eligibility_debug
import spikeyboi.ui.dendrogram_debug
import spikeyboi.ui.spike_debug
import spikeyboi.ui.fps_debug
import spikeyboi.ui.file_dialog


import pickle


import pygame as pg
import pygame_gui as gui


class App():
    def __init__(self, title='spikeyboi', size=(800,600)):
        spikeyboi.app_instance = self

        self.on_save_event = []
        self.on_load_event = []
        self.on_load_completed_event = []
        self.on_agent_selected_event = []

        self.run = True
        self.time_accum = 0.0
        self.fixed_delta_time = 0.04
        self.size = size

        pg.init()

        self.display = pg.display.set_mode(size)
        pg.display.set_caption(title)

        theme = gui.PackageResource('data.themes', 'default.json')

        self.manager = gui.UIManager(size, theme)

        menubar_data = {
            'File': ['New Brain', 'Save Brain', 'Load Brain'],
            'View': ['Synapses', 'Deltas', 'Rewards', 'Eligibility', 'Spikes', 'Toggle FPS', 'Toggle Blur'],
            'Analyze': ['Dendrogram'],
            'Debug': ['Physics', 'Quadtree']
        }

        self.menubar = spikeyboi.ui.menubar.UIMenuBar(pg.Rect((0,0),(size[0],50)), self.manager, menubar_data)

        self.menubar.bind_action('New Brain', self.new_brain)
        self.menubar.bind_action('Save Brain', self.show_save_dialog)
        self.menubar.bind_action('Load Brain', self.show_load_dialog)

        self.menubar.bind_action('Synapses', lambda: self.toggle_window(self.sd_view))
        self.menubar.bind_action('Deltas', lambda: self.toggle_window(self.dd_view))
        self.menubar.bind_action('Rewards', lambda: self.toggle_window(self.rd_view))
        self.menubar.bind_action('Eligibility', lambda: self.toggle_window(self.ed_view))
        self.menubar.bind_action('Spikes', lambda: self.toggle_window(self.spike_view))
        self.menubar.bind_action('Toggle FPS', lambda: self.toggle_window(self.fps_debug))
        self.menubar.bind_action('Toggle Blur', self.toggle_kernels)

        self.menubar.bind_action('Dendrogram', lambda: self.toggle_window(self.dendro_view))

        self.menubar.bind_action('Physics', self.toggle_physics_debug)
        self.menubar.bind_action('Quadtree', self.toggle_quadtree_debug)

        self.viewport = spikeyboi.ui.viewport.UIViewport(pg.Rect((0,0),(size[0], size[1] - 30)), self.manager, anchors={'top_target': self.menubar})

        self.dd_view = spikeyboi.ui.delta_debug.UIDeltaDebugger('Synaptic Deltas', (100,100,200,200), self.manager)
        self.sd_view = spikeyboi.ui.synapse_debug.UISynapseDebugger('Synaptic Weights', (100,100,200,200), self.manager)
        self.rd_view = spikeyboi.ui.reward_debug.UIRewardDebugger('Synaptic Rewards', (100,100,200,200), self.manager)
        self.ed_view = spikeyboi.ui.eligibility_debug.UIEligibilityDebugger('Reward Eligibility', (100,100,200,200), self.manager)
        self.spike_view = spikeyboi.ui.spike_debug.UISpikeDebugger('Spikes', (100,100,200,200), self.manager)

        self.dendro_view = spikeyboi.ui.dendrogram_debug.UIDendrogramDebugger('Dendrogram', (100,100,300,200), self.manager)

        self.fps_debug = spikeyboi.ui.fps_debug.UIFPSDebugger((-100,5), self.manager)

        self.toolbar_play = self.menubar.add_toolbar_button('#run', self.toolbar_play_pressed)
        self.toolbar_pause = self.menubar.add_toolbar_button('#pause', self.toolbar_pause_pressed)

        self.toolbar_play.disable()

        self.buffer = pg.Surface(self.display.size, pg.SRCALPHA)

        self.pause = False

        self.load_dialog = None
        self.save_dialog = None

    def toolbar_play_pressed(self):
        self.toolbar_play.disable()
        self.toolbar_pause.enable()
        self.pause = False

    def toolbar_pause_pressed(self):
        self.toolbar_play.enable()
        self.toolbar_pause.disable()
        self.pause = True

    def toggle_physics_debug(self):
        spikeyboi.spikey.sim_instance.debug_physics = not spikeyboi.spikey.sim_instance.debug_physics

    def toggle_quadtree_debug(self):
        spikeyboi.spikey.sim_instance.debug_quadtree = not spikeyboi.spikey.sim_instance.debug_quadtree

    def process_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.run = False

            if e.type == pg.WINDOWSIZECHANGED:
                self.buffer = pg.Surface((e.x, e.y))
                self.manager.set_window_resolution((e.x, e.y))

            if e.type == spikeyboi.ui.file_dialog.UI_LOAD_FILE_EVENT:
                res = e.file_path

                print(f'Loading: {res}')
                self.on_load(e.file_path)

            if e.type == spikeyboi.ui.file_dialog.UI_SAVE_FILE_EVENT:
                res = e.file_path
                print(f'Saving: {res}')

                self.on_save(e.file_path)

            if e.type == spikeyboi.ui.viewport.UI_AGENT_SELECTED:
                spikeyboi.spikey.sim_instance.agent = e.agent
                self.on_agent_selected(e.agent)

            if e.type == gui.UI_WINDOW_CLOSE:
                if e.ui_element == self.load_dialog:
                    self.load_dialog = None
                elif e.ui_element == self.save_dialog:
                    self.save_dialog = None

            self.manager.process_events(e)

    def update(self, delta_time):
        # if not self.pause:
        #     spikeyboi.spikey.sim_instance.update(delta_time)
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

    def show_load_dialog(self):
        # default_path = 'sim.pickle'
        # self.load_dialog = spikeyboi.ui.file_dialog.UIFileDialog(self.viewport.get_abs_rect(), self.manager, method='load')
        x,y = 10,10
        w,h = self.size
        self.load_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h), self.manager, method='load')

    def show_save_dialog(self):
        # default_path = 'sim.pickle'
        # self.save_dialog = spikeyboi.ui.file_dialog.UIFileDialog(self.viewport.get_abs_rect(), self.manager, method='save')
        x,y = 10,10
        w,h = self.size
        self.save_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h), self.manager, method='save')

    def new_brain(self):
        spikeyboi.spikey.sim_instance.agent.brain.reset()

    def on_load(self, path):
        for e in self.on_load_event:
            e(path)

        for e in self.on_load_completed_event:
            e()

    def on_save(self, path):
        for e in self.on_save_event:
            e(path)

    def on_agent_selected(self, agent):
        for e in self.on_agent_selected_event:
            e(agent)

    def fixed_update(self, fixed_delta):
        if not self.pause:
            spikeyboi.spikey.sim_instance.fixed_update(fixed_delta)

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

            while self.time_accum >= self.fixed_delta_time:
                self.fixed_update(self.fixed_delta_time)
                self.time_accum -= self.fixed_delta_time

            self.draw()

