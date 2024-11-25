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
import spikeyboi.ui.settings


import pickle
import datetime


import pygame as pg
import pygame_gui as gui


class App():
    def __init__(self, title='spikeyboi', size=(800,600)):
        spikeyboi.app_instance = self

        self.on_save_brain_event = []
        self.on_load_brain_event = []
        self.on_load_brain_completed_event = []
        self.on_agent_selected_event = []

        self.run = True
        self.time_accum = 0.0
        self.fixed_delta_time = 0.04
        self.size = size

        pg.init()

        self.display = pg.display.set_mode(size)
        pg.display.set_caption(title)

        theme = gui.PackageResource('res.themes', 'default.json')

        self.manager = gui.UIManager(size, theme)

        menubar_data = {
            'Agent': ['New Brain', 'Load Brain', 'Save Brain'],
            'Simulation': ['Reset Sim', 'Load Sim', 'Save Sim'],
            'View': ['Synapses', 'Deltas', 'Rewards', 'Eligibility', 'Spikes', 'Toggle Blur'],
            'Analyze': ['Dendrogram'],
            'Debug': ['Physics', 'Quadtree', 'Toggle FPS']
        }

        self.menubar = spikeyboi.ui.menubar.UIMenuBar(pg.Rect((0,0),(size[0],50)), self.manager, menubar_data)

        # Agent options
        self.menubar.bind_action('New Brain', self.new_brain)
        self.menubar.bind_action('Save Brain', self.show_save_brain_dialog)
        self.menubar.bind_action('Load Brain', self.show_load_brain_dialog)

        # Simulation options
        self.menubar.bind_action('Reset Sim', self.reset_sim)
        self.menubar.bind_action('Save Sim', self.show_save_sim_dialog)
        self.menubar.bind_action('Load Sim', self.show_load_sim_dialog)

        # View options
        self.menubar.bind_action('Synapses', lambda: self.toggle_window(self.sd_view))
        self.menubar.bind_action('Deltas', lambda: self.toggle_window(self.dd_view))
        self.menubar.bind_action('Rewards', lambda: self.toggle_window(self.rd_view))
        self.menubar.bind_action('Eligibility', lambda: self.toggle_window(self.ed_view))
        self.menubar.bind_action('Spikes', lambda: self.toggle_window(self.spike_view))
        self.menubar.bind_action('Toggle FPS', lambda: self.toggle_window(self.fps_debug))
        self.menubar.bind_action('Toggle Blur', self.toggle_kernels)

        # Analysis options
        self.menubar.bind_action('Dendrogram', lambda: self.toggle_window(self.dendro_view))

        # Debug options
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

        self.menubar.add_spacer(50)
        self.toolbar_settings = self.menubar.add_toolbar_button('#settings', self.toolbar_settings_pressed)

        self.toolbar_play.disable()

        self.buffer = pg.Surface(self.display.size, pg.SRCALPHA)

        self.pause = False

        self.load_dialog = None
        self.save_dialog = None
        self.settings_dialog = None

    def toolbar_play_pressed(self):
        self.toolbar_play.disable()
        self.toolbar_pause.enable()
        self.pause = False

    def toolbar_pause_pressed(self):
        self.toolbar_play.enable()
        self.toolbar_pause.disable()
        self.pause = True

    def toolbar_settings_pressed(self):
        x,y = 10,10
        w,h = self.size
        self.settings_dialog = spikeyboi.ui.settings.UISettingsWindow(pg.Rect(x,y,w,h), self.manager)

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

            if e.type == spikeyboi.ui.viewport.UI_AGENT_SELECTED:
                spikeyboi.spikey.sim_instance.agent = e.agent
                self.on_agent_selected(e.agent)

            if e.type == gui.UI_WINDOW_CLOSE:
                if e.ui_element == self.load_dialog:
                    self.load_dialog = None
                elif e.ui_element == self.save_dialog:
                    self.save_dialog = None
                elif e.ui_element == self.settings_dialog:
                    self.settings_dialog = None

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
        self.spike_view.kernel_enabled = not self.spike_view.kernel_enabled

    def show_load_brain_dialog(self):
        x,y = 10,10
        w,h = self.size
        path = f'data/saves/brain/'
        self.load_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h),
                                                                self.manager,
                                                                self.on_load_brain,
                                                                method='load',
                                                                initial_file_path=path)

    def show_save_brain_dialog(self):
        x,y = 10,10
        w,h = self.size
        agent = spikeyboi.spikey.sim_instance.agent
        path = f'data/saves/brain/agent_{agent.id}.pickle'
        self.save_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h),
                                                                self.manager,
                                                                self.on_save_brain,
                                                                method='save',
                                                                initial_file_path=path)

    def show_load_sim_dialog(self):
        x,y = 10,10
        w,h = self.size
        path = f'data/saves/sim/'
        self.load_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h),
                                                                self.manager,
                                                                self.on_load_sim,
                                                                method='load',
                                                                initial_file_path=path)

    def show_save_sim_dialog(self):
        x,y = 10,10
        w,h = self.size
        agent = spikeyboi.spikey.sim_instance.agent
        path = f'data/saves/sim/sim-{datetime.datetime.utcnow()}.pickle'
        self.save_dialog = spikeyboi.ui.file_dialog.UIFileDialog(pg.Rect(x,y,w,h),
                                                                self.manager,
                                                                self.on_save_sim,
                                                                method='save',
                                                                initial_file_path=path)

    def new_brain(self):
        spikeyboi.spikey.sim_instance.agent.brain.reset()

    def on_load_brain(self, path):
        for e in self.on_load_brain_event:
            e(path)

        for e in self.on_load_brain_completed_event:
            e()

    def on_save_brain(self, path):
        for e in self.on_save_brain_event:
            e(path)

    def reset_sim(self):
        size = self.size

        self.viewport.kill()
        del self.viewport

        self.viewport = spikeyboi.ui.viewport.UIViewport(pg.Rect((0,0),(size[0], size[1] - 30)), self.manager, anchors={'top_target': self.menubar})

    def on_load_sim(self, path):
        pass

    def on_save_sim(self, path):
        pass

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

