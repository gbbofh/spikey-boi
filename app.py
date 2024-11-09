import numpy as np


import pygame


import time
import pickle


import util
import agent
import target
import network
import debugger


class App():

    params = {
        'dt': 0.1,
        'I_ext_std': 0.5,
        'I_ext_mean': 1.35,
        'I_ext_adaptive_scale': 0.05,
        # 'P_syn': 0.3,
        'd_max': 6,
        # 'A_plus': 0.008,
        # 'A_minus': 0.008 * 1.1,
        # 'A_plus': 2.2,
        # 'A_minus': 2.2 * 1.1,
        # 'A_plus': 0.01,
        # 'A_minus': 0.01*1.1,
        'A_plus': 0.01,
        'A_minus': 0.01 * 1.1,
        'r_min': -1,
        'r_max': 1,
        'tau_r': 100.0,
        # 'tau_e': 400.0,
        'P_syn_gen': 0.00005,
        'F_t': 15,
    }

    agent_params = {

        'I_scale': 1.5,
    }

    # MS_PER_UPDATE = 10
    MS_PER_UPDATE = 4

    def __init__(self, num_neurons=50):

        pygame.init()

        self.seed = int(time.time())
        # np.random.seed(self.seed)
        util.random = np.random.default_rng(self.seed)

        self.display = pygame.display.set_mode((800,600))
        self.buffer = pygame.Surface((800,600))
        self.size = self.display.get_size()

        self.num_neurons = num_neurons
        self.net = network.Network(self.num_neurons, params=self.params)
        self.agent = agent.Agent(self.net)

        self.target = target.Target()

        self.agent.set_target(self.target)

        self.debug = debugger.DebugManager(self)
        # self.debug.ui.append(debugger.FPSCounter())
        # self.debug.ui.append(debugger.SynapseDebugger(net=self.net))
        # self.debug.ui.append(debugger.STDPDebugger(net=self.net))
        # self.debug.ui.append(debugger.RewardDebugger(net=self.net))
        # self.debug.ui.append(debugger.VoltageDebugger(self.agent))
        # # self.debug.ui.append(debugger.AgentDebugger(self.agent))
        # self.debug.ui.append(debugger.TargetDebugger(self.agent))
        # self.debug.ui.append(debugger.AgentHeatmapDebugger(self.agent))
        # # self.debug.ui.append(debugger.AgentStatsDebugger(self.agent))
        # # self.debug.ui.append(debugger.FiringRateHeatmapDebugger(self.net))
        # # self.debug.ui.append(debugger.SpikeHistogramDebugger(self.net))
        # # self.debug.ui[4].enabled = False
        # # self.debug.ui.append(debugger.SpikeHistDebugger(self.agent))

        self.run = True

        # self.prev_time = pygame.time.get_ticks()
        # self.time = pygame.time.get_ticks()
        # self.delta_time = 0
        self.time_accum = 0
        self.font = pygame.freetype.SysFont('Arial', 16)

        self.run_time_s = 0
        self.run_time_m = 0
        self.run_time_h = 0

        self.sim_time_s = 0
        self.sim_time_m = 0
        self.sim_time_h = 0

    def process_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.run = False
            if e.type == pygame.KEYDOWN:
                self.on_key_down(e.key)
            if e.type == pygame.MOUSEBUTTONDOWN:
                self.on_mouse_button_down(e.button)

            # if e.type == pygame.MOUSEBUTTONDOWN:
            #     pos = pygame.mouse.get_pos()
            #     self.target.set_pos(*pos)

    def on_mouse_button_down(self, button):
        lut = {
                1: lambda: self.target.set_pos(pygame.mouse.get_pos()),
                3: lambda: self.agent.set_pos(pygame.mouse.get_pos())
        }

        cb = lut.get(button)
        if cb is not None:
            cb()

    def on_key_down(self, key):
        lut = {
            pygame.K_BACKQUOTE: lambda: self.debug.toggle(),
            pygame.K_1: lambda: self.debug.toggle(1),
            pygame.K_2: lambda: self.debug.toggle(2),
            pygame.K_3: lambda: self.debug.toggle(3),
            pygame.K_4: lambda: self.debug.toggle(4),
            pygame.K_5: lambda: self.debug.toggle(5),
            pygame.K_6: lambda: self.debug.toggle(6),
            pygame.K_7: lambda: self.debug.toggle(7),
            pygame.K_8: lambda: self.debug.toggle(8),
            pygame.K_9: lambda: self.debug.toggle(9),
            pygame.K_0: lambda: self.debug.toggle_kernels(),
            pygame.K_SLASH: lambda: self.debug._agent_trace.toggle(),
            pygame.K_n: lambda: self.net.toggle_noise(),
            pygame.K_i: lambda: self.agent.toggle_inputs(),
            pygame.K_q: lambda: self.net.toggle_adaptive_noise(),
            pygame.K_g: lambda: self.debug._graph.toggle(),
            pygame.K_SPACE: lambda: self.save_state(),
            pygame.K_BACKSPACE: lambda: self.load_state(),
        }

        cb = lut.get(key)
        if cb is not None:
            cb()

    def update(self):
        self.agent.update()
        self.debug.update()
        self.sim_time_s += self.agent.net.params.dt / 1000
        self.sim_time_m += self.sim_time_s >= 60
        self.sim_time_h += self.sim_time_m >= 60

        self.sim_time_s = np.mod(self.sim_time_s, 60)
        self.sim_time_m = np.mod(self.sim_time_m, 60)
        self.sim_time_h = np.mod(self.sim_time_h, 100)

    def draw(self):
        self.display.fill((0,0,0))

        self.debug.draw()
        self.agent.draw()
        self.target.draw()

        x = self.size[0] - 50
        y = self.size[1] - 70

        pos = (x, y)
        color = (255,255,255,255)
        fmt = f'{self.agent.collision_count}'
        self.font.render_to(self.display, pos, fmt, color)

        time_units = 's'
        time_var = self.run_time_s

        if self.run_time_m > 0:
            time_units = 'm'
            time_var = self.run_time_m + self.run_time_s / 60
        if self.run_time_h > 0:
            time_units = 'h'
            time_var = self.run_time_h + self.run_time_m / 60

        pos = (x - 20, y + 20)
        fmt = f'{time_var:0.01f} {time_units}'
        self.font.render_to(self.display, pos, fmt, color)

        time_units = 's'
        time_var = self.sim_time_s

        if self.sim_time_m > 0:
            time_units = 'm'
            time_var = self.sim_time_m + self.sim_time_s / 60
        if self.sim_time_h > 0:
            time_units = 'h'
            time_var = self.sim_time_h + self.run_time_m / 60

        pos = (x - 20, y + 40)
        fmt = f'{time_var:0.01f} {time_units}'
        self.font.render_to(self.display, pos, fmt, color)

        pygame.display.flip()

    def save_state(self, name='state'):
        with open(name, 'wb') as fp:
            # pickle.dump(self.seed, fp)
            # pickle.dump(self.run_time_h, fp)
            # pickle.dump(self.run_time_m, fp)
            # pickle.dump(self.run_time_s, fp)
            # pickle.dump(self.sim_time_h, fp)
            # pickle.dump(self.sim_time_m, fp)
            # pickle.dump(self.sim_time_s, fp)
            # pickle.dump(self.net, fp)
            # pickle.dump(self.agent, fp)
            # pickle.dump(self.target, fp)
            # pickle.dump(self.debug, fp)
            pickle.dump(self.seed, fp)
            pickle.dump(self.net, fp)
            pickle.dump((self.agent.pos, self.agent.angle), fp)
            pickle.dump(self.agent.collision_count, fp)
            pickle.dump((self.run_time_h, self.run_time_m, self.run_time_s), fp)
            pickle.dump((self.sim_time_h, self.sim_time_m, self.sim_time_s), fp)
            pickle.dump(self.target.pos, fp)

    def load_state(self, name='state'):
        with open(name, 'rb') as fp:
            # self.seed = pickle.load(fp)
            # self.run_time_h = pickle.load(fp)
            # self.run_time_m = pickle.load(fp)
            # self.run_time_s = pickle.load(fp)
            # self.sim_time_h = pickle.load(fp)
            # self.sim_time_m = pickle.load(fp)
            # self.sim_time_s = pickle.load(fp)
            # self.net = pickle.load(fp)
            # self.agent = pickle.load(fp)
            # self.target = pickle.load(fp)
            # self.debug = pickle.load(fp)
            self.seed = pickle.load(fp)
            self.net = pickle.load(fp)
            pos, angle = pickle.load(fp)
            cc = pickle.load(fp)
            rh,rm,rs = pickle.load(fp)
            sh,sm,ss = pickle.load(fp)
            tpos = pickle.load(fp)

            self.agent = agent.Agent(self.net)
            self.agent.collision_count = cc
            self.agent.x, self.agent.y = pos
            self.agent.angle = angle

            self.target = target.Target()
            self.target.set_pos(tpos)

            self.agent.set_target(self.target)

            self.run_time_h = rh
            self.run_time_m = rm
            self.run_time_s = rs

            self.sim_time_h = sh
            self.sim_time_m = sm
            self.sim_time_s = ss

            debug_enable = self.debug.enabled
            states = [e.enabled for e in self.debug.ui]

            self.debug = debugger.DebugManager(self)
            util.random = np.random.default_rng(self.seed)

            for i, s in enumerate(states):
                self.debug.ui[i].enabled = s

            self.debug.enabled = debug_enable

    def main(self):

        util.Time.prev_time = pygame.time.get_ticks()

        autosave_time = 0

        while self.run:

            self.process_events()

            util.Time.update()

            self.run_time_s += util.Time.delta_time / 1000
            self.run_time_m += self.run_time_s >= 60
            self.run_time_h += self.run_time_m >= 60

            self.run_time_s = np.mod(self.run_time_s, 60)
            self.run_time_m = np.mod(self.run_time_m, 60)
            self.run_time_h = np.mod(self.run_time_h, 100)

            self.delta_time = util.Time.delta_time
            self.time_accum += self.delta_time
            autosave_time += self.delta_time / 1000

            if autosave_time >= 60:
                self.save_state('autosave')

            while self.time_accum >= self.MS_PER_UPDATE:
                self.update()
                self.time_accum  -= self.MS_PER_UPDATE

            self.draw()

