import numpy as np
import scipy.ndimage


import functools


import pygame
import pygame.gfxdraw
import pygame.freetype
import pygame.surfarray
import pygame.transform


import util
import agent
import target
import network


class DebugManager():
    def __init__(self, app):
        display = pygame.display.get_surface()
        self.display_size = display.get_size()

        self.ui = []
        self.buffer = pygame.Surface(self.display_size, pygame.SRCALPHA)
        self.enabled = True
        self.app = app

        # DebuggerGroups are currently buggy and need some changes to be used
        # debugGroup = DebuggerGroup(app=app)
        # debugGroup.add(SynapseDebugger(app=app, net=app.net))
        # debugGroup.add(STDPDebugger(app=app, net=app.net))
        # debugGroup.add(RewardDebugger(app=app, net=app.net))

        self.ui.append(FPSCounter(app=app))
        # self.ui.append(debugGroup)
        # self.ui.append(SynapseDebugger(app=app, net=app.net))
        # self.ui.append(STDPDebugger(app=app, net=app.net))
        # self.ui.append(RewardDebugger(app=app, net=app.net))
        # self.ui.append(VoltageDebugger(app=app, agent=app.agent))
        # rt = int(np.sqrt(n))
        # qn = (rt + 1) ** 2

        self._ptrace_scale = 10
        w,h = self.display_size
        w,h = w//self._ptrace_scale, h//self._ptrace_scale

        # Agent position trace
        ptrace = HeatmapDebugger(app=app, size=(w,h), ratio=self._ptrace_scale, alpha=0.1, name='position')

        self._agent_pos_record = np.zeros((w,h))

        ptrace.set_data_ref(self._agent_pos_record)
        ptrace.set_color_map('plasma')
        ptrace.transpose = False
        # ptrace.set_kernel(5)

        # k = 1/6.6 * np.array([
        #     [0.0, 0.2, 0.2, 0.2, 0.0],
        #     [0.2, 0.3, 0.5, 0.3, 0.2],
        #     [0.2, 0.5, 1.0, 0.5, 0.2],
        #     [0.2, 0.3, 0.5, 0.3, 0.2],
        #     [0.0, 0.2, 0.2, 0.2, 0.0],
        # ])

        k = 1/21 * np.array([
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ])

        ptrace.kernel = k

        # Ratio for neuron an synapse traces
        r = 7
        n = app.net.num_neurons

        # Per neuron traces
        # Original membrane voltage trace -- not integrated with synaptic traces
        # volt_trace = HeatmapDebugger(app=app, pos=(0, -10), anchor=Debugger.ANCHOR_BOTTOM, size=(n,1), ratio=r, alpha=1)
        volt_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.5, name='voltage')
        fr_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.5, name='firing rate')
        volt_corr_trace = HeatmapDebugger(app=app,size=(n,n), ratio=r, alpha=0.05, zero_empty=True, name='voltage corr')
        spike_trace = HeatmapDebugger(app=app,size=(n,n), ratio=r, alpha=0.95, name='spikes')
        elig_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.95, zero_empty=False, name='eligibility')

        # Is_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.95, name='I_syn')
        # Ie_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.95, name='I_ext')
        # Ii_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.5, name='I_inj')
        It_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.95, name='I_total')

        # Synaptic traces
        w_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.95, name='synapses', zero_empty=True)
        dw_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.8, name='weight delta')
        reward_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=1, zero_empty=False, name='rewards')

        cdw_trace = HeatmapDebugger(app=app, size=(n,n), ratio=r, alpha=0.001, name='abs. weight delta')

        volt_trace.set_data_ref(app.net.v_m[:,np.newaxis])
        volt_trace.set_color_map('zebra')
        volt_trace.set_diagonalize(True)
        volt_trace.set_kernel(3)

        fr_trace.set_data_ref(app.net.firing_rates[:,  np.newaxis])
        fr_trace.set_color_map('inferno')
        fr_trace.set_diagonalize(True)
        fr_trace.set_kernel(3)

        volt_corr_trace.set_data_ref((app.net.V_pre, app.net.V_post), np.outer)
        volt_corr_trace.set_color_map('magma')
        volt_corr_trace.set_kernel(3)

        # Is_trace.set_data_ref(app.net.I_syn)
        # Is_trace.set_color_map('spectral', False)
        # Is_trace.set_diagonalize(True)
        # Is_trace.set_kernel(3)

        # Ie_trace.set_data_ref(app.net.I_ext)
        # Ie_trace.set_color_map('spectral', False)
        # Ie_trace.set_diagonalize(True)
        # Ie_trace.set_kernel(3)

        # Ii_trace.set_data_ref(app.net.I_inj)
        # Ii_trace.set_color_map('spectral', False)
        # Ii_trace.set_diagonalize(True)
        # Ii_trace.set_kernel(3)

        try:
            It_trace.set_data_ref(app.net.I_total)
            It_trace.set_color_map('inferno', False)
            It_trace.set_diagonalize(True)
            It_trace.set_kernel(3)
        except:
            pass

        # Diagonal kernel for some neuron info
        k = np.array([
            [0.0, 0.0, 0.2],
            [0.0, 0.4, 0.4],
            [0.4, 1.0, 0.4],
            [0.4, 0.4, 0.0],
            [0.2, 0.0, 0.0],
        ])
        volt_trace.kernel = k
        fr_trace.kernel = k
        volt_corr_trace.kernel = k

        spike_trace.set_data_ref(app.net.spikes)
        spike_trace.set_color_map('zebra')
        spike_trace.set_diagonalize(True)

        k = np.array([
            [0.0, 0.1, 0.1, 0.1, 0.3],
            [0.1, 0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 1.0, 0.2, 0.1],
            [0.1, 0.6, 0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1, 0.1, 0.0],
        ])
        # k = np.array([
        #     [0.0, 0.1, 0.0],
        #     [0.2, 0.4, 0.2],
        #     [0.3, 0.8, 0.3],
        #     [0.4, 1.0, 0.4],
        #     [0.3, 0.8, 0.3],
        #     [0.2, 0.4, 0.2],
        #     [0.0, 0.1, 0.0],
        # ]).T
        spike_trace.kernel = k

        elig_trace.set_data_ref(app.net.E_syn)
        elig_trace.set_color_map('plasma')
        elig_trace.set_kernel(5)

        w_trace.set_data_ref((app.net.w, app.net.neuron_type[np.newaxis,:]))
        w_trace.set_color_map('rdgr')
        w_trace.set_kernel(2)

        dw_trace.set_data_ref(app.net.dw)
        dw_trace.set_color_map('ylgn')
        dw_trace.set_kernel(4)

        reward_trace.set_data_ref(app.net.reward)
        reward_trace.set_color_map('rdbu')
        reward_trace.set_kernel(4)

        cdw_trace.set_data_ref(np.abs(app.net.dw))
        cdw_trace.set_color_map('inferno')
        cdw_trace.set_kernel(4)

        self.ui.append(volt_trace)
        self.ui.append(fr_trace)
        self.ui.append(volt_corr_trace)
        self.ui.append(spike_trace)

        self.ui.append(w_trace)
        # self.ui.append(dw_trace)
        self.ui.append(reward_trace)
        self.ui.append(elig_trace)
        self.ui.append(cdw_trace)

        # self.ui.append(Is_trace)
        # self.ui.append(Ie_trace)
        # self.ui.append(Ii_trace)
        try:
            self.ui.append(It_trace)
        except:
            pass

        self.ui.append(ptrace)

        self._w_trace = w_trace
        self._cdw_trace = cdw_trace
        self._agent_trace = ptrace

        meta = DebuggerMetaInfo(app=app)
        self.ui.append(meta)

        self.toggle()

    def update(self):
        mp = pygame.mouse.get_pos()

        # This info needs updated every frame -- so a hack since I'm dumb
        self._w_trace.set_data_ref((self.app.net.w, self.app.net.neuron_type[:,np.newaxis]))
        self._cdw_trace.set_data_ref(np.abs(self.app.net.dw))

        # print(self.app.net.w[:,-2],self.app.net.w[-2,:])

        x,y = self.app.agent.pos
        x = int(x//self._ptrace_scale)
        y = int(y//self._ptrace_scale)

        w,h = self._agent_pos_record.shape

        # Decay position record -- ugly hack, but it works to keep colors visible
        # and updating over a good timeframe
        self._agent_pos_record[:] *= np.exp(-1/1000)
        # self._agent_pos_record[x,y] = self.app.agent.angle
        self._agent_pos_record[x,y] += 1

        for e in self.ui:
            if e.rect.collidepoint(mp) and pygame.mouse.get_focused():
                e.on_mouse_over((mp[0] // e.ratio, mp[1] // e.ratio))
            e.update()

    def draw(self):
        if not self.enabled:
            return

        self.buffer.fill((0,0,0,0))

        for i,e in enumerate(sorted(self.ui)):
            if e.enabled:
                e.draw()

                size = (e.size[0] * e.ratio, e.size[1] * e.ratio)
                tmp = pygame.transform.scale(e.buffer, size)
                # alpha = pygame.surfarray.pixels_alpha(tmp)
                # alpha = 3 * alpha // 4
                # del alpha

                tmp.premul_alpha()

                pos = e.pos
                if e.draw_over:
                    self.buffer.fill((0,0,0,0), pygame.Rect(pos, size))
                self.buffer.blit(tmp, pos,special_flags=pygame.BLEND_PREMULTIPLIED)
                # self.buffer.blit(tmp, pos,special_flags=pygame.BLEND_RGBA_MULT)

        pygame.display.get_surface().blit(self.buffer, (0,0))

    def toggle(self, id=None):
        if id is None:
            self.enabled = not self.enabled
        elif id < len(self.ui):
            self.ui[id].toggle()

    def toggle_kernels(self):
        for e in self.ui[5:]:
            e.toggle_kernel()


class Debugger():

    ANCHOR_TOP = 1 << 0
    ANCHOR_LEFT = 1 << 1
    ANCHOR_BOTTOM = 1 << 2
    ANCHOR_RIGHT = 1 << 3
    ANCHOR_CENTER = 1 << 4
    ANCHOR_NONE = 1 << 8

    def __init__(self, name=None, pos=None, size=None, ratio=1, anchor=ANCHOR_LEFT|ANCHOR_TOP, app=None):
        self.display = pygame.display.get_surface()
        self.display_size = self.display.get_size()
        self.name = name or self.__class__.__name__
        self.app = app

        self.anchor = anchor
        self.draw_order = 1000

        pos = pos or (0, 0)
        size = size or self.display_size

        if anchor & self.ANCHOR_RIGHT:
            pos = (self.display_size[0] + pos[0], pos[1])
        if anchor & self.ANCHOR_BOTTOM:
            pos = (pos[0], self.display_size[1] + pos[1])
        if anchor & self.ANCHOR_CENTER:
            pos = ((self.display_size[0] - size[0]) // 2, (self.display_size[1] - size[1]) // 2)

        w, h = size

        self.ratio = ratio
        # self.size = (w // ratio, h // ratio)
        self.size = size
        self.pos = pos

        self.buffer = pygame.Surface(self.size, pygame.SRCALPHA)
        self.enabled = True
        self.clear = True
        self.draw_over = False

        self.rect = pygame.Rect(self.pos, size)

        self.font_small = pygame.freetype.SysFont('Arial', 8)
        self.font_medium = pygame.freetype.SysFont('Arial', 12)
        self.font_large = pygame.freetype.SysFont('Arial', 16)

    def draw(self):
        if self.clear:
            self.buffer.fill((0,0,0,0))

    def update(self):
        pass

    def toggle_kernel(self):
        pass

    def toggle(self):
        self.enabled = not self.enabled

    def on_mouse_over(self, pos):
        pass

    def on_mouse_button_down(self, button):
        pass

    def on_key_pressed(self, key):
        pass

    def __lt__(self, other):
        return self.draw_order < other.draw_order


class DebuggerMetaInfo(Debugger):

    def __init__(self, *args, **kwargs):
        super().__init__(size=(160,130), pos=(0,-130), anchor=Debugger.ANCHOR_BOTTOM, *args, **kwargs)

    def draw(self):
        self.buffer.fill((255,255,255,120))

        for i,e in enumerate(self.app.debug.ui[1:]):
            enabled = 'enabled' if e.enabled else 'disabled'
            if e == self:
                continue

            pos = (5, 5 + 15*i)
            self.font_medium.render_to(self.buffer, pos, f'{e.name}: {enabled}')


class DebuggerGroup(Debugger):

    def __init__(self, ratio=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        size = self.display_size[0] // 4, self.display_size[1] // 4

        self.buffer = pygame.Surface(size, pygame.SRCALPHA)
        self.debuggers = []

    def add(self, debugger):
        self.debuggers.append(debugger)

    def draw(self):

        self.buffer.fill((0,0,0,0))

        for e in self.debuggers:
            e.draw()
            size = (e.size[0] * e.ratio, e.size[1] * e.ratio)
            tmp = pygame.transform.scale(e.buffer, size)
            pos = e.pos
            r = self.buffer.blit(tmp, pos)


class SpikeHistogramDebugger(Debugger):

    def __init__(self, net : network.Network = None, anchor=Debugger.ANCHOR_CENTER, *args, **kwargs):
        super().__init__()

        self.cell_w = 1
        self.cell_h = 1
        self.net = net
        self.spike_trace = net.spike_trace[:,-100:]

    def update(self):
        super().update()

    def draw(self):
        super().draw()

        inds = np.where(self.spike_trace != 0)

        for i, j in np.nditer(inds, flags=['zerosize_ok']):
            pos = (j * self.cell_w, i * self.cell_h)
            size = (self.cell_w, self.cell_h)
            rect = pygame.Rect(pos, size)

            color = (255, 255, 255, 255 * self.spike_trace[i,j])

            pygame.draw.rect(self.buffer, color, rect)


class FPSCounter(Debugger):

    def __init__(self, *args, **kwargs):
        super().__init__(pos=(-48,16), size=(48,16), ratio=1, anchor=Debugger.ANCHOR_RIGHT, *args, **kwargs)

        self.delta_time = 0

        self.font_name = pygame.freetype.get_default_font()
        self.font_size = 16

        self.font = pygame.freetype.SysFont('Arial', self.font_size)

    def draw(self):
        super().draw()

        self.delta_time = util.Time.delta_time

        fps = 1 / (self.delta_time / 1000) if self.delta_time else 999
        string = f'{round(fps)}'

        # pos = (self.size[0] - 36, 12)

        self.font.render_to(self.buffer, (0,0), string, (255,255,255))


class AgentDebugger(Debugger):

    def __init__(self, a : agent.Agent, *args, **kwargs):
        super().__init__(pos=(-100,-30), anchor=Debugger.ANCHOR_BOTTOM|Debugger.ANCHOR_RIGHT, size=(100,30), ratio=1, *args, **kwargs)

        # self.clear = False

        self.agent = a
        self.net = a.net
        # self.color = (0, 255, 128, 128)
        self.agent.on_target_found.add(self.on_target_found)
        self.last_found_time = util.Time.cur_time
        self.found_delta = 0
        self.color = (255,255,255,255)

    def draw(self):
        super().draw()

        fmt=f'delta: {self.found_delta:0.02f} s'
        self.font_medium.render_to(self.buffer, (0,0), fmt, self.color)

        time = util.Time.cur_time
        fmt=f'current: {(time - self.last_found_time)/1000.0:0.02f} s'
        self.font_medium.render_to(self.buffer, (0,12), fmt, self.color)

    def update(self):
        super().update()

        # x = int(self.agent.x // self.ratio)
        # y = int(self.agent.y // self.ratio)
        # pos = (x, y)

        # self.buffer.set_at(pos, self.color)
    def on_target_found(self):
        time = util.Time.cur_time
        delta = time - self.last_found_time
        self.last_found_time = time

        delta /= 1000.0
        self.found_delta = delta

class AgentStatsDebugger(Debugger):

    def __init__(self, agent : agent.Agent, radius=15, *args, **kwargs):
        super().__init__(size=(150,100), ratio=1, *args, **kwargs)

        self.agent = agent
        self.net = agent.net

        self.color = (255,255,255,255)
        self.color_alt = (200,0,200,255)

        self.radius = radius

    def draw(self):
        super().draw()

        r = self.radius

        pos = (self.size[0] // 2 // self.ratio, self.size[1] // 2 // self.ratio)

        fmt = f'{self.agent.get_angle():.03f}'
        self.font_medium.render_to(self.buffer, (pos[0] + r,pos[1] + r), fmt, self.color)

        fmt = f'{np.mean(self.agent.net.firing_rates)}'
        self.font_medium.render_to(self.buffer, (pos[0] + r,pos[1] - r), fmt, self.color)

        # inp = self.agent.inputs[:self.agent.NUM_INPUTS]
        # inp = inp[:, np.newaxis].reshape((self.agent.NUM_INPUTS // 3, 3))

        # for i,row in enumerate(inp):
        #     fmt = f'{" ".join(f"{r:0.02f}" for r in row)}'
        #     self.font_small.render_to(self.buffer, (0, 10 * i), fmt, self.color)

        # top = i
        out = ((self.agent.L_fr, self.agent.R_fr, self.agent.F_fr),)

        top = 0
        for i,row in enumerate(out):
            fmt = f'{" ".join(f"{r:0.02f}" for r in row)}'
            self.font_small.render_to(self.buffer, (0,10 * (top + i + 1)), fmt, self.color_alt)

    def update(self):
        super().update()

        x = (self.agent.x - self.size[0] // 2) // self.ratio
        y = (self.agent.y - self.size[1] // 2) // self.ratio

        x = np.clip(x, 0, self.display_size[0] - self.size[0])
        y = np.clip(y, 0, self.display_size[1] - self.size[1])

        self.pos = (x,y)

class HeatmapDebugger(Debugger):
    # MS_PER_DRAW = 10 # Update display every 500 ms to mitigate frame drops
    def gradient_map(value, start=(255,0,128,200), mid=(0,0,0,200), end=(0,255,120,200), alpha_midpoint=True):
        # Define the color stops
        start_color = np.array(start)   # Color at 0.0
        middle_color = np.array(mid)    # Color at 0.5
        end_color = np.array(end)       # Color at 1.0

        # Create masks for the two ranges
        lower_half_mask = value <= 0.5
        upper_half_mask = value > 0.5

        # Initialize RGBA channels
        r = np.zeros_like(value, dtype=float)
        g = np.zeros_like(value, dtype=float)
        b = np.zeros_like(value, dtype=float)
        a = np.zeros_like(value, dtype=float)

        # Interpolate for lower half (0 to 0.5)
        t_lower = value[lower_half_mask] / 0.5
        r[lower_half_mask] = start_color[0] * (1 - t_lower) + middle_color[0] * t_lower
        g[lower_half_mask] = start_color[1] * (1 - t_lower) + middle_color[1] * t_lower
        b[lower_half_mask] = start_color[2] * (1 - t_lower) + middle_color[2] * t_lower
        a[lower_half_mask] = start_color[3] * (1 - t_lower) + middle_color[3] * t_lower

        # Interpolate for upper half (0.5 to 1)
        t_upper = (value[upper_half_mask] - 0.5) / 0.5
        r[upper_half_mask] = middle_color[0] * (1 - t_upper) + end_color[0] * t_upper
        g[upper_half_mask] = middle_color[1] * (1 - t_upper) + end_color[1] * t_upper
        b[upper_half_mask] = middle_color[2] * (1 - t_upper) + end_color[2] * t_upper
        a[upper_half_mask] = middle_color[3] * (1 - t_upper) + end_color[3] * t_upper

        # Stack channels into a single RGBA array
        rgba = np.stack([r, g, b, a], axis=-1).astype(int)
        return rgba

    def viridis(values, alpha_midpoint=True):
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.full_like(values, 255)

        # Masks for color segments
        mask_blue = values < 0.33
        mask_mid = (values >= 0.33) & (values < 0.66)
        mask_yellow = values >= 0.66

        # Blue to Teal (0 to 0.33)
        r[mask_blue] = 68 + (30 - 68) * (values[mask_blue] / 0.33)
        g[mask_blue] = 1 + (144 - 1) * (values[mask_blue] / 0.33)
        b[mask_blue] = 84 + (255 - 84) * (values[mask_blue] / 0.33)

        # Teal to Green (0.33 to 0.66)
        r[mask_mid] = 30 + (253 - 30) * ((values[mask_mid] - 0.33) / 0.33)
        g[mask_mid] = 144 + (231 - 144) * ((values[mask_mid] - 0.33) / 0.33)
        b[mask_mid] = 255 + (37 - 255) * ((values[mask_mid] - 0.33) / 0.33)

        # Green to Yellow (0.66 to 1)
        r[mask_yellow] = 253 + (255 - 253) * ((values[mask_yellow] - 0.66) / 0.34)
        g[mask_yellow] = 231 + (255 - 231) * ((values[mask_yellow] - 0.66) / 0.34)
        b[mask_yellow] = 37 + (191 - 37) * ((values[mask_yellow] - 0.66) / 0.34)

        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            mask_low_alpha = values < 0.5
            a[mask_low_alpha] = 200 - (200 - 0) * (values[mask_low_alpha] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            mask_high_alpha = values >= 0.5
            a[mask_high_alpha] = 0 + (200 - 0) * ((values[mask_high_alpha] - 0.5) * 2)

        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def magma(values, alpha_midpoint=True):
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        mask_purple = values < 0.33
        mask_orange = (values >= 0.33) & (values < 0.66)
        mask_yellow = values >= 0.66

        # Dark Purple to Deep Orange (0 to 0.33)
        r[mask_purple] = 0 + (187 - 0) * (values[mask_purple] / 0.33)
        g[mask_purple] = 0 + (55 - 0) * (values[mask_purple] / 0.33)
        b[mask_purple] = 4 + (125 - 4) * (values[mask_purple] / 0.33)

        # Deep Orange to Light Orange (0.33 to 0.66)
        r[mask_orange] = 187 + (252 - 187) * ((values[mask_orange] - 0.33) / 0.33)
        g[mask_orange] = 55 + (157 - 55) * ((values[mask_orange] - 0.33) / 0.33)
        b[mask_orange] = 125 + (54 - 125) * ((values[mask_orange] - 0.33) / 0.33)

        # Light Orange to Yellow (0.66 to 1)
        r[mask_yellow] = 252 + (255 - 252) * ((values[mask_yellow] - 0.66) / 0.34)
        g[mask_yellow] = 157 + (255 - 157) * ((values[mask_yellow] - 0.66) / 0.34)
        b[mask_yellow] = 54 + (191 - 54) * ((values[mask_yellow] - 0.66) / 0.34)

        a[:] = 255 * values

        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def inferno(values, alpha_midpoint=True):
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.full_like(values, 255)

        mask_purple = values < 0.33
        mask_orange = (values >= 0.33) & (values < 0.66)
        mask_yellow = values >= 0.66

        # Purple to Dark Orange (0 to 0.33)
        r[mask_purple] = 0 + (230 - 0) * (values[mask_purple] / 0.33)
        g[mask_purple] = 0 + (74 - 0) * (values[mask_purple] / 0.33)
        b[mask_purple] = 4 + (24 - 4) * (values[mask_purple] / 0.33)

        # Dark Orange to Bright Orange (0.33 to 0.66)
        r[mask_orange] = 230 + (252 - 230) * ((values[mask_orange] - 0.33) / 0.33)
        g[mask_orange] = 74 + (177 - 74) * ((values[mask_orange] - 0.33) / 0.33)
        b[mask_orange] = 24 + (53 - 24) * ((values[mask_orange] - 0.33) / 0.33)

        # Bright Orange to Yellow (0.66 to 1)
        r[mask_yellow] = 252 + (255 - 252) * ((values[mask_yellow] - 0.66) / 0.34)
        g[mask_yellow] = 177 + (255 - 177) * ((values[mask_yellow] - 0.66) / 0.34)
        b[mask_yellow] = 53 + (191 - 53) * ((values[mask_yellow] - 0.66) / 0.34)

        a[:] = 255 * values

        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def plasma(values, alpha_midpoint=True):
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        # Masks for color segments
        mask_purple = values < 0.33
        mask_mid = (values >= 0.33) & (values < 0.66)
        mask_yellow = values >= 0.66

        # Purple to Pink (0 to 0.33)
        r[mask_purple] = 12 + (239 - 12) * (values[mask_purple] / 0.33)
        g[mask_purple] = 7 + (81 - 7) * (values[mask_purple] / 0.33)
        b[mask_purple] = 134 + (156 - 134) * (values[mask_purple] / 0.33)

        # Pink to Orange (0.33 to 0.66)
        r[mask_mid] = 239 + (245 - 239) * ((values[mask_mid] - 0.33) / 0.33)
        g[mask_mid] = 81 + (210 - 81) * ((values[mask_mid] - 0.33) / 0.33)
        b[mask_mid] = 156 + (66 - 156) * ((values[mask_mid] - 0.33) / 0.33)

        # Orange to Yellow (0.66 to 1)
        r[mask_yellow] = 245 + (255 - 245) * ((values[mask_yellow] - 0.66) / 0.34)
        g[mask_yellow] = 210 + (255 - 210) * ((values[mask_yellow] - 0.66) / 0.34)
        b[mask_yellow] = 66 + (191 - 66) * ((values[mask_yellow] - 0.66) / 0.34)

        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            mask_low_alpha = values < 0.5
            a[mask_low_alpha] = 200 - (200 - 0) * (values[mask_low_alpha] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            mask_high_alpha = values >= 0.5
            a[mask_high_alpha] = 0 + (200 - 0) * ((values[mask_high_alpha] - 0.5) * 2)

        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def zebra(values, *args, **kwargs):
        # Initialize RGBA arrays
        r = 255 * values  # From 0 (gray) to 255 (white)
        g = 255 * values  # From 0 (gray) to 255 (white)
        b = 255 * values  # From 0 (gray) to 255 (white)
        a = 255 * values  # Transparent if value is low

        # Stack RGBA channels into a single array
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def spectral(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        # Create masks for each segment of the colormap
        mask_red_yellow = values < 0.25
        mask_yellow_green = (values >= 0.25) & (values < 0.5)
        mask_green_cyan = (values >= 0.5) & (values < 0.75)
        mask_cyan_blue = values >= 0.75

        # Interpolate colors for each segment
        # Red to Yellow (0 to 0.25)
        r[mask_red_yellow] = 158 + (255 - 158) * (values[mask_red_yellow] / 0.25)
        g[mask_red_yellow] = 1 + (255 - 1) * (values[mask_red_yellow] / 0.25)
        b[mask_red_yellow] = 66 + (72 - 66) * (values[mask_red_yellow] / 0.25)

        # Yellow to Green (0.25 to 0.5)
        r[mask_yellow_green] = 255 + (127 - 255) * ((values[mask_yellow_green] - 0.25) / 0.25)
        g[mask_yellow_green] = 255 + (191 - 255) * ((values[mask_yellow_green] - 0.25) / 0.25)
        b[mask_yellow_green] = 72 + (123 - 72) * ((values[mask_yellow_green] - 0.25) / 0.25)

        # Green to Cyan (0.5 to 0.75)
        r[mask_green_cyan] = 127 + (120 - 127) * ((values[mask_green_cyan] - 0.5) / 0.25)
        g[mask_green_cyan] = 191 + (198 - 191) * ((values[mask_green_cyan] - 0.5) / 0.25)
        b[mask_green_cyan] = 123 + (194 - 123) * ((values[mask_green_cyan] - 0.5) / 0.25)

        # Cyan to Blue (0.75 to 1)
        r[mask_cyan_blue] = 120 + (50 - 120) * ((values[mask_cyan_blue] - 0.75) / 0.25)
        g[mask_cyan_blue] = 198 + (136 - 198) * ((values[mask_cyan_blue] - 0.75) / 0.25)
        b[mask_cyan_blue] = 194 + (189 - 194) * ((values[mask_cyan_blue] - 0.75) / 0.25)

        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            a[mask_red_yellow] = 200 - (200 - 0) * (values[mask_red_yellow] / 0.5)
            a[mask_yellow_green] = 200 - (200 - 0) * (values[mask_yellow_green] / 0.5)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            a[mask_green_cyan] = 0 + (200 - 0) * ((values[mask_green_cyan] - 0.5) / 0.5)
            a[mask_cyan_blue] = 0 + (200 - 0) * ((values[mask_cyan_blue] - 0.5) / 0.5)

        # Stack RGBA channels into a single array with shape (N, 4) and convert to integers
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def mabu(values, alpha_midpoint=True):
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        mask_low = values < 0.5
        mask_high = ~mask_low

        r[mask_low] = 230 + (230 - 0) * values[mask_low] * 2
        g[mask_low] = 0
        b[mask_low] = 200 + (250 - 200) * values[mask_low] * 2

        r[mask_high] = 230 + (0 - 230) * values[mask_high] * 2
        g[mask_high] = 0
        b[mask_high] = 200 + (250 - 200) * values[mask_high] * 2

        a[:] = 200

        if alpha_midpoint:
            a[mask_low] = 200 - (200 - 0) * values[mask_low] * 2
            a[mask_high] = 0 + (200 - 0) * values[mask_high] * 2

        rgba = np.stack([r, g, b, a], axis=-1).astype(int)
        return rgba

    def ylgn(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)  # Alpha channel with specified fading

        # Create masks for each color segment
        mask_yellow_to_light_green = values < 0.33
        mask_light_green_to_green = (values >= 0.33) & (values < 0.66)
        mask_green_to_dark_green = values >= 0.66

        # Yellow to Light Green (0 to 0.33)
        r[mask_yellow_to_light_green] = 255 + (173 - 255) * (values[mask_yellow_to_light_green] / 0.33)
        g[mask_yellow_to_light_green] = 255 + (233 - 255) * (values[mask_yellow_to_light_green] / 0.33)
        b[mask_yellow_to_light_green] = 204 + (47 - 204) * (values[mask_yellow_to_light_green] / 0.33)

        # Light Green to Green (0.33 to 0.66)
        r[mask_light_green_to_green] = 173 + (35 - 173) * ((values[mask_light_green_to_green] - 0.33) / 0.33)
        g[mask_light_green_to_green] = 233 + (139 - 233) * ((values[mask_light_green_to_green] - 0.33) / 0.33)
        b[mask_light_green_to_green] = 47 + (69 - 47) * ((values[mask_light_green_to_green] - 0.33) / 0.33)

        # Green to Dark Green (0.66 to 1)
        r[mask_green_to_dark_green] = 35 + (0 - 35) * ((values[mask_green_to_dark_green] - 0.66) / 0.34)
        g[mask_green_to_dark_green] = 139 + (69 - 139) * ((values[mask_green_to_dark_green] - 0.66) / 0.34)
        b[mask_green_to_dark_green] = 69 + (20 - 69) * ((values[mask_green_to_dark_green] - 0.66) / 0.34)


        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            mask_low_alpha = values < 0.5
            a[mask_low_alpha] = 200 - (200 - 0) * (values[mask_low_alpha] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            mask_high_alpha = values >= 0.5
            a[mask_high_alpha] = 0 + (200 - 0) * ((values[mask_high_alpha] - 0.5) * 2)

        # Stack RGBA channels into a single array
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def puor(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        # Create masks for values below and above 0.5
        mask_low = values < 0.5
        mask_high = ~mask_low  # Equivalent to values >= 0.5

        # Calculate colors for values < 0.5 (purple to white)
        r[mask_low] = 127 + (247 - 127) * values[mask_low] * 2
        g[mask_low] = 59 + (247 - 59) * values[mask_low] * 2
        b[mask_low] = 126 + (247 - 126) * values[mask_low] * 2

        # Calculate colors for values >= 0.5 (white to orange)
        r[mask_high] = 247 + (179 - 247) * (values[mask_high] - 0.5) * 2
        g[mask_high] = 247 + (88 - 247) * (values[mask_high] - 0.5) * 2
        b[mask_high] = 247 + (6 - 247) * (values[mask_high] - 0.5) * 2

        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            a[mask_low] = 200 - (200 - 0) * (values[mask_low] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            a[mask_high] = 0 + (200 - 0) * ((values[mask_high] - 0.5) * 2)

        # Stack RGBA channels into a single array with shape (N, 4) and convert to integers
        rgba = np.stack([r, g, b, a], axis=-1).astype(int)
        return rgba

    def coolwarm(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        # Masks for two segments
        mask_blue_white = values < 0.5
        mask_white_red = ~mask_blue_white

        # Blue to White (0 to 0.5)
        r[mask_blue_white] = 59 + (255 - 59) * (values[mask_blue_white] * 2)
        g[mask_blue_white] = 76 + (255 - 76) * (values[mask_blue_white] * 2)
        b[mask_blue_white] = 192 + (255 - 192) * (values[mask_blue_white] * 2)

        # White to Red (0.5 to 1)
        r[mask_white_red] = 255 + (180 - 255) * ((values[mask_white_red] - 0.5) * 2)
        g[mask_white_red] = 255 + (4 - 255) * ((values[mask_white_red] - 0.5) * 2)
        b[mask_white_red] = 255 + (38 - 255) * ((values[mask_white_red] - 0.5) * 2)

        a[:] = 200

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            a[mask_blue_white] = 200 - (200 - 0) * (values[mask_blue_white] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            a[mask_white_red] = 0 + (200 - 0) * ((values[mask_white_red] - 0.5) * 2)

        # Stack RGBA channels into a single array
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def rdgr(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        mask_red_white = values < 0.5
        mask_white_green = ~mask_red_white

        # r[mask_red_white] = 200
        # g[mask_red_white] = 255 * (values[mask_red_white] * 2)
        # b[mask_red_white] = 255 * (values[mask_red_white] * 2)

        # r[mask_white_green] = 255 * (1 - (values[mask_white_green] - 0.5) * 2)
        # b[mask_white_green] = 255 * (1 - (values[mask_white_green] - 0.5) * 2)
        # g[mask_white_green] = 200

        r[mask_red_white] = 255 * (values[mask_red_white] * 2)
        g[mask_red_white] = 0
        # b[mask_red_white] = 120 * (values[mask_red_white] * 2)
        b[mask_red_white] = 120
        a[mask_red_white] = 255 * (values[mask_red_white] * 2)

        r[mask_white_green] = 0
        g[mask_white_green] = 255 * (1 - (values[mask_white_green] - 0.5) * 2)
        # g[mask_white_green] = 255 * (1 - (values[mask_white_green] - 0.5) * 2)
        b[mask_white_green] = 120
        a[mask_white_green] = 255 * (1 - (values[mask_white_green] - 0.5) * 2)

        # r[mask_red_white] = 103 + (255 - 103) * (values[mask_red_white] * 2)
        # g[mask_red_white] = 31 + (255 - 31) * (values[mask_red_white] * 2)
        # b[mask_red_white] = 0 + (255 - 0) * (values[mask_red_white] * 2)

        # r[mask_white_green] = 255 + (34 - 255) * ((values[mask_white_green] - 0.5) * 2)
        # b[mask_white_green] = 255 + (94 - 255) * ((values[mask_white_green] - 0.5) * 2)
        # g[mask_white_green] = 255 + (168 - 255) * ((values[mask_white_green] - 0.5) * 2)

        # a[:] = 128

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            a[mask_red_white] = 128 - (128 - 0) * (values[mask_red_white] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            a[mask_white_green] = 0 + (128 - 0) * ((values[mask_white_green] - 0.5) * 2)

        # Stack RGBA channels into a single array
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def rdbu(values, alpha_midpoint=True):
        # Initialize RGBA arrays
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)
        a = np.zeros_like(values)

        # Create masks for values below and above 0.5
        mask_red_white = values < 0.5
        mask_white_blue = ~mask_red_white  # Equivalent to values >= 0.5

        # Bright Red to White (0 to 0.5)
        r[mask_red_white] = 200  # Start at bright red
        g[mask_red_white] = 255 * (values[mask_red_white] * 2)  # Interpolate to white
        b[mask_red_white] = 255 * (values[mask_red_white] * 2)

        # White to Bright Blue (0.5 to 1)
        r[mask_white_blue] = 255 * (1 - (values[mask_white_blue] - 0.5) * 2)  # Fade red out to blue
        g[mask_white_blue] = 255 * (1 - (values[mask_white_blue] - 0.5) * 2)  # Fade green out to blue
        b[mask_white_blue] = 200  # End at bright blue

        # # Red to White (0 to 0.5)
        # r[mask_red_white] = 103 + (255 - 103) * (values[mask_red_white] * 2)
        # g[mask_red_white] = 0 + (255 - 0) * (values[mask_red_white] * 2)
        # b[mask_red_white] = 31 + (255 - 31) * (values[mask_red_white] * 2)

        # # White to Blue (0.5 to 1)
        # r[mask_white_blue] = 255 + (34 - 255) * ((values[mask_white_blue] - 0.5) * 2)
        # g[mask_white_blue] = 255 + (94 - 255) * ((values[mask_white_blue] - 0.5) * 2)
        # b[mask_white_blue] = 255 + (168 - 255) * ((values[mask_white_blue] - 0.5) * 2)

        a[:] = 128

        if alpha_midpoint:
            # Alpha fade: 200 to 0 (midpoint) back to 200
            # Alpha transitions from 200 to 0 for values < 0.5
            a[mask_red_white] = 128 - (128 - 0) * (values[mask_red_white] * 2)

            # Alpha transitions from 0 back to 200 for values >= 0.5
            a[mask_white_blue] = 0 + (128 - 0) * ((values[mask_white_blue] - 0.5) * 2)

        # Stack RGBA channels into a single array
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    maps = {
        'rdgr': lambda x: HeatmapDebugger.gradient_map(x, (255, 0, 120, 200), (0, 0, 0, 200), (0, 255, 120, 200)),
        'puor': lambda x: HeatmapDebugger.gradient_map(x, (128, 50, 128, 200), (0, 0, 0, 0), (247, 247, 247, 200)),
        'ylgn': lambda x: HeatmapDebugger.gradient_map(x, (100, 255, 0, 200), (0, 0, 0, 200), (0, 255, 0, 200)),
        'rdbu': lambda x: HeatmapDebugger.gradient_map(x, (255, 0, 50, 200), (0, 0, 0, 0), (50, 0, 255, 200)),
        'zebra': lambda x: HeatmapDebugger.gradient_map(x, (0, 0, 0, 200), (100, 100, 100, 200), (255, 255, 255, 200)),
        'mabu': lambda x: HeatmapDebugger.gradient_map(x, (255, 0, 255, 200), (0,0,0,0), (0, 0, 255, 200)),
        'plasma': lambda x: HeatmapDebugger.gradient_map(x, (13, 8, 135, 200), (189, 55, 84, 200), (246, 251, 130, 200)),
        'coolwarm': lambda x: HeatmapDebugger.gradient_map(x, (59, 76, 192, 200), (255, 255, 255, 200), (180, 4, 38, 200)),
        'inferno': lambda x: HeatmapDebugger.gradient_map(x, (0, 0, 4, 200), (120, 15, 100, 200), (252, 255, 164, 200)),
        'magma': lambda x: HeatmapDebugger.gradient_map(x, (0, 0, 3, 200), (128, 18, 97, 200), (252, 253, 191, 200)),
        'viridis': lambda x: HeatmapDebugger.gradient_map(x, (68, 1, 84, 200), (32, 144, 140, 200), (253, 231, 37, 200)),
        'spectral': lambda x: HeatmapDebugger.gradient_map(x, (158, 1, 66, 200), (255, 255, 191, 200), (94, 79, 162, 200)),
        'fallback': lambda x: HeatmapDebugger.gradient_map(x),
    }

    # maps = (
    #     viridis,
    #     magma,
    #     inferno,
    #     plasma,
    #     zebra,
    #     coolwarm,
    #     spectral,
    #     puor,
    #     rdbu,
    #     mabu,
    #     ylgn,
    #     rdgr,
    #     gradient_map,
    # )
    # maps = {m.__name__: m for m in maps}

    def __init__(self, alpha=0.5, ratio=2, zero_empty=False, *args, **kwargs):
        super().__init__(ratio=ratio, *args, **kwargs)

        self.heatmap = np.zeros((self.size[0], self.size[1]))
        self.cur_time = pygame.time.get_ticks()
        self.prev_time = self.cur_time
        self.delta_time = 0
        self.time_accum = 0
        self.draw_order = 999
        self.alpha = alpha
        self.kernel = None
        self.data = None
        self.cmap = self.maps['fallback']

        self.inspect = (0,0)
        self.diagonalize = False
        self.combine = np.multiply
        self.zero_empty = zero_empty
        self.apply_kernel = True

        F = 1
        H = F / 2
        Q = F / 4

        self.kernels = [
            None,
            1 * np.array([
                [1]
            ]),
            1/4 * np.array([
                [1, 1],
                [1, 1],
            ]),
            1/9 * np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]),
            1/16 * np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]),
            1/25 * np.array([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ])
            # np.array([
            #     [F],
            # ]),
            # np.array([
            #     [H, H],
            #     [H, H],
            # ]),
            # np.array([
            #     [0, H, 0],
            #     [H, F, H],
            #     [0, H, 0],
            # ]),
            # np.array([
            #     [0, H, H, 0],
            #     [H, F, F, H],
            #     [H, F, F, H],
            #     [0, H, H, 0],
            # ]),
            # np.array([
            #     [0, Q, H, Q, 0],
            #     [Q, H, F, H, Q],
            #     [H, F, F, F, H],
            #     [Q, H, F, H, Q],
            #     [0, Q, H, Q, 0],
            # ]),
        ]

        self.kernel = self.kernels[0]
        self.min = None
        self.max = None
        self.transpose = True

        # self.buffer.set_alpha(128)

        # self.kernel = np.array([
        #     [0, H, H, 0],
        #     [H, F, F, H],
        #     [H, F, F, H],
        #     [0, H, H, 0],
        #     ])

    def update(self):
        self.cur_time = pygame.time.get_ticks()
        self.delta_time = self.cur_time - self.prev_time
        self.prev_time = self.cur_time
        self.time_accum += self.delta_time

        # min = np.abs(np.min(self.data))
        # max = np.max(self.data + min)

        # data = self.data + min
        # data = data / max if max != 0 else data

        if isinstance(self.data, tuple):
            data = np.ones_like(self.data[0])
            for d in self.data:
                data *= d
            # max = np.max(np.abs(data))
            max = self.max if self.max is not None else np.max(np.abs(data))
            data = data / max if max != 0 else data
        elif isinstance(self.data, np.ndarray):
            # max = np.max(np.abs(self.data))
            max = self.max if self.max is not None else np.max(np.abs(self.data))
            data = self.data / max if max != 0 else self.data

        if self.diagonalize:
            norm = np.linalg.norm(data)
            # data = np.outer(data, data.T) / np.linalg.norm(data) * np.eye(data.shape[0])
            data = np.outer(data, data.T)
            if norm > 0:
                data /= norm
            data *= np.eye(data.shape[0], dtype=data.dtype)

        self.heatmap[:] = self.alpha * data + (1 - self.alpha) * self.heatmap[:]

    def draw(self):
        super().draw()

        # if self.time_accum < AgentHeatmapDebugger.MS_PER_DRAW:
        #     return

        map = self.cmap

        self.time_accum = 0
        norm = self.heatmap

        min = self.min if self.min is not None else np.min(norm)
        norm = norm + np.abs(min)
        norm = norm / np.max(np.abs(norm)) if np.max(np.abs(norm)) != 0 else norm

        # min = np.min(norm)
        # norm = norm - min if min < 0 else norm
        # norm = norm / np.max(np.abs(norm)) if np.max(np.abs(norm)) > 0 else norm
        # norm = norm.T

        if self.apply_kernel:
            px = scipy.ndimage.convolve(norm, self.kernel) if self.kernel is not None else norm
            # rgba_array = self.cmap(px)
        else:
            px = norm
            # rgba_array = self.cmap(px)

        if self.transpose:
            px = px.T

        rgba_array = self.cmap(px)
        if self.zero_empty:
            mask = self.data != 0
            mask = ~mask
            rgba_array[mask,:] = 0

        # mask = px > 0
        # mask = ~mask

        # rgba_array[mask,3] = 0

        # Create an RGBA surface directly with SRCALPHA
        surf = pygame.surfarray.make_surface(rgba_array[:, :, :3])
        surf = surf.convert_alpha()

        pygame.surfarray.pixels_alpha(surf)[:,:] = rgba_array[:,:,3]

        alpha = pygame.surfarray.pixels_alpha(surf)
        alpha[:,:] = rgba_array[:,:,3]
        del alpha

        self.buffer.blit(surf, (0,0))

    def on_mouse_over(self, pos):
        self.inspect = pos

    def make_kernel(self, n):
        return np.full((n, n), 1, dtype=np.float64)

    def set_kernel(self, id):
        self.kernel = self.kernels[id]

    def set_data_ref(self, array, combine=np.multiply):
        self.data = array
        self.combine = combine

    def set_color_map(self, cmap=None, alpha_midpoint=True):
        # self.cmap = lambda v: self.maps[map](v)
        cm = self.maps.get(cmap)
        cm = cm or self.maps['fallback']
        # self.cmap = lambda v: cm(v)
        self.cmap = cm

    def set_diagonalize(self, diag):
        self.diagonalize = diag

    def toggle_kernel(self):
        self.apply_kernel = not self.apply_kernel

    def set_data_min(self, min):
        self.min = min

    def set_data_max(self, max):
        self.max = max

class AgentHeatmapDebugger(Debugger):

    MS_PER_DRAW = 500 # Update display every 500 ms to mitigate frame drops

    def __init__(self, agent : agent.Agent, *args, **kwargs):
        super().__init__(ratio=8)

        self.agent = agent
        self.heatmap = np.zeros((self.size[0], self.size[1]))
        self.cur_time = pygame.time.get_ticks()
        self.prev_time = self.cur_time
        self.delta_time = 0
        self.time_accum = 0
        self.draw_order = 999
        self.decay_const = 1 / 5000

    def update(self):
        self.cur_time = pygame.time.get_ticks()
        self.delta_time = self.cur_time - self.prev_time
        self.prev_time = self.cur_time
        self.time_accum += self.delta_time

        i = int(self.agent.x // self.ratio)
        j = int(self.agent.y // self.ratio)
        self.heatmap[i,j] += 1

        # if self.time_accum >= self.MS_PER_UPDATE:
        #     self.time_accum = 0

        #     i = int(self.agent.x // self.ratio)
        #     j = int(self.agent.y // self.ratio)

        #     self.heatmap[i, j] += 1
        self.heatmap *= np.exp(-self.decay_const)

        small = np.where(self.heatmap < 0.1)
        self.heatmap[small] = 0

    def draw(self):

        if self.time_accum < AgentHeatmapDebugger.MS_PER_DRAW:
            return

        self.time_accum = 0

        mv = np.max(self.heatmap)
        norm = (self.heatmap / mv) if mv > 0 else self.heatmap

        F = 1
        H = F / 2
        Q = F / 4

        kernel = np.array([
            [0, Q, H, Q, 0],
            [Q, Q, H, Q, Q],
            [H, H, F, H, H],
            [Q, Q, H, Q, Q],
            [0, Q, H, Q, 0],
            ])

        px = scipy.ndimage.convolve(norm, kernel)

        inx = np.where(px == 0)

        for i, j in np.nditer(inx, flags=['zerosize_ok']):
            rect = pygame.Rect(i, j, 1, 1)
            self.buffer.fill((0,0,0,0),rect)

        inx = np.where(px > 0)

        for i, j in np.nditer(inx, flags=['zerosize_ok']):
            v = px[i, j]
            v = np.clip(v, 0, 1)

            r = int(255 * v)
            g = 0
            b = int(255 * (1 - v))
            a = 128 * v

            color = (r,g,b,a)

            rect = pygame.Rect(i, j, 1, 1)
            self.buffer.fill((0,0,0,0),rect)
            pygame.draw.rect(self.buffer, color, rect)

        # surf = pygame.surfarray.make_surface(255 * self.heatmap)
        # self.buffer.blit(surf, (0,0))

class SpikeHistDebugger(Debugger):

    def __init__(self, a : agent.Agent, *args, **kwargs):
        super().__init__(pos=(-200,0), size=(200,100), anchor=Debugger.ANCHOR_RIGHT, ratio=4, *args, **kwargs)

        self.agent = a
        self.cell_w = self.size[0] // a.net.num_neurons
        self.cell_h = self.cell_w

    def update(self):
        pass

    def draw(self):
        inx = np.where(self.agent.s_trace[:, :100] != 0)

        for i, j in np.nditer(inx, flags=['zerosize_ok']):
            x = j * self.cell_w
            y = i * self.cell_h

            rect = pygame.Rect(x, y, self.cell_w, self.cell_h)
            pygame.draw.rect(self.buffer, (255,255,255,255), rect)

        # rect = pygame.Rect((0,0), self.size)
        # pygame.draw.rect(self.buffer, (255,0,255,255), rect)


class TargetDebugger(Debugger):

    def __init__(self, agent : agent.Agent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent = agent
        self.net = agent.net
        self.color = (255, 0, 128, 64)
        self.target_history = set()
        self.target = agent.target

    # def update(self):

    #     x = self.target.x // self.ratio
    #     y = self.target.y // self.ratio

    #     if (x, y) not in self.target_history:
    #         self.target_history.add((x, y))

    def draw(self):
        super().draw()

        ax = int(self.agent.x // self.ratio)
        ay = int(self.agent.y // self.ratio)

        tx = int(self.target.x // self.ratio)
        ty = int(self.target.y // self.ratio)

        start = (ax, ay)
        end = (tx, ty)

        pygame.draw.line(self.buffer, self.color, start, end)

        r = self.target.radius // self.ratio + 2

        # for pos in self.target_history:
        #     pygame.draw.circle(self.buffer, (128,0,128,64), pos, r)

