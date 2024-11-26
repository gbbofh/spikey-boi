import io


import numpy as np
import pygame as pg
import pygame_gui as gui


import matplotlib.pyplot as plt


import networkx
import community


import spikeyboi
import spikeyboi.spikey
import spikeyboi.spikey.agent
import spikeyboi.spikey.brain
import spikeyboi.ui.debug_window


class UICommunitiesDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = self.sim.agent.brain.net

        fig, ax = plt.subplots(1,1)

        self.fig = fig
        self.ax = ax

        self.img = None
        self.original_rect = self.rect
        self.aspect_ratio = self.rect.width / self.rect.height

        self.time_accum = 10.0
        self.REBUILD_EVERY = 10.0 # seconds

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def on_update(self, delta_time):
        self.time_accum += delta_time
        if self.time_accum >= self.REBUILD_EVERY:
            self.time_accum -= self.REBUILD_EVERY
            self.ax.clear()

            G = networkx.from_numpy_array(self.net.w)
            partition = community.best_partition(G)

            ice = []

            for (u,v) in G.edges():
                if partition[u] == partition[v]:
                    ice.append((u,v))

            G_ice = networkx.Graph()
            G_ice.add_nodes_from(G.nodes())
            G_ice.add_edges_from(ice)

            communities = {}
            for n,id in partition.items():
                if id not in communities:
                    communities[id] = []
                communities[id].append(n)

            pos = {}
            off = 3
            cx,cy = 0,0

            for i, (cid,ns) in enumerate(communities.items()):
                sg = G_ice.subgraph(ns)
                cpos = networkx.spring_layout(sg)
                for n, (x,y) in cpos.items():
                    pos[n] = (x + cx, y + cy)
                cy += off
                if i % 2 == 1:
                    cy = 0
                    cx += off

            colors = [partition[n] for n in G_ice.nodes()]
            networkx.draw(G_ice, pos, ax=self.ax,
                        node_color=colors, cmap=plt.cm.tab20, 
                        with_labels=True, node_size=100, 
                        edge_color="gray", alpha=0.6, font_color='white')

            buf = io.BytesIO()
            self.fig.savefig(buf, format='jpg')

            buf.seek(0)

            self.img = pg.image.load(buf, '.jpg')
            self.img = self.img.convert()

            self.disp_surf.image.fill((0,0,0))

            pg.transform.scale(self.img, self.disp_surf.image.size, self.disp_surf.image)


    def on_close_window_button_pressed(self):
        self.hide()

    def process_event(self, e):
        if e.type == gui.UI_WINDOW_RESIZED:
            if e.ui_element == self:
                w, h = self.get_abs_rect().size
                if w / h > self.aspect_ratio:
                    h = w / self.aspect_ratio
                else:
                    w = h * self.aspect_ratio
                self.set_dimensions((w,h))

                self.disp_surf.image.fill((0,0,0))
                pg.transform.scale(self.img, self.disp_surf.image.size, self.disp_surf.image)

                return True
        return super().process_event(e)

    def on_load_completed(self):
        pass

    def on_agent_selected(self, agent):
        self.net = agent.brain.net
        self.time_accum = 10.0
