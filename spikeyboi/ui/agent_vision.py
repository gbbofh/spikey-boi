import numpy as np
import scipy as sp
import pygame as pg
import pygame_gui as gui


import spikeyboi.ui
import spikeyboi.ui.debug_window


class UIAgentVisionDebugger(spikeyboi.ui.debug_window.UIDebugWindow):

    def __init__(self, *args, **kwargs):
        self.toggle_depth_button = None

        super().__init__(*args, **kwargs)
        self.brain = self.sim.agent.brain

        # First 7 inputs are raycast results
        self.buffer = pg.Surface((self.brain.num_inputs - 3, self.brain.num_inputs - 3), pg.SRCALPHA)
        self.kernel = 1 / 16 * np.array([
            [ 1, 2, 1 ],
            [ 2, 4, 2 ],
            [ 1, 2, 1 ]
        ])

        self.kernel_enabled = True
        self.show_depth = True

        self.data = np.zeros((self.brain.num_inputs - 3, self.brain.num_inputs - 3), dtype=np.float64)
        self.alpha = 1.0

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

    def rebuild(self):
        super().rebuild()

        titlebar = self.title_bar


        if self.toggle_depth_button:
            pos = (-2 * self.title_bar_height, 0)
            dims = (self.title_bar_close_button_width, self.title_bar_height)

            self.toggle_depth_button.set_dimensions(dims)
            self.toggle_depth_button.set_relative_position(pos)
        else:
            rect = self._window_root_container.relative_rect
            dims = (rect.width - 2 * self.title_bar_close_button_width, self.title_bar_height)
            titlebar.set_dimensions(dims)

            rect = pg.Rect((-2 * self.title_bar_close_button_width,0),(self.title_bar_close_button_width,self.title_bar_height))

            self.toggle_depth_button = gui.elements.UIButton(rect, '\u23e5',
                                                    manager=self.ui_manager,
                                                    container=self._window_root_container,
                                                    parent_element=self,
                                                    object_id='#toggle_depth',
                                                    anchors={
                                                        'top': 'top',
                                                        'bottom': 'top',
                                                        'left': 'right',
                                                        'right': 'right',
                                                    })
            self.toggle_depth_button.bind(gui.UI_BUTTON_PRESSED, self.toggle_depth_enable)

    def show(self):
        super().show()

        self.toggle_depth_button.show()

    def hide(self):
        super().hide()

        self.toggle_depth_button.hide()

    def toggle_depth_enable(self):
        self.show_depth = not self.show_depth
        if not self.show_depth:
            # self.toggle_depth_button.set_text('\u0336'.join('3D')+'\u0336')
            self.toggle_depth_button.set_text('\u23e2')
        else:
            self.toggle_depth_button.set_text('\u23e5')

    def on_update(self, delta_time):
        self.data[:] = 0
        if self.show_depth:
            self.raycast_3d(delta_time)
        else:
            self.data[:] = self.brain.inputs[np.newaxis, :-3]
        values = self.data.T / 0.8

        # rgba = spikeyboi.ui.colormaps['zebra'](values)
        colorstops = [
            (0, (0,0,0,200)),
            (0.2, (100, 100, 255, 200)),
            (0.4, (255, 100, 100, 200)),
            (0.5, (100, 255, 100, 200)),
        ]
        rgba = spikeyboi.ui.gradient_map(values, colorstops)

        if not (self.kernel is None) and self.kernel_enabled:
            rgba[:,:,0] = sp.ndimage.convolve(rgba[:,:,0], self.kernel)
            rgba[:,:,1] = sp.ndimage.convolve(rgba[:,:,1], self.kernel)
            rgba[:,:,2] = sp.ndimage.convolve(rgba[:,:,2], self.kernel)
            rgba[:,:,3] = sp.ndimage.convolve(rgba[:,:,3], self.kernel)

        self.buffer.fill((0,0,0,255))
        pg.surfarray.blit_array(self.buffer, rgba[:,:,:-1])

        alpha = pg.surfarray.pixels_alpha(self.buffer)
        alpha[:] = rgba[:,:,-1]
        del alpha

        pg.transform.scale(self.buffer, self.disp_surf.image.size, self.disp_surf.image)

    def raycast_3d(self, delta_time):
        agent = spikeyboi.spikey.sim_instance.agent

        inv_dist = 1 - agent.mean_distances

        wall_height = (inv_dist * 7).astype(np.int32)
        wall_height = np.maximum(wall_height, 1)
        center_row = 7 // 2

        start_rows = center_row - wall_height // 2
        end_rows = center_row + wall_height // 2

        start_rows = np.clip(start_rows, 0, 7)
        end_rows = np.clip(end_rows, 0, 7)

        rows = np.arange(7).reshape(-1, 1)
        cols = np.arange(7).reshape(1, -1)

        start_rows = start_rows.reshape(1, -1)
        end_rows = end_rows.reshape(1, -1)

        mask = (rows >= start_rows) & (rows < end_rows)

        shading = np.tile((self.brain.inputs[:7]).reshape(1,-1), (7, 1))
        self.data[mask] = shading[mask]

    def on_load_completed(self):
        self.brain = self.sim.agent.brain
        self.data[:] = 0

    def on_agent_selected(self, agent):
        self.brain = agent.brain
        self.data[:] = 0
