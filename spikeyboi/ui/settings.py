import pygame as pg
import pygame_gui as gui


import json


class UISettingsWindow(gui.elements.UIWindow):

    def __init__(self, rect, manager, save_callback):
        rect = pg.Rect(rect)
        super().__init__(rect, manager=manager, window_display_title='Settings')

        self.save_callback = save_callback

        container = self.get_container()
        size = container.get_size()

        tabs_rect = pg.Rect((10,10), (size[0] - 20, size[1] - 100))

        self.tabs = gui.elements.UITabContainer(tabs_rect, manager=manager, container=container, parent_element=self)

        self.button_save = gui.elements.UIButton(pg.Rect(-60,-40,60,30),
                                                'Save', manager=manager,
                                                container=container,
                                                parent_element=self, 
                                                anchors={'centerx': 'centerx',
                                                        'bottom': 'bottom'})
        self.button_cancel = gui.elements.UIButton(pg.Rect(0,-40,60,30),
                                                'Cancel', manager=manager,
                                                container=container,
                                                parent_element=self, 
                                                anchors={'centerx': 'centerx',
                                                        'bottom': 'bottom'})

        display_tab_id = self.tabs.add_tab('Display', '#display_tab')
        display_tab = self.tabs.get_tab_container(display_tab_id)

        sim_tab_id = self.tabs.add_tab('Simulation', '#sim_tab')
        sim_tab = self.tabs.get_tab_container(sim_tab_id)

        self.config = self.load()

    def load(self):
        conf = gui.core.utility.create_resource_path('data/config.json')
        with open(conf, 'r') as fp:
            data = json.load(fp)

        return data

    def save(self):
        conf = gui.core.utility.create_resource_path('data/config.json')
        with open(conf, 'w') as fp:
            json.dump(self.config, fp)

    def process_event(self, e):
        if e.type == gui.UI_BUTTON_PRESSED and e.ui_element == self.button_save:
            self.save_callback(self.config)
