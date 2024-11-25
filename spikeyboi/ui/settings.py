import pygame as pg
import pygame_gui as gui


class UISettingsWindow(gui.elements.UIWindow):

    def __init__(self, rect, manager):
        rect = pg.Rect(rect)
        super().__init__(rect, manager=manager, window_display_title='Settings')

        # container_rect = self.get_container().get_relative_rect()
        container = self.get_container()
        size = container.get_size()
        tabs_rect = pg.Rect((10,10), (size[0] - 20, size[1] - 20))
        self.tabs = gui.elements.UITabContainer(tabs_rect, manager=manager, container=container, parent_element=self)
        # id = self.tabs.add_tab('Test 1', '#tab1')
        # tab = self.tabs.get_tab_container(id)

        # self.tabs.add_tab('Test 2', '#tab2')

        display_tab_id = self.tabs.add_tab('Display', '#display_tab')
        display_tab = self.tabs.get_tab_container(display_tab_id)
