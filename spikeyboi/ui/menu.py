import pygame as pg
import pygame_gui as gui

import pathlib


class Menubar(gui.elements.UIPanel):

    def __init__(self, relative_rect: pg.Rect, manager: gui.UIManager):
        super().__init(relative_rect, manager)

        self.menus = {}

    def add_menu_item(menu_path):
        menu_path = pathlib.PurePath(menu_path)

    def add_action(menu_path):
        pass
