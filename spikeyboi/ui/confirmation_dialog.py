import pygame as pg
import pygame_gui as gui


class UIConfirmationDialog(gui.windows.ui_confirmation_dialog.UIConfirmationDialog):

    def __init__(self, confirm_callback, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.confirm_button.bind(gui.UI_BUTTON_PRESSED, confirm_callback)
