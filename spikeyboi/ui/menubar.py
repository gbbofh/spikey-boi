import pygame
import pygame_gui


import locale


class UIMenuBar(pygame_gui.elements.UIPanel):
    def __init__(self, relative_rect, manager, menu_data):
        super().__init__(relative_rect, manager=manager)

        locale.setlocale(locale.LC_ALL, '')

        # toolbar buttons - not associated with a panel, just an action
        self.toolbar_buttons = []

        # Dictionary to hold menu buttons and their dropdown panels
        self.menu_buttons = {}
        self.dropdown_panels = {}
        self.action_callbacks = {}

        self.menu_count = len(menu_data.keys())
        self.menu_width = 0
        self.button_width = 90
        self.button_x = 20

        # Create menu buttons and dropdowns based on menu_data
        button_x = 20
        for menu_name, options in menu_data.items():
            # Create the main menu button
            button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((self.button_x, 0), (self.button_width - 10, relative_rect.height)),
                text=menu_name,
                manager=manager,
                container=self,
                object_id=pygame_gui.core.ObjectID(class_id='@menu_button')
            )
            self.menu_buttons[menu_name] = button

            # Create a hidden dropdown panel for the menu
            dropdown_panel = pygame_gui.elements.UIPanel(
                relative_rect=pygame.Rect((self.button_x, relative_rect.height), (120, 30 * len(options) + 5)),
                manager=manager,
                visible=False,
                starting_height=10
            )
            self.dropdown_panels[menu_name] = dropdown_panel

            # Create options in the dropdown
            for i, option in enumerate(options):
                option_button = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, i * 30), (100, 30)),
                    text=option,
                    manager=manager,
                    container=dropdown_panel,
                    anchors={'centerx': 'centerx'},
                    object_id=pygame_gui.core.ObjectID(class_id='@menu_button')
                )

                # Store the action associated with the option
                option_button.action = option

            self.button_x += self.button_width  # Adjust for the next button
            self.menu_width = self.button_x

        self.add_spacer(20)

        # Track the open state of the dropdowns
        self.open_dropdown = None

    def bind_action(self, action, callback):
        self.action_callbacks[action] = callback

    def add_toolbar_button(self, id, callback=None):
        # button_x = self.menu_width + 20
        # button_x += 50 * len(self.toolbar_buttons) + 10
        button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.button_x, 0), (50, self.relative_rect.height - 5)),
            text='',
            object_id=id,
            manager=self.ui_manager,
            container=self)

        self.button_x += 50

        button.bind(pygame_gui.UI_BUTTON_PRESSED, callback)
        self.toolbar_buttons.append(button)

        return button

    def add_spacer(self, width):
        self.button_x += width

    def process_event(self, event):
        # Handle button presses for menu options
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                # Check if a menu button was clicked
                for menu_name, button in self.menu_buttons.items():
                    if event.ui_element == button:
                        # Toggle dropdown visibility
                        if self.open_dropdown == menu_name:
                            self.dropdown_panels[menu_name].hide()
                            self.open_dropdown = None
                        else:
                            if self.open_dropdown is not None:
                                # Hide the previously opened dropdown
                                self.dropdown_panels[self.open_dropdown].hide()
                            self.dropdown_panels[menu_name].show()
                            self.open_dropdown = menu_name
                        return True  # Event has been consumed

                # Check if an option button was clicked in any dropdown
                if self.open_dropdown is not None:
                    for element in self.dropdown_panels[self.open_dropdown].get_container().elements:
                        if event.ui_element == element:
                            # Print once when an option is selected and immediately close the dropdown
                            # print(f"{element.action} selected from {self.open_dropdown}")
                            cb = self.action_callbacks.get(element.action, None)
                            if cb:
                                cb()
                            self.dropdown_panels[self.open_dropdown].hide()
                            self.open_dropdown = None
                            return True  # Event has been consumed

        return super().process_event(event)

