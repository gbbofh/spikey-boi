import pygame
import pygame_gui

class Menubar(pygame_gui.elements.UIPanel):
    def __init__(self, relative_rect, manager, menu_data):
        super().__init__(relative_rect, manager=manager)
        
        # Dictionary to hold menu buttons and their dropdown panels
        self.menu_buttons = {}
        self.dropdown_panels = {}
        
        # Create menu buttons and dropdowns based on menu_data
        button_x = 10
        for menu_name, options in menu_data.items():
            # Create the main menu button
            button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((button_x, 0), (80, relative_rect.height)),
                text=menu_name,
                manager=manager,
                container=self
            )
            self.menu_buttons[menu_name] = button
            
            # Create a hidden dropdown panel for the menu
            dropdown_panel = pygame_gui.elements.UIPanel(
                relative_rect=pygame.Rect((button_x, relative_rect.height), (120, 30 * len(options))),
                manager=manager,
                visible=False
            )
            self.dropdown_panels[menu_name] = dropdown_panel
            
            # Create options in the dropdown
            for i, option in enumerate(options):
                option_button = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, i * 30), (100, 30)),
                    text=option,
                    manager=manager,
                    container=dropdown_panel,
                    anchors={'centerx': 'centerx'}
                )
                
                # Store the action associated with the option
                option_button.action = option

            button_x += 90  # Adjust for the next button

        # Track the open state of the dropdowns
        self.open_dropdown = None

    def process_event(self, event):
        # Handle button presses for menu options
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
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
                    return True

                # Check if an option button was clicked in any dropdown
                for menu_name, dropdown in self.dropdown_panels.items():
                    for element in dropdown.get_container().elements:
                        if event.ui_element == element:
                            print(f"{element.action} selected from {menu_name}")
                            # Hide the dropdown after an option is selected
                            dropdown.hide()
                            self.open_dropdown = None
                            return True

        return super().process_event(event)

# Example usage:
pygame.init()
window_size = (800, 600)
window_surface = pygame.display.set_mode(window_size)

manager = pygame_gui.UIManager(window_size)
menu_height = 30

# Define the menu structure
menu_data = {
    "File": ["New", "Open", "Save", "Exit"],
    "Edit": ["Undo", "Redo", "Preferences"],
    "Help": ["About", "Documentation"]
}

# Create an instance of the custom menubar
menubar = Menubar(
    relative_rect=pygame.Rect((0, 0), (window_size[0], menu_height)),
    manager=manager,
    menu_data=menu_data
)

clock = pygame.time.Clock()
running = True

while running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Process menubar events
        # menubar.process_event(event)

        manager.process_events(event)

    manager.update(time_delta)
    window_surface.fill((0, 0, 0))
    manager.draw_ui(window_surface)
    pygame.display.update()

pygame.quit()

