import pygame as pg
import pygame_gui as gui


UI_LOAD_FILE_EVENT = pg.event.custom_type()
UI_SAVE_FILE_EVENT = pg.event.custom_type()


class UIFileDialog(gui.windows.UIFileDialog):
    def __init__(self, rect, manager, ok_callback, *args, method='load', **kwargs):
        rect = pg.Rect(rect)
        super().__init__(rect, manager, *args, **kwargs)

        self.dialog_type = method.lower()
        self.ok_callback = ok_callback

        if method == 'load':
            self.set_display_title('Load File')
            self.allow_existing_files_only = True

        elif method == 'save':
            self.set_display_title('Save File')

    def _process_ok_cancel_events(self, e):
        super()._process_ok_cancel_events(e)

        # if e.type == gui.UI_FILE_DIALOG_PATH_PICKED and e.ui_element == self:
        if e.type == gui.UI_BUTTON_PRESSED and e.ui_element == self.ok_button:
            # m = {
            #     's': UI_SAVE_FILE_EVENT,
            #     'l': UI_LOAD_FILE_EVENT,
            # }

            # c = self.dialog_type.lower()[0]
            # event_type = m[c]

            # res = gui.core.utility.create_resource_path(self.current_file_path)
            # event_data = {
            #     'file_path': res,
            #     'ui_element': self,
            #     'ui_object_id': self.most_specific_combined_id,

            # }
            # pg.event.post(pg.event.Event(event_type, event_data))
            res = gui.core.utility.create_resource_path(self.current_file_path)

            self.ok_callback(res)
