import pygame as pg
import pygame_gui as gui


import spikeyboi


class UIAgentInfo(gui.elements.UIPanel):

    # def __init__(self, rect, manager, anchors={'right': 'right', 'top':'top'}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.change_object_id('#agent_info')

        manager = self.ui_manager

        self.agent_id = gui.elements.UILabel((0,0), '', manager,
                                            container=self, parent_element=self)
        self.agent_fitness = gui.elements.UILabel((0,0), '', manager,
                                            container=self, parent_element=self,
                                            anchors={'top_target': self.agent_id})

        spikeyboi.app_instance.on_agent_selected_event.append(self.on_agent_selected)

        self.agent = spikeyboi.spikey.sim_instance.agent

        self.show()

    def update(self, delta_time):
        super().update(delta_time)

        self.agent_id.set_text(f'Agent ID:{self.agent.id}')
        self.agent_fitness.set_text(f'Fitness: {self.agent.food_consumed}')

    def on_agent_selected(self, agent):
        self.agent = agent
