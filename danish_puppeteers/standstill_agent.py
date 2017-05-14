from __future__ import division

import random

from danish_puppet import DanishPuppet
from malmopy.agent import BaseAgent


class StandstillAgent(BaseAgent):
    ActionMap = DanishPuppet.ActionMap

    def __init__(self, name, visualizer=None, turn=True):
        self.turn = turn
        self.current_agent = self
        self.actions = []

        if self.turn:
            self.actions += [StandstillAgent.ActionMap.turn_r, StandstillAgent.ActionMap.turn_l]

        super(StandstillAgent, self).__init__(name, len(self.actions), visualizer)

    def act(self, state, reward, done, is_training=False):

        return random.choice(self.actions)
