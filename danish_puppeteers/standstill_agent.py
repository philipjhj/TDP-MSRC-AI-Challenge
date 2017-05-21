from __future__ import division

import random

from malmopy.agent import BaseAgent
from constants import AllActions


class StandstillAgent(BaseAgent):
    ActionMap = AllActions(move=(), turn=(1, -1), strafe=(), jump=False, wait=False)

    def __init__(self, name, visualizer=None, rotate=True):
        self.rotate = rotate
        self.current_agent = self
        self.actions = [StandstillAgent.ActionMap.turn_r, StandstillAgent.ActionMap.turn_l]

        super(StandstillAgent, self).__init__(name, len(self.actions), visualizer)

    def act(self, state, reward, done, is_training=False):

        if not self.rotate:
            return random.choice(self.actions)
        else:
            return self.actions[0]
