# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import division

import numpy as np
from six.moves import range

from agent import FocusedAgent
from bad_guy import BadGuy
from common import ENV_TARGET_NAMES, ENV_ACTIONS
from environment import PigChaseEnvironment
from malmopy.agent import BaseAgent, RandomAgent
from standstill_agent import StandstillAgent

FocusedAgentWeight = 2
RandomAgentWeight = 1
BadGuyWeight = 1
StandstillAgentWeight = 1


class ChallengerFactory(BaseAgent):
    AGENT_TYPE = {
        RandomAgent: PigChaseEnvironment.AGENT_TYPE_0,
        FocusedAgent: PigChaseEnvironment.AGENT_TYPE_1,
        BadGuy: PigChaseEnvironment.AGENT_TYPE_2,
        StandstillAgent: PigChaseEnvironment.AGENT_TYPE_3
    }

    def __init__(self, name, visualizer=None, focused=True, random=True, bad_guy=True, standstill=False):
        nb_actions = len(ENV_ACTIONS)
        super(ChallengerFactory, self).__init__(name, nb_actions,
                                                visualizer=visualizer)
        # List of possible agents
        self._agents = []
        self._agent_probabilities = []
        self._helmets = []

        print("Allowing challengers:")
        if focused:
            self._agents.append(FocusedAgent(name, ENV_TARGET_NAMES[0],
                                             visualizer=visualizer))
            self._agent_probabilities.append(FocusedAgentWeight)
            self._helmets.append(ChallengerFactory.AGENT_TYPE[FocusedAgent])
            print("   FocusedAgent")

        if random:
            self._agents.append(RandomAgent(name, nb_actions,
                                            visualizer=visualizer))
            self._agent_probabilities.append(RandomAgentWeight)
            self._helmets.append(ChallengerFactory.AGENT_TYPE[RandomAgent])
            print("   RandomAgent")

        if bad_guy:
            self._agents.append(BadGuy(name, visualizer=visualizer))
            self._agent_probabilities.append(BadGuyWeight)
            self._helmets.append(ChallengerFactory.AGENT_TYPE[BadGuy])
            print("   BadGuy")

        if standstill:
            self._agents.append(StandstillAgent(name, visualizer=visualizer))
            self._agent_probabilities.append(StandstillAgentWeight)
            self._helmets.append(ChallengerFactory.AGENT_TYPE[StandstillAgent])
            print("   StandstillAgent")

        # Select first agent
        n = sum(self._agent_probabilities)
        self._agent_probabilities = [item / n for item in self._agent_probabilities]
        self.current_agent = self._select_agent(self._agent_probabilities)

    def get_helmets(self):
        return list(self._helmets)

    def _select_agent(self, probabilties):
        return self._agents[np.random.choice(range(len(self._agents)),
                                             p=probabilties)]

    def act(self, new_state, reward, done, is_training=False):
        if done:
            self.current_agent = self._select_agent(self._agent_probabilities)
        return self.current_agent.act(new_state, reward, done, is_training)

    def save(self, out_dir):
        self.current_agent.save(out_dir)

    def load(self, out_dir):
        self.current_agent(out_dir)

    def inject_summaries(self, idx):
        self.current_agent.inject_summaries(idx)
