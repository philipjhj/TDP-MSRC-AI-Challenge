from __future__ import division

from collections import namedtuple, OrderedDict

import numpy as np

from danish_puppet import DanishPuppet
from utility.ai import EntityPosition, Neighbour


class BadGuy(DanishPuppet):
    class ActionMap:
        move_f = 0
        turn_l = 1
        turn_r = 2

    ACTIONS = ["move 1",  # 0
               "turn -1",  # 1
               "turn 1",  # 2
               ]
    Position = namedtuple("Position", "x, z")

    def __init__(self, name, visualizer=None, wait_for_pig=True):
        super(BadGuy, self).__init__(name, visualizer, wait_for_pig)

    def act(self, state, reward, done, is_training=False, frame=None):

        if state is None:
            return np.random.randint(0, self.nb_actions)

        entities = state[1]
        state = state[0]

        # Parse positions from grid
        positions = self.parse_positions(state)
        for idx, entity in enumerate(entities):
            entity['x'] = positions[entity['name']][0]
            entity['z'] = positions[entity['name']][1]

        # Initialize entities information
        if self._entities is None:
            self._entities = OrderedDict((name, None) for name in BadGuy.EntityNames)
            for item in entities:
                self._entities[item['name']] = EntityPosition(name=item['name'], yaw=item['yaw'],
                                                              x=item['x'], z=item['z'])
        else:
            for item in entities:
                # print(item)
                self._entities[item['name']].update(name=item['name'], yaw=item['yaw'],
                                                    x=item['x'], z=item['z'])

        # Entities
        challenger = self._entities['Agent_1']  # type: EntityPosition

        ###############################################################################
        # Determine possible targets

        # Target positions
        targets = [Neighbour(x=1, z=4, direction=0, action=""), Neighbour(x=7, z=4, direction=0, action="")]

        ###############################################################################
        # Compute possible plans for each player plans

        # Find own paths
        own_paths = self._astar_multi_search(start=challenger.to_neighbour(),
                                             goals=targets,
                                             state=state,
                                             actions=BadGuy.ACTIONS)[0]

        # Find shortest path to an exit
        the_plan = own_paths[np.argmin([len(path) for path in own_paths])]
        return BadGuy.ACTIONS.index(the_plan[1].action)
