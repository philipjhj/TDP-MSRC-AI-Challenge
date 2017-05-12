from __future__ import division

from collections import namedtuple

import numpy as np
from common import ENV_ACTIONS
from malmopy.agent import AStarAgent
from six.moves import range

P_FOCUSED = .75
CELL_WIDTH = 33


class DanishPuppet(AStarAgent):
    ACTIONS = ENV_ACTIONS
    Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])

    def __init__(self, name, target, visualizer=None):
        super(DanishPuppet, self).__init__(name, len(DanishPuppet.ACTIONS),
                                           visualizer=visualizer)
        self._target = str(target)
        self._previous_target_pos = None
        self._action_list = []

    def act(self, state, reward, done, is_training=False):
        if done:
            self._action_list = []
            self._previous_target_pos = None

        if state is None:
            return np.random.randint(0, self.nb_actions)

        entities = state[1]
        state = state[0]
        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        me_details = [e for e in entities if e['name'] == self.name][0]
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.
        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k]

        # Get agent and target nodes
        me = DanishPuppet.Neighbour(1, me[0][0], me[0][1], direction, "")
        target = DanishPuppet.Neighbour(1, target[0][0], target[0][1], 0, "")

        # If distance to the pig is one, just turn and wait
        if self.heuristic(me, target) == 1:
            return DanishPuppet.ACTIONS.index("turn 1")  # substitutes for a no-op command

        if not self._previous_target_pos == target:
            # Target has moved, or this is the first action of a new mission - calculate a new action list
            self._previous_target_pos = target

            path, costs = self._find_shortest_path(me, target, state=state)
            self._action_list = []
            for point in path:
                self._action_list.append(point.action)

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return DanishPuppet.ACTIONS.index(action)

        # reached end of action list - turn on the spot
        return DanishPuppet.ACTIONS.index("turn 1")  # substitutes for a no-op command

    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []
        inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z
        # add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in DanishPuppet.ACTIONS:
            if action.startswith("turn"):
                neighbors.append(
                    DanishPuppet.Neighbour(1, pos.x, pos.z, (pos.direction + int(action.split(' ')[1])) % 4, action))
            if action.startswith("move "):  # note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                weight = 1 if sign == 1 else 1.5
                neighbors.append(
                    DanishPuppet.Neighbour(weight, inc_x(pos.x, pos.direction, sign), inc_z(pos.z, pos.direction, sign),
                                           pos.direction, action))
            if action == "movenorth":
                neighbors.append(DanishPuppet.Neighbour(1, pos.x, pos.z - 1, pos.direction, action))
            elif action == "moveeast":
                neighbors.append(DanishPuppet.Neighbour(1, pos.x + 1, pos.z, pos.direction, action))
            elif action == "movesouth":
                neighbors.append(DanishPuppet.Neighbour(1, pos.x, pos.z + 1, pos.direction, action))
            elif action == "movewest":
                neighbors.append(DanishPuppet.Neighbour(1, pos.x - 1, pos.z, pos.direction, action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)

    def matches(self, a, b):
        return a.x == b.x and a.z == b.z  # don't worry about dir and action

