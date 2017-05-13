from __future__ import division

from collections import deque
from collections import namedtuple, OrderedDict
from itertools import product
from heapq import heapify, heappop, heappush

import numpy as np
from six.moves import range

from malmopy.agent import AStarAgent

P_FOCUSED = .75
CELL_WIDTH = 33


class EntityPosition:
    def __init__(self, unparsed_info):
        """
        Parses information from the Minecraft API to an object, which is easy to work with. 
        :param dict unparsed_info: 
        """

        self.name = unparsed_info['name']
        self.direction = self.find_direction(unparsed_info['yaw'])
        self.z = unparsed_info['z']
        self.x = unparsed_info['x']

    def _angle_float(self, angle):
        return ((angle % 360) // 90) - 1

    def find_direction(self, angle):
        angle = self._angle_float(angle - 45)
        return int(angle % 4)

    def update(self, unparsed_info):
        self.name = unparsed_info['name']
        self.z = unparsed_info['z']
        self.x = unparsed_info['x']

        # Ensure consistency in angles
        self.direction = self.find_direction(unparsed_info['yaw'])

    def __str__(self):
        return "{: >16s}(x={:d}, z={:d}, direction={})".format(self.name + "_Position",
                                                               self.x,
                                                               self.z,
                                                               self.direction)


class Plan:
    def __init__(self, target, target_position, utility, path):
        self.target = target
        self.target_position = target_position
        self.utility = utility
        self.path = path

    def __str__(self):
        return "Plan({}, ({}, {}), {})".format(self.target,
                                               self.target_position[0],
                                               self.target_position[1],
                                               self.utility)


class DanishPuppet(AStarAgent):
    ACTIONS = ["move 1",  # 0
               "turn -1",  # 1
               "turn 1",  # 2
               'move -1',  # 3
               "strafe 1",  # 4
               "strafe -1",  # 5
               'jump 1'  # 6 (wait)
               ]
    Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])
    Position = namedtuple("Position", "x, z")
    KeysMapping = {'L': ACTIONS.index('turn -1'),
                   'l': ACTIONS.index('turn -1'),
                   'R': ACTIONS.index('turn 1'),
                   'r': ACTIONS.index('turn 1'),
                   'U': ACTIONS.index('move 1'),
                   'u': ACTIONS.index('move 1'),
                   'F': ACTIONS.index('move 1'),
                   'f': ACTIONS.index('move 1'),
                   'B': ACTIONS.index('move -1'),
                   'b': ACTIONS.index('move -1'),
                   # AWSD + QE
                   "a": ACTIONS.index("strafe -1"),
                   "A": ACTIONS.index("strafe -1"),
                   "w": ACTIONS.index("move 1"),
                   "W": ACTIONS.index("move 1"),
                   "s": ACTIONS.index("move -1"),
                   "S": ACTIONS.index("move -1"),
                   "d": ACTIONS.index("strafe 1"),
                   "D": ACTIONS.index("strafe 1"),
                   'q': ACTIONS.index('turn -1'),
                   'Q': ACTIONS.index('turn -1'),
                   'e': ACTIONS.index('turn 1'),
                   'E': ACTIONS.index('turn 1'),
                   # Jump
                   ' ': ACTIONS.index('jump 1')
                   }
    EntityNames = ["Agent_1", "Agent_2", "Pig"]
    PigCatchPrize = 25
    ExitPrice = 5

    def __init__(self, name, target, visualizer=None):
        super(DanishPuppet, self).__init__(name, len(DanishPuppet.ACTIONS),
                                           visualizer=visualizer)
        self._target = str(target)
        self._previous_target_pos = None
        self._action_list = []

        # New fields
        self._entities = None
        self.manual = False

    def parse_positions(self, state):
        entity_positions = dict()

        # Go through rows and columns
        for row_nr, row in enumerate(state):
            for col_nr, cell in enumerate(row):

                # Go through entities that still have not been found
                for entity_nr, entity in enumerate(DanishPuppet.EntityNames):

                    # Check if found
                    if entity in cell:
                        entity_positions[entity] = (abs(col_nr), abs(row_nr))

                    # Check if all found
                    if len(entity_positions) == 3:
                        return entity_positions

        return entity_positions

    def paths_to_plans(self, paths, exits, pig_neighbours):
        plans = []
        for path in paths:
            offset = 1 if len(path) == 1 else 2
            if any([self.matches(target, path[-1]) for target in exits]):
                final_position = path[-1]
                plan = Plan(target="Exit",
                            target_position=(final_position.x, final_position.z),
                            utility=DanishPuppet.ExitPrice - len(path) + offset,
                            path=path)
                plans.append(plan)
            if any([self.matches(target, path[-1]) for target in pig_neighbours]):
                final_position = path[-1]
                plan = Plan(target="PigCatch",
                            target_position=(final_position.x, final_position.z),
                            utility=DanishPuppet.PigCatchPrize - len(path) + offset,
                            path=path)
                plans.append(plan)
        plans = sorted(plans, key=lambda plan: -plan.utility)
        return plans

    def act(self, state, reward, done, is_training=False):

        # TODO: Strafe and possibly the ability to move backwards is not nessesarily in the ASTAR and BFS algorithms
        #   TODO: It all depends on the neighbors-method

        if done:
            self._action_list = []
            self._previous_target_pos = None

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
            self._entities = OrderedDict((name, None) for name in DanishPuppet.EntityNames)
            for item in entities:
                self._entities[item['name']] = EntityPosition(item)
        else:
            for item in entities:
                # print(item)
                self._entities[item['name']].update(item)

        print("")
        for item in self._entities.values():
            print(item)
        print("")

        # Get Details of our agent
        me = [(col_nr, row_nr)
              for row_nr, row in enumerate(state)
              for col_nr, cell in enumerate(row)
              if self.name in cell]
        me_details = [e for e in entities if e['name'] == self.name][0]

        # Convert angle to direction
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

        #####
        # Compute possible plans for each player plans

        # Exist positions
        exits = [DanishPuppet.Neighbour(1, 1, 4, 0, ""), DanishPuppet.Neighbour(1, 7, 4, 0, "")]

        # Get pig position
        pig_node = DanishPuppet.Position(self._entities["Pig"].x, self._entities["Pig"].z)

        # Get neighbours
        pig_neighbours = []
        for x_diff, z_diff in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_position = (pig_node.x + x_diff, pig_node.z + z_diff)
            if "grass" in state[new_position[0], new_position[1]]:
                pig_neighbours.append(DanishPuppet.Position(*new_position))

        print("Pig neighbours:")
        for neighbour in pig_neighbours:
            print("   {}".format(neighbour))
        print("")

        # All target positions
        targets = pig_neighbours + exits + \
                  [DanishPuppet.Neighbour(1, self._entities["Agent_2"].x, self._entities["Agent_2"].z, direction, "")]

        # Find own paths
        start = DanishPuppet.Neighbour(1, self._entities["Agent_2"].x, self._entities["Agent_2"].z, direction, "")
        own_paths = self._astar_multi_search(start=start,
                                             goals=targets,
                                             state=state)[0]

        # Find challengers paths
        start = DanishPuppet.Neighbour(1, self._entities["Agent_1"].x, self._entities["Agent_1"].z, direction, "")
        challengers_paths = self._astar_multi_search(start=start,
                                                     goals=targets,
                                                     state=state)[0]

        print("Own {} plans:".format(len(own_paths)))
        for plan in self.paths_to_plans(own_paths, exits, pig_neighbours):
            print("   {}".format(plan))
        print("Challenger {} plans:".format(len(challengers_paths)))
        for plan in self.paths_to_plans(challengers_paths, exits, pig_neighbours):
            print("   {}".format(plan))

        target = [(col_nr, row_nr)
                  for row_nr, row in enumerate(state)
                  for col_nr, cell in enumerate(row)
                  if self._target in cell]

        # Get agent and target nodes
        me = DanishPuppet.Neighbour(1, me[0][0], me[0][1], direction, "")
        target = DanishPuppet.Neighbour(1, target[0][0], target[0][1], 0, "")

        # Check if manual control is wanted
        if self.manual:
            while True:
                choice = raw_input("\nType action (AWSD + QE): ")
                if choice in DanishPuppet.KeysMapping:
                    print("   Choice {} translated to {}".format(choice, DanishPuppet.KeysMapping[choice]))
                    return DanishPuppet.KeysMapping[choice]

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

    def _astar_multi_search(self, start, goals, **kwargs):
        """
        Searches the entire graph for the shortest path from one location to any of a list of goals. 
        :param start: 
        :param list goals: 
        :param kwargs: 
        :return: 
        """
        # Ensure uniqueness of goal-positions
        goals_remaining = list(set(DanishPuppet.Position(item.x, item.z) for item in goals))

        # Previous node, cost so far and goal-nodes
        came_from, cost_so_far = {}, {}
        goal_nodes = []

        # Priority queue for holding future nodes
        explorer = []
        heapify(explorer)

        # Initialize start-state
        heappush(explorer, (0, start))
        came_from[start] = None
        cost_so_far[start] = 0
        current = None

        # Keep fetching nodes from queue
        while len(explorer) > 0:
            _, current = heappop(explorer)

            # Check if any goal is reached
            for idx, goal in enumerate(goals_remaining):
                if self.matches(current, goal):
                    goal_nodes.append(current)
                    del goals_remaining[idx]

            if not goals_remaining:
                break

            # Go through neighbours
            for nb in self.neighbors(current, **kwargs):
                # Compute new cost
                cost = nb.cost if hasattr(nb, "cost") else 1
                new_cost = cost_so_far[current] + cost

                # Check if node has never been seen before or if cost is lower than
                # previously found path (shouldn't happen)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost

                    # Find minimum heuristic
                    heuristic = min([self.heuristic(goal, nb, **kwargs) for goal in goals_remaining])

                    # Add to priority queue
                    priority = new_cost + heuristic
                    heappush(explorer, (priority, nb))
                    came_from[nb] = current

        # Build paths
        paths = []
        for node in goal_nodes:
            path = deque()
            while node is not start:
                path.appendleft(node)
                node = came_from[node]
            path.appendleft(start)
            paths.append(path)

        return paths, cost_so_far
