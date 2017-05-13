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
    Exit = "Exit"
    PigCatch = "PigCatch"

    def __init__(self, target, x, z, utility, path):
        self.target = target
        self.x = x
        self.z = z
        self.utility = utility
        self.path = path

    def __getitem__(self, item):
        return self.path[item]

    def __len__(self):
        return len(self.path)

    def __str__(self):
        return "Plan({}, ({}, {}), {})".format(self.target,
                                               self.x,
                                               self.z,
                                               self.utility)


class Neighbour:
    def __init__(self, x, z, direction, action):
        self.x = x
        self.z = z
        self.direction = direction
        self.action = action

    def __field_list(self):
        return [self.x,
                self.z,
                self.direction,
                self.action]

    def __getitem__(self, item):
        return self.__field_list()[item]

    def __len__(self):
        return 4

    def __str__(self):
        return "Neighbour{}".format(tuple(self.__field_list()))


class DanishPuppet(AStarAgent):
    ACTIONS = ["move 1",  # 0
               "turn -1",  # 1
               "turn 1",  # 2
               'move -1',  # 3
               "strafe 1",  # 4
               "strafe -1",  # 5
               "jump 1",  # 6
               'wait'  # 7
               ]
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
                   ' ': ACTIONS.index('jump 1'),
                   # No-op
                   "n": ACTIONS.index("wait"),
                   "N": ACTIONS.index("wait"),
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
        self.waiting_for_pig = False

    @staticmethod
    def parse_positions(state):
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
                plan = Plan(target=Plan.Exit,
                            x=final_position.x,
                            z=final_position.z,
                            utility=DanishPuppet.ExitPrice - len(path) + offset,
                            path=path)
                plans.append(plan)
            if any([self.matches(target, path[-1]) for target in pig_neighbours]):
                final_position = path[-1]
                plan = Plan(target=Plan.PigCatch,
                            x=final_position.x,
                            z=final_position.z,
                            utility=DanishPuppet.PigCatchPrize - len(path) + offset,
                            path=path)
                plans.append(plan)
        plans = sorted(plans, key=lambda plan: -plan.utility)
        return plans

    @staticmethod
    def direction_towards_position(own_position, other_position):
        x_diff = other_position.x - own_position.x
        z_diff = other_position.z - own_position.z

        if abs(x_diff) > abs(z_diff):
            if x_diff < 0:
                return 4
            else:
                return 1
        else:
            if z_diff < 0:
                return 0
            else:
                return 3

    def directional_wait_action(self, entity, other_position):
        # Determine direction
        direction = self.direction_towards_position(entity, other_position)

        # Determine action
        direction_diff = direction - entity.direction
        while direction_diff < -2:
            direction_diff += 4
        while direction_diff > 2:
            direction_diff -= 4
        if direction_diff < 0:
            return DanishPuppet.ACTIONS.index('turn -1')
        elif direction_diff > 0:
            return DanishPuppet.ACTIONS.index('turn 1')
        else:
            return DanishPuppet.ACTIONS.index('jump 1')

    def act(self, state, reward, done, is_training=False):

        # TODO: Strafe and possibly the ability to move backwards is not nessesarily in the ASTAR and BFS algorithms
        #   TODO: It all depends on the neighbors-method

        ###############################################################################
        # Get information from environment

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

        # Get Details of our agent
        me = [(col_nr, row_nr)
              for row_nr, row in enumerate(state)
              for col_nr, cell in enumerate(row)
              if self.name in cell]
        me_details = [e for e in entities if e['name'] == self.name][0]

        # Convert angle to direction
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

        ###############################################################################
        # Compute possible plans for each player plans

        # Exist positions
        exits = [Neighbour(x=1, z=4, direction=0, action=""), Neighbour(x=7, z=4, direction=0, action="")]

        # Get pig position
        pig_node = DanishPuppet.Position(self._entities["Pig"].x, self._entities["Pig"].z)

        # Get neighbours
        pig_neighbours = []
        for x_diff, z_diff in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_position = (pig_node.x + x_diff, pig_node.z + z_diff)
            if "grass" in state[new_position[0], new_position[1]]:
                pig_neighbours.append(DanishPuppet.Position(*new_position))

        # All target positions
        targets = pig_neighbours + exits + \
                  [Neighbour(x=self._entities["Agent_2"].x,
                             z=self._entities["Agent_2"].z,
                             direction=direction,
                             action="")]

        # Find own paths
        start = Neighbour(x=self._entities["Agent_2"].x,
                          z=self._entities["Agent_2"].z,
                          direction=direction,
                          action="")
        own_paths = self._astar_multi_search(start=start,
                                             goals=targets,
                                             state=state)[0]
        own_plans = self.paths_to_plans(own_paths, exits, pig_neighbours)

        # Find challengers paths
        start = Neighbour(x=self._entities["Agent_1"].x,
                          z=self._entities["Agent_1"].z,
                          direction=direction,
                          action="")
        challengers_paths = self._astar_multi_search(start=start,
                                                     goals=targets,
                                                     state=state)[0]
        challengers_plans = self.paths_to_plans(challengers_paths, exits, pig_neighbours)

        ###############################################################################
        # If pig is not in a useful place - wait

        if len(pig_neighbours) > 2:
            if not self.waiting_for_pig:
                print("Waiting for pig ...")
                self.waiting_for_pig = True
            return DanishPuppet.ACTIONS.index("wait")
        self.waiting_for_pig = False

        ###############################################################################
        # Prints

        print("\n---------------------------\n")
        for item in self._entities.values():
            print(item)
        print("")

        print("Pig neighbours:")
        for neighbour in pig_neighbours:
            print("   {}".format(neighbour))
        print("")

        print("Own {} plans:".format(len(own_paths)))
        for plan in own_plans:
            print("   {}".format(plan))
        print("Challenger {} plans:".format(len(challengers_paths)))
        for plan in challengers_plans:
            print("   {}".format(plan))
        del own_paths
        del challengers_paths

        ###############################################################################
        # Manual overwrite

        # Check if manual control is wanted
        if self.manual:
            while True:
                choice = raw_input("\nType action (AWSD + QE): ")
                if choice in DanishPuppet.KeysMapping:
                    print("   Choice {} translated to {}".format(choice, DanishPuppet.KeysMapping[choice]))
                    return DanishPuppet.KeysMapping[choice]

        ###############################################################################
        # Pig is catchable

        # Strategies are:
        # 1: Random-walk idiot
        # 2: Naive Cooperative
        # 3: Optimal cooperative
        # 4: Douche
        challenger_strategy = 2

        # If challenger is an idiot or a douche - backstab him!
        if challenger_strategy == 1 or challenger_strategy == 4:
            # Exit plan
            plan = sorted([plan for plan in own_plans if plan.target == Plan.Exit],
                          key=lambda plan: -plan.utility)[0]

            # Return next action (0th element is current position)
            action = plan[1].action
            return DanishPuppet.ACTIONS.index(action)

        # If challenger is naive cooperative - go to the pig on the side farthest from him!
        elif challenger_strategy == 2:
            # Pig plans for both agents
            own_pig_plans = [plan for plan in own_plans if plan.target == Plan.PigCatch]
            challenger_pig_plans = [plan for plan in challengers_plans if plan.target == Plan.PigCatch]

            # Find shortest plan from challenger
            challenger_pig_plan = sorted(challenger_pig_plans, key=lambda plan: -plan.utility)[0]

            # Find remaining spot for own agent
            own_pig_plan = next((plan for plan in own_pig_plans
                                 if not self.matches(plan, challenger_pig_plan)))

            # Check if already at pig
            if len(own_pig_plan) < 2:
                print("Waiting for challenger to help with pig ...")
                return self.directional_wait_action(entity=self._entities["Agent_2"],
                                                    other_position=self._entities["Agent_1"])

            # Return next action (0th element is current position)
            action = own_pig_plan[1].action
            return DanishPuppet.ACTIONS.index(action)

        ###############################################################################
        # Original code

        target = [(col_nr, row_nr)
                  for row_nr, row in enumerate(state)
                  for col_nr, cell in enumerate(row)
                  if self._target in cell]

        # Get agent and target nodes
        me = Neighbour(x=me[0][0],
                       z=me[0][1],
                       direction=direction,
                       action="")
        target = Neighbour(x=target[0][0],
                           z=target[0][1],
                           direction=0,
                           action="")

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
        # State information
        state_width = state.shape[1]
        state_height = state.shape[0]

        # Direction integers
        dir_north, dir_east, dir_south, dir_west = range(4)

        # Initialize list of neighbours
        neighbors = []

        # Functions for incrementing position based on direction and sign of move
        move_inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        move_inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z

        # Functions for incrementing position based on direction and sign of strafe
        strafe_inc_x = lambda x, dir, delta: x + delta if dir == dir_north else x - delta if dir == dir_south else x
        strafe_inc_z = lambda z, dir, delta: z + delta if dir == dir_east else z - delta if dir == dir_west else z

        # Add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in DanishPuppet.ACTIONS:

            # Turning actions
            if action.startswith("turn"):
                new_direction = (pos.direction + int(action.split(' ')[1])) % 4
                new_state = Neighbour(x=pos.x,
                                      z=pos.z,
                                      direction=new_direction,
                                      action=action)
                neighbors.append(new_state)

            # Moving actions
            if action.startswith("move "):  # Note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                neighbors.append(
                    Neighbour(x=move_inc_x(pos.x, pos.direction, sign),
                              z=move_inc_z(pos.z, pos.direction, sign),
                              direction=pos.direction,
                              action=action))

            # Strafing actions
            if action.startswith("strafe "):
                sign = int(action.split(' ')[1])
                neighbors.append(
                    Neighbour(x=strafe_inc_x(pos.x, pos.direction, sign),
                              z=strafe_inc_z(pos.z, pos.direction, sign),
                              direction=pos.direction,
                              action=action))

            # Abstract actions
            if action == "movenorth":
                neighbors.append(Neighbour(x=pos.x,
                                           z=pos.z - 1,
                                           direction=pos.direction,
                                           action=action))
            elif action == "moveeast":
                neighbors.append(Neighbour(x=pos.x + 1,
                                           z=pos.z,
                                           direction=pos.direction,
                                           action=action))
            elif action == "movesouth":
                neighbors.append(Neighbour(x=pos.x,
                                           z=pos.z + 1,
                                           direction=pos.direction,
                                           action=action))
            elif action == "movewest":
                neighbors.append(Neighbour(x=pos.x - 1,
                                           z=pos.z,
                                           direction=pos.direction,
                                           action=action))

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
