from __future__ import division

from Queue import Queue
from collections import deque
from collections import namedtuple, OrderedDict
from heapq import heapify, heappop, heappush
from time import time
import cPickle as pickle

import numpy as np
from pathlib2 import Path
from six.moves import range

from malmopy.agent import AStarAgent

P_FOCUSED = .75
CELL_WIDTH = 33

MEMORY_SIZE = 10

DEBUG_STORE_IMAGE = True


def print_if(condition, string):
    if condition:
        print(string)


# Print settings
class Print:
    # Pre-game
    history_length = False

    # Debug
    act_reached = False
    code_line_print = False

    # In iterations
    timing = True
    iteration_line = True
    map = True
    positions = True
    steps_to_other = True
    pig_neighbours = False
    own_plans = False
    challenger_plans = False
    detailed_plans = False
    feature_vector = False
    feature_matrix = False
    expected_challenger_move = False
    challenger_strategy = True
    waiting_info = True
    repeated_waiting_info = False

    # Post game
    game_summary = True


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

    def to_neighbour(self):
        return Neighbour(x=self.x, z=self.z, direction=self.direction, action="")

    def __str__(self):
        return "{: >16s}(x={:d}, z={:d}, direction={})".format(self.name + "_Position",
                                                               self.x,
                                                               self.z,
                                                               self.direction)

    def __repr__(self):
        return str(self)


class Features:
    def __init__(self, challenger_pig_distance, own_pig_distance, challenger_exit_distance, own_exit_distance,
                 delta_challenger_pig_distance, delta_challenger_exit_distance, compliance):
        # Distances
        self.challenger_pig_distance = challenger_pig_distance
        self.own_pig_distance = own_pig_distance
        self.challenger_exit_distance = challenger_exit_distance
        self.own_exit_distance = own_exit_distance

        # Number of non-delta features
        self.n_non_delta = 4

        # Delta-distance
        self.delta_challenger_pig_distance = delta_challenger_pig_distance
        self.delta_challenger_exit_distance = delta_challenger_exit_distance

        # Compliance
        self.compliance = compliance

    def compute_deltas(self, challenger_pig_distance, challenger_exit_distance):
        delta_challenger_pig_distance = challenger_pig_distance - self.challenger_pig_distance
        delta_challenger_exit_distance = challenger_exit_distance - self.challenger_exit_distance

        # Compliance
        if self.challenger_pig_distance == 0:
            compliance = 1
        else:
            if delta_challenger_pig_distance < 0:
                compliance = 1
            else:
                compliance = 0

        return delta_challenger_pig_distance, delta_challenger_exit_distance, compliance

    def to_list(self):
        return [
            self.challenger_pig_distance,
            self.own_pig_distance,
            self.challenger_exit_distance,
            self.own_exit_distance,
            self.delta_challenger_pig_distance,
            self.delta_challenger_exit_distance,
            self.compliance
        ]

    def to_named_list(self):
        items = self.to_list()
        names = [
            "challenger_pig_distance",
            "own_pig_distance",
            "challenger_exit_distance",
            "own_exit_distance",
            "delta_challenger_pig_distance",
            "delta_challenger_exit_distance",
            "compliance",
        ]
        return list(zip(names, items))

    def to_vector(self):
        return np.array(
            self.to_list()
        )

    def __str__(self):
        string = ", ".join(["{}={}".format(name, item) for name, item in self.to_named_list()])
        return "Features({})".format(string)

    def __repr__(self):
        return str(self)

    def to_non_delta_vector(self):
        return self.to_vector()[:self.n_non_delta]


class FeatureSequence:
    def __init__(self):
        self.features = []

    def update(self, features):
        self.features.append(features)

    def to_matrix(self):
        return np.array([feature.to_vector() for feature in self.features])

    def last_features(self):
        return self.features[-1]


class GameSummary:
    def __init__(self, feature_matrix, reward, prize, final_state, pig_is_caught):
        self.feature_matrix = feature_matrix
        self.reward = reward
        self.prize = prize
        self.final_state = final_state
        self.pig_is_caught = pig_is_caught

    def __str__(self):
        return "GameSummary(reward={}, pig_is_caught={})".format(self.reward, self.pig_is_caught)

    def __repr__(self):
        return str(self)


class Plan:
    Exit = "Exit"
    PigCatch = "PigCatch"
    NoGoal = "None"

    def __init__(self, target, x, z, prize, path):
        self.target = target
        self.x = x
        self.z = z
        self.path = path

        self.utility = prize - self.plan_length()

    def __getitem__(self, item):
        return self.path[item]

    def __len__(self):
        return len(self.path)

    def plan_length(self):
        if len(self) == 1:
            return 0
        else:
            return len(self) - 1

    def path_print(self):
        if len(self) == 1:
            return "[]"
        else:
            return str(list(self.path)[1:])

    def __str__(self):
        return "Plan({}, loc=({}, {}), utility={}, moves_left={})".format(self.target,
                                                                          self.x,
                                                                          self.z,
                                                                          self.utility,
                                                                          self.plan_length())

    def __repr__(self):
        return str(self)


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

    def __repr__(self):
        return str(self)


class DanishPuppet(AStarAgent):
    class ActionMap:
        move_f = 0
        turn_l = 1
        turn_r = 2
        move_b = 3
        strafe_r = 4
        strafe_l = 5
        jump = 6
        wait = 7

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

    def __init__(self, name, visualizer=None, wait_for_pig=True):
        super(DanishPuppet, self).__init__(name, len(DanishPuppet.ACTIONS),
                                           visualizer=visualizer)
        self._previous_target_pos = None
        self._action_list = []

        # New fields
        self._entities = None
        self.manual = False
        self.waiting_for_pig = False
        self.game_features = None
        self.history_queue = Queue(maxsize=MEMORY_SIZE)
        self.first_act_call = True

        # Timing and iterations
        self.moves = -1
        self.n_games = 0
        self.n_total_moves = 0
        self.time_start = time()

        # Pig stuff
        self.previous_pig = None
        self.wait_for_pig_if_necessary = wait_for_pig
        self.pig_wait_counter = 0

    def time_alive(self):
        return time() - self.time_start

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

    @staticmethod
    def directional_steps_to_other(agent, other):
        x_diff = other.x - agent.x
        z_diff = other.z - agent.z

        if agent.direction == 0:
            forward = -z_diff
            side = x_diff
        elif agent.direction == 1:
            forward = x_diff
            side = z_diff
        elif agent.direction == 2:
            forward = z_diff
            side = -x_diff
        elif agent.direction == 3:
            forward = -x_diff
            side = -z_diff
        else:
            forward = side = None

        return forward, side

    @staticmethod
    def map_view(state):
        string_rows = []

        for row in state:
            string_row1 = []
            string_row2 = []
            for cell in row:
                if not "grass" in cell and not "lapis_block" in cell:
                    string_row1.append("XXX")
                    string_row2.append("XXX")
                else:
                    bottom_corners = "E" if "lapis_block" in cell else " "
                    string_row1.append(("A" if "Agent_2" in cell else " ") + " " +
                                       ("P" if "Pig" in cell else " "))
                    string_row2.append(bottom_corners + ("C" if "Agent_1" in cell else " ") + bottom_corners)
            string_rows.append("".join(string_row1))
            string_rows.append("".join(string_row2))

        return "\n".join(string_rows)

    def paths_to_plans(self, paths, exits, pig_neighbours):
        plans = []
        for path in paths:
            if any([self.matches(target, path[-1]) for target in exits]):
                final_position = path[-1]
                plan = Plan(target=Plan.Exit,
                            x=final_position.x,
                            z=final_position.z,
                            prize=DanishPuppet.ExitPrice - self.moves,
                            path=path)
                plans.append(plan)
            elif any([self.matches(target, path[-1]) for target in pig_neighbours]):
                final_position = path[-1]
                plan = Plan(target=Plan.PigCatch,
                            x=final_position.x,
                            z=final_position.z,
                            prize=DanishPuppet.PigCatchPrize - self.moves,
                            path=path)
                plans.append(plan)
            else:
                final_position = path[-1]
                plan = Plan(target=Plan.NoGoal,
                            x=final_position.x,
                            z=final_position.z,
                            prize=0 - self.moves,
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

    def was_pig_caught(self, prize):
        if prize > 20:
            return True
        return False

    def note_game_end(self, reward_sequence, state):
        if self.history_queue.full():
            self.history_queue.get()

        prize = int(max(reward_sequence))
        reward = int(sum(reward_sequence))

        game_summary = GameSummary(feature_matrix=self.game_features,
                                   reward=reward,
                                   prize=prize,
                                   final_state=state,
                                   pig_is_caught=self.was_pig_caught(prize=prize))

        if Print.game_summary:
            print("\nGame Summary:")
            print("   {}".format(game_summary))
            print("   From reward-sequence: {}".format(reward_sequence))

        self.history_queue.put(game_summary)

    def act(self, state, reward, done, is_training=False, frame=None):

        if Print.act_reached:
            print("DanishPuppet.act() called")

        ###############################################################################
        # Get information from environment
        print_if(Print.code_line_print, "CODE: Information parsing")

        if done:
            if self.first_act_call:

                if Print.history_length:
                    print("\nLength of history: {}".format(self.history_queue.qsize()))

                self._action_list = []
                self._previous_target_pos = None
                self.game_features = None
                self.moves = -1

                self.n_games += 1

                if Print.timing:
                    c_time = self.time_alive()
                    print("\nTiming stats:")
                    print("\tNumber of games: {}".format(self.n_games))
                    print("\tTotal time: {:.3f}s".format(c_time))
                    print("\tAverage game time: {:.3f}s".format(c_time / self.n_games))
                    if self.n_total_moves >= 0:
                        print("\tAverage move time: {:.3f}s".format(c_time / self.n_total_moves))
            self.first_act_call = False
        else:
            self.first_act_call = True

        if state is None:
            return np.random.randint(0, self.nb_actions)

        entities = state[1]
        state = state[0]
        self.moves += 1

        if Print.iteration_line and not self.waiting_for_pig:
            print("\n---------------------------\n")

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

        # Entities
        pig = self._entities['Pig']  # type: EntityPosition
        me = self._entities['Agent_2']  # type: EntityPosition
        challenger = self._entities['Agent_1']  # type: EntityPosition

        if Print.map and not self.waiting_for_pig:
            print(self.map_view(state))

        if Print.positions and not self.waiting_for_pig:
            for item in self._entities.values():
                print(item)

        if Print.steps_to_other:
            print("Steps to challenger: {}".format(self.directional_steps_to_other(me, challenger)))

        if DEBUG_STORE_IMAGE:
            storage_path = Path("..", "data_dumps")
            files_in_directory = [str(item.stem) for item in storage_path.glob("*.p")]
            try:
                next_file_number = max([int(item.replace("file_", ""))
                                        for item in files_in_directory]) + 1
            except ValueError:
                next_file_number = 0

            file = ((me, challenger, pig), self.directional_steps_to_other(me, challenger), state, frame)

            pickle.dump(file, Path(storage_path, "file_{}.p".format(next_file_number)).open("wb"))

        ###############################################################################
        # Determine possible targets
        print_if(Print.code_line_print, "CODE: Determining targets")

        # Exist positions
        exits = [Neighbour(x=1, z=4, direction=0, action=""), Neighbour(x=7, z=4, direction=0, action="")]

        # Get pig position
        pig_node = DanishPuppet.Position(self._entities["Pig"].x, self._entities["Pig"].z)

        # Get neighbours
        pig_neighbours = []
        neighbour_cells = []
        for x_diff, z_diff in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_position = (pig_node.x + x_diff, pig_node.z + z_diff)
            if "grass" in state[new_position[1], new_position[0]]:
                pig_neighbours.append(DanishPuppet.Position(*new_position))
                neighbour_cells.append(state[new_position[0], new_position[1]])

        # All target positions
        targets = pig_neighbours + exits

        if Print.pig_neighbours:
            print("\nPig neighbours:")
            for neighbour, cell in zip(pig_neighbours, neighbour_cells):
                print("   {}: {}".format(neighbour, cell))

        ###############################################################################
        # Compute possible plans for each player plans
        print_if(Print.code_line_print, "CODE: Computing plans")

        # Find own paths
        own_paths = self._astar_multi_search(start=me.to_neighbour(),
                                             goals=targets,
                                             state=state)[0]
        own_plans = self.paths_to_plans(own_paths, exits, pig_neighbours)

        # Find challengers paths
        challengers_paths = self._astar_multi_search(start=challenger.to_neighbour(),
                                                     goals=targets,
                                                     state=state,
                                                     must_turn=True)[0]
        challengers_plans = self.paths_to_plans(challengers_paths, exits, pig_neighbours)

        if Print.own_plans:
            print("\nOwn {} plans:".format(len(own_paths)))
            for plan in own_plans:
                print("   {}".format(plan))
                if Print.detailed_plans:
                    print("      {}".format(plan.path_print()))

        if Print.challenger_plans:
            print("\nChallenger {} plans:".format(len(challengers_paths)))
            for plan in challengers_plans:
                print("   {}".format(plan))
                if Print.detailed_plans:
                    print("      {}".format(plan.path_print()))
        del own_paths
        del challengers_paths

        # Pig plans for both agents
        own_pig_plans = [plan for plan in own_plans if plan.target == Plan.PigCatch]
        challenger_pig_plans = [plan for plan in challengers_plans if plan.target == Plan.PigCatch]

        ###############################################################################
        # If pig is not in a useful place - wait
        print_if(Print.code_line_print, "CODE: Considering pig-wait")

        if len(pig_neighbours) > 2 and self.wait_for_pig_if_necessary:
            if Print.repeated_waiting_info or not self.waiting_for_pig:
                if Print.waiting_info:
                    print("\nWaiting for pig at {}...".format(pig_node))
                    if Print.map:
                        print(self.map_view(state))
                self.waiting_for_pig = True
            self.pig_wait_counter += 1
            return DanishPuppet.ACTIONS.index("wait")
        self.waiting_for_pig = False
        self.pig_wait_counter = 0

        ###############################################################################
        # Feature Extraction
        print_if(Print.code_line_print, "CODE: Extracting features")

        # Pig distances
        own_pig_distance = min([plan.plan_length() for plan in own_plans
                                if plan.target == Plan.PigCatch])
        challenger_pig_distance = min([plan.plan_length() for plan in challengers_plans
                                       if plan.target == Plan.PigCatch])
        challenger_pig_plan = next(plan for plan in challengers_plans
                                   if plan.target == Plan.PigCatch and plan.plan_length() == challenger_pig_distance)

        if Print.expected_challenger_move:
            if challenger_pig_plan.plan_length() > 0:
                print("Next expected move: {}".format(challenger_pig_plan[1].action))
            else:
                print("Next expected move: None")

        # Exit distances
        own_exit_distance = min([plan.plan_length() for plan in own_plans
                                 if plan.target == Plan.Exit])
        challenger_exit_distance = min([plan.plan_length() for plan in challengers_plans
                                        if plan.target == Plan.Exit])

        # Check if first iteration in game
        if self.game_features is None:
            self.game_features = FeatureSequence()
            features = Features(challenger_pig_distance=challenger_pig_distance,
                                own_pig_distance=own_pig_distance,
                                challenger_exit_distance=challenger_exit_distance,
                                own_exit_distance=own_exit_distance,
                                delta_challenger_pig_distance=None,
                                delta_challenger_exit_distance=None,
                                compliance=1)
            self.game_features.update(features)

        # Otherwise compute deltas
        else:
            # If pig has moved, then accept challenger to be moving towards either old or new location
            if not self.matches(pig, self.previous_pig):
                previous_challenger_pig_distance = challenger_pig_distance
                plans_to_new_pig_position = self._astar_multi_search(start=challenger.to_neighbour(),
                                                                     goals=[self.previous_pig],
                                                                     state=state,
                                                                     must_turn=True)
                alternative_plans = self.paths_to_plans(plans_to_new_pig_position, exits, pig_neighbours)

                challenger_pig_distance = min(challenger_pig_distance,
                                              *[plan.plan_length() for plan in alternative_plans])

                print("Pig moved. Challenger distance goes from {} to {}".format(previous_challenger_pig_distance,
                                                                                 challenger_pig_distance))

            # Get last features and compute deltas
            last_features = self.game_features.last_features()  # type: Features
            deltas = last_features.compute_deltas(challenger_pig_distance=challenger_pig_distance,
                                                  challenger_exit_distance=challenger_exit_distance)
            delta_challenger_pig_distance, delta_challenger_exit_distance, compliance = deltas

            # Make new features
            features = Features(challenger_pig_distance=challenger_pig_distance,
                                own_pig_distance=own_pig_distance,
                                challenger_exit_distance=challenger_exit_distance,
                                own_exit_distance=own_exit_distance,
                                delta_challenger_pig_distance=delta_challenger_pig_distance,
                                delta_challenger_exit_distance=delta_challenger_exit_distance,
                                compliance=compliance)

            # Add features
            self.game_features.update(features)

        if Print.feature_vector:
            print("\nFeature vector:")
            print("   {}".format(features))

        if Print.feature_matrix:
            print("\nFeature matrix:")
            print(self.game_features.to_matrix())

        ###############################################################################
        # Update memory of last iteration

        self.previous_pig = pig

        ###############################################################################
        # This is a move

        self.n_total_moves += 1

        ###############################################################################
        # Manual overwrite
        print_if(Print.code_line_print, "CODE: Considering manual overwrite")

        # Check if manual control is wanted
        if self.manual:
            while True:
                choice = raw_input("\nType action (AWSD + QE): ")
                if choice in DanishPuppet.KeysMapping:
                    print("   Choice {} translated to {}".format(choice, DanishPuppet.KeysMapping[choice]))
                    return DanishPuppet.KeysMapping[choice]

        ###############################################################################
        # Determine challengers strategy
        print_if(Print.code_line_print, "CODE: Determining challenger strategy")

        # Strategies are:
        # 0: Initial round (no information)
        # 1: Random-walk Idiot
        # 2: Naive Cooperative
        # 3: Optimal Cooperative
        # 4: Douche
        # 5: Pig can be caught by one person

        # Data on past
        matrix = self.game_features.to_matrix()
        compliances = matrix[:, 6]

        # Check if pig can be caught alone
        if len(own_pig_plans) == 1:
            challenger_strategy = 5

        # Base predicted strategy on compliance of challenger
        elif compliances.mean() < 0.5:
            challenger_strategy = 1
        else:
            challenger_strategy = 2

        ###############################################################################
        # Determine plan based on challengers strategy
        print_if(Print.code_line_print, "CODE: Selecting plan based on strategy")

        # If challenger is an idiot or a douche - backstab him!
        if challenger_strategy in {1, 4}:
            if Print.challenger_strategy:
                print("\nChallenger seems to be an idiot.")

            # Exit plan
            plan = sorted([plan for plan in own_plans if plan.target == Plan.Exit],
                          key=lambda plan: -plan.utility)[0]

            # Return next action (0th element is current position)
            action = plan[1].action
            return DanishPuppet.ACTIONS.index(action)

        # If challenger is naive cooperative - go to the pig on the side farthest from him!
        elif challenger_strategy in {0, 2, 3, 5}:
            if Print.challenger_strategy:
                if challenger_strategy == 0:
                    print("\nFirst round. Challenger is assumed compliant.")
                elif challenger_strategy == 5:
                    print("\nCatching pig on my own!")
                else:
                    print("\nChallenger seems compliant.")

            # Check if pig can be caught by one person - then catch it!
            if challenger_strategy == 5:
                own_pig_plan = own_pig_plans[0]
                action = own_pig_plan[1].action
                return DanishPuppet.ACTIONS.index(action)

            # Find shortest plan from challenger
            challenger_pig_plan = sorted(challenger_pig_plans, key=lambda plan: -plan.utility)[0]

            # Find remaining spot for own agent
            own_pig_plan = next((plan for plan in own_pig_plans
                                 if not self.matches(plan, challenger_pig_plan)))

            # Check if already at pig
            if len(own_pig_plan) < 2:
                if Print.waiting_info:
                    print("\nWaiting for challenger to help with pig ...")
                return self.directional_wait_action(entity=self._entities["Agent_2"],
                                                    other_position=self._entities["Agent_1"])

            # Return next action (0th element is current position)
            action = own_pig_plan[1].action
            return DanishPuppet.ACTIONS.index(action)

    def neighbors(self, pos, must_turn=False, state=None):
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
            if must_turn:
                if action.startswith("turn"):
                    new_direction = (pos.direction + int(action.split(' ')[1])) % 4
                    new_state = Neighbour(x=pos.x,
                                          z=pos.z,
                                          direction=new_direction,
                                          action=action)
                    neighbors.append(new_state)

            # Strafing actions
            else:
                if action.startswith("strafe "):
                    sign = int(action.split(' ')[1])
                    neighbors.append(
                        Neighbour(x=strafe_inc_x(pos.x, pos.direction, sign),
                                  z=strafe_inc_z(pos.z, pos.direction, sign),
                                  direction=pos.direction,
                                  action=action))

            # Moving actions
            if action.startswith("move "):  # Note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                neighbors.append(
                    Neighbour(x=move_inc_x(pos.x, pos.direction, sign),
                              z=move_inc_z(pos.z, pos.direction, sign),
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

    def _astar_multi_search(self, start, goals, state, must_turn=False, **kwargs):
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
            current_priority, current = heappop(explorer)

            # Check if any goal is reached
            for idx, goal in enumerate(goals_remaining):
                if self.matches(current, goal):
                    goal_nodes.append(current)
                    del goals_remaining[idx]

            if not goals_remaining:
                break

            # Go through neighbours
            for nb in self.neighbors(current, must_turn=must_turn, state=state):
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
