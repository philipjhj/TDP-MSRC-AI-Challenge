from collections import OrderedDict

from ai import EntityPosition, Neighbour, Location, GamePlanner
from constants import ENTITY_NAMES, AllActions


def map_view(state):
    """
    Returns a print-friendly view of the state of the game. 
    :param list state: 
    :return: str
    """
    string_rows = []

    for row in state:
        string_row1 = []
        string_row2 = []
        for cell in row:
            if "grass" not in cell and "lapis_block" not in cell:
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


class GameTimer:
    TOTAL_TIME_FACTOR = 1. / 20.
    TOTAL_GAME_TIME = 1000.

    def __init__(self):
        self._start_time = None
        self.time_left = None

    def reset(self):
        self._start_time = None
        self.time_left = None

    def update(self, total_time):
        if self._start_time is None:
            self._start_time = total_time
            self.time_left = GameTimer.TOTAL_GAME_TIME
        else:
            self.time_left = GameTimer.TOTAL_GAME_TIME - \
                             (total_time - self._start_time) * GameTimer.TOTAL_TIME_FACTOR


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


class GameObserver:
    def __init__(self, helmets):
        self._entities = None

    def reset(self):
        self._entities = None

    @staticmethod
    def parse_positions(state):
        entity_positions = dict()

        # Go through rows and columns
        for row_nr, row in enumerate(state):
            for col_nr, cell in enumerate(row):

                # Go through entities that still have not been found
                for entity_nr, entity in enumerate(ENTITY_NAMES):

                    # Check if found
                    if entity in cell:
                        entity_positions[entity] = (abs(col_nr), abs(row_nr))

                    # Check if all found
                    if len(entity_positions) == 3:
                        return entity_positions

        return entity_positions

    @staticmethod
    def was_pig_caught(prize):
        if prize > 20:
            return True
        return False

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

    @staticmethod
    def directional_wait_action(entity, other_position):
        # Determine direction
        direction = GameObserver.direction_towards_position(entity, other_position)

        # Determine action
        direction_diff = direction - entity.direction
        while direction_diff < -2:
            direction_diff += 4
        while direction_diff > 2:
            direction_diff -= 4
        if direction_diff < 0:
            return AllActions.turn_l
        elif direction_diff > 0:
            return AllActions.turn_r
        else:
            return AllActions.jump

    def get_entities(self):
        return list(self._entities.values())

    def create_entity_positions(self, state, entities):
        # Parse positions from grid
        positions = GameObserver.parse_positions(state)
        for idx, entity in enumerate(entities):
            entity['x'] = positions[entity['name']][0]
            entity['z'] = positions[entity['name']][1]

        # Initialize entities information
        if self._entities is None:
            self._entities = OrderedDict((name, None) for name in ENTITY_NAMES)
            for item in entities:
                self._entities[item['name']] = EntityPosition(name=item['name'], yaw=item['yaw'],
                                                              x=item['x'], z=item['z'])
        else:
            for item in entities:
                # print(item)
                self._entities[item['name']].update(name=item['name'], yaw=item['yaw'],
                                                    x=item['x'], z=item['z'])

        # Entities
        me = self._entities['Agent_2']  # type: EntityPosition
        challenger = self._entities['Agent_1']  # type: EntityPosition
        pig = self._entities['Pig']  # type: EntityPosition

        # Return
        return me, challenger, pig

    @staticmethod
    def determine_targets(state, pig):
        # Exist positions
        exits = [Neighbour(x=1, z=4, direction=0, action=""),
                 Neighbour(x=7, z=4, direction=0, action="")]

        # Get pig position
        pig_node = Location(pig.x, pig.z)

        # Get neighbours
        pig_neighbours = []
        neighbour_cells = []
        for x_diff, z_diff in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_position = (pig_node.x + x_diff, pig_node.z + z_diff)
            if "grass" in state[new_position[1], new_position[0]]:
                pig_neighbours.append(Location(*new_position))
                neighbour_cells.append(state[new_position[0], new_position[1]])

        return pig_neighbours, exits

    @staticmethod
    def search_for_plans(start, exits, pig_neighbours, moves, state, actions):
        goals = exits + pig_neighbours
        paths, _ = GamePlanner.astar_multi_search(start=start,
                                                  goals=goals,
                                                  state=state,
                                                  actions=actions)
        plans = GamePlanner.paths_to_plans(paths=paths,
                                           exits=exits,
                                           pig_neighbours=pig_neighbours,
                                           moves=moves)
        return plans
