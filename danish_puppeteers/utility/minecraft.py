from collections import OrderedDict

from ai import EntityPosition, SearchNode, Location, GamePlanner
from constants import EntityNames
from ml import FeatureSequence


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
    """
    Keeps track of the in-game time. 
    Is based on manual observations on the TotalTime from the environment.
    Also... Why do we not have easy access to the game time!?
    """
    TOTAL_TIME_FACTOR = 1. / 20.
    TOTAL_GAME_TIME = 1000.

    def __init__(self):
        self._start_time = None
        self.time_left = None

    def reset(self):
        """
        Resets timer before game.
        """
        self._start_time = None
        self.time_left = None

    def update(self, total_time):
        """
        Updates timer based on input from environment. 
        :param int total_time: Should be fetched with env.world_observations["TotalTime"]
        """
        if self._start_time is None:
            self._start_time = total_time
            self.time_left = GameTimer.TOTAL_GAME_TIME
        else:
            self.time_left = GameTimer.TOTAL_GAME_TIME - \
                             (total_time - self._start_time) * GameTimer.TOTAL_TIME_FACTOR


class GameSummary:
    """
    Keeps a summary of a game.
    Used for evaluating how an agent is performing. 
    """
    def __init__(self, feature_sequence, reward, prize, final_state, pig_is_caught):
        """
        :param FeatureSequence feature_sequence: 
        :param int reward: Final reward (prize - n_moves)
        :param int prize: Prize from game.
        :param list final_state: State from environment at end.
        :param bool pig_is_caught: Whether the pig was caught or not. 
        """
        self.feature_matrix = feature_sequence
        self.reward = reward
        self.prize = prize
        self.final_state = final_state
        self.pig_is_caught = pig_is_caught

    def __str__(self):
        return "GameSummary(reward={}, pig_is_caught={})".format(self.reward, self.pig_is_caught)

    def __repr__(self):
        return str(self)


class GameObserver:
    """
    Used for handling the information from the Minecraft environment. 
    """
    def __init__(self):
        self._entities = None

    def reset(self):
        self._entities = None

    @staticmethod
    def parse_positions(state):
        """
        Parses a state-matrix and returns the positions of the entities in the map.
        :param list state: 
        :return: dict
        """
        entity_positions = dict()

        # Go through rows and columns
        for row_nr, row in enumerate(state):
            for col_nr, cell in enumerate(row):

                # Go through entities that still have not been found
                for entity_nr, entity in enumerate(EntityNames):

                    # Check if found
                    if entity in cell:
                        entity_positions[entity] = (abs(col_nr), abs(row_nr))

                    # Check if all found
                    if len(entity_positions) == 3:
                        return entity_positions

        return entity_positions

    @staticmethod
    def was_pig_caught(prize):
        """
        Returns true if the pig was caught, based on the given prize.
        The prize seems to vary (but 24 and 25 was observed), so we based it on a lower threshold.
        :param int prize: 
        :return: bool
        """
        if prize > 20:
            return True
        return False

    def get_entities(self):
        """
        For looping over entities.
        :return: list[str, Position]
        """
        return list(self._entities.values())

    def create_entity_positions(self, state, entities):
        """
        Used for making various corrections to the information from the state and entities objects
        given from the environment. 
        :param list state: 
        :param dict entities: 
        :return: EntityPosition, EntityPosition, EntityPosition
        """
        # Parse positions from grid
        positions = GameObserver.parse_positions(state)
        for idx, entity in enumerate(entities):
            entity['x'] = positions[entity['name']][0]
            entity['z'] = positions[entity['name']][1]

        # Initialize entities information
        if self._entities is None:
            self._entities = OrderedDict((name, None) for name in EntityNames)
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
        """
        Determines the current target-locations in the game (exits and cells by the pig).
        :param np.array state: 
        :param EntityPosition pig: 
        :return: (list[Location], list[Location])
        """
        # Exist positions
        exits = [SearchNode(x=1, z=4, direction=0, action=""),
                 SearchNode(x=7, z=4, direction=0, action="")]

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
        """
        Finds an agents shortest plans in game.
        :param EntityPosition start: 
        :param list[Location] exits: 
        :param list[Location] pig_neighbours: 
        :param int moves: 
        :param np.array state: 
        :param list[int] actions: 
        :return: 
        """
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
