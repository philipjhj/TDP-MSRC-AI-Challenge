
PIG_CATCH_PRIZE = 25
EXIT_PRICE = 5


HELMET_NAMES = [
    "iron_helmet",
    "golden_helmet",
    "diamond_helmet",
    "leather_helmet"
]


class EntityNames:
    """
    Holds the names of in-game entities.
    """
    challenger = "Agent_1"
    me = "Agent_2"
    pig = "Pig"

    @staticmethod
    def to_list():
        return [EntityNames.challenger, EntityNames.me, EntityNames.pig]

    class __metaclass__(type):
        def __getitem__(self, item):
            return [EntityNames.challenger, EntityNames.me, EntityNames.pig][item]

        def __iter__(self):
            for name in EntityNames.to_list():
                yield name


class Direction:
    """
    The directions of the game.
    """
    north, east, south, west = range(4)

    class __metaclass__(type):
        def __getitem__(self, item):
            return ["north", "east", "south", "west"][item]


class CellGoalType:
    """
    Cells can be either an Exit-cell, a PigCatch-cell (next to the pig) of a NoGoal-cell.
    """
    Exit = "Exit"
    PigCatch = "PigCatch"
    NoGoal = "None"


class AllActions:
    """
    The actions of a game. 
    The static methods can be used to analyse and iterate through actions.
    """
    move_f = 0
    turn_l = 1
    turn_r = 2
    move_b = 3
    strafe_r = 4
    strafe_l = 5
    jump = 6
    wait = 7

    _action2command = {
        move_f: "move 1",
        turn_l: "turn -1",
        turn_r: "turn 1",
        move_b: "move -1",
        strafe_r: "strafe 1",
        strafe_l: "strafe -1",
        jump: "jump 1",
        wait: "wait"
    }

    def __init__(self, move=(1, -1), turn=(1, -1), strafe=(1, -1), jump=False, wait=False):
        """
        Creates an entity with a subset of the possible actions.
        Can be used to allow agents to have different subsets of actions.
        :param tuple move: Can agent move forward (has a 1) and backwards (has a -1)
        :param tuple turn: Can agent turn right (has a 1) and left (has a -1)
        :param tuple strafe: Can agent strafe right (has a 1) and left (has a -1) 
        :param bool jump: Can agent jump.
        :param bool wait: Can agent output a wait-flag which does not take up a turn, 
            but refreshed information from server.
        """
        self._actions = []

        # Move forth and back
        if 1 in move:
            self._actions.append(AllActions.move_f)
        if -1 in move:
            self._actions.append(AllActions.move_b)

        # Turn
        if 1 in turn:
            self._actions.append(AllActions.turn_r)
        if -1 in turn:
            self._actions.append(AllActions.turn_l)

        # Strafe
        if 1 in strafe:
            self._actions.append(AllActions.strafe_r)
        if -1 in strafe:
            self._actions.append(AllActions.strafe_l)

        # Jump
        if jump:
            self._actions.append(AllActions.jump)

        # Wait
        if wait:
            self._actions.append(AllActions.wait)

    @staticmethod
    def is_move(action):
        return action in {AllActions.move_f, AllActions.move_b}

    @staticmethod
    def is_turn(action):
        return action in {AllActions.turn_l, AllActions.turn_r}

    @staticmethod
    def is_strafe(action):
        return action in {AllActions.strafe_l, AllActions.strafe_r}

    @staticmethod
    def sign(action):
        if action in {AllActions.turn_l, AllActions.move_b, AllActions.strafe_l}:
            return -1
        else:
            return 1

    @staticmethod
    def all_commands():
        return [AllActions._action2command[idx] for idx in range(8)]

    @staticmethod
    def all_actions():
        return list(range(8))

    @staticmethod
    def action_to_command(action):
        return AllActions._action2command[action]

    def __iter__(self):
        for action in self._actions:
            yield action

    def __len__(self):
        return len(self._actions)


KeysMapping = {'L': AllActions.turn_l,
               'l': AllActions.turn_l,
               'R': AllActions.turn_r,
               'r': AllActions.turn_r,
               'U': AllActions.move_f,
               'u': AllActions.move_f,
               'F': AllActions.move_f,
               'f': AllActions.move_f,
               'B': AllActions.move_b,
               'b': AllActions.move_b,
               # AWSD + QE
               "a": AllActions.strafe_l,
               "A": AllActions.strafe_l,
               "w": AllActions.move_f,
               "W": AllActions.move_f,
               "s": AllActions.move_b,
               "S": AllActions.move_b,
               "d": AllActions.strafe_r,
               "D": AllActions.strafe_r,
               'q': AllActions.turn_l,
               'Q': AllActions.turn_l,
               'e': AllActions.turn_r,
               'E': AllActions.turn_r,
               # Jump
               ' ': AllActions.jump,
               # No-op
               "n": AllActions.wait,
               "N": AllActions.wait,
               }
