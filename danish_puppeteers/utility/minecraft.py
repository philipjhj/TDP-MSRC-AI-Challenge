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


class AllActions:
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

    def __init__(self, move=(1, -1), turn=(1, -1), strafe=(1, -1), jump=False, wait=False):
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
