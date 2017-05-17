from collections import deque
from heapq import heapify, heappush, heappop

from utility.minecraft import AllActions


class Location:
    """
    Holds a location in the game. Basically works as a named tuple of x- and z-coordinates. 
    """
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return [self.x, self.z][item]

    def __iter__(self):
        for item in [self.x, self.z]:
            return item

    def __str__(self):
        return "({}, {})".format(self.x, self.z)


class Neighbour(Location):
    def __init__(self, x, z, direction, action):
        """
        Expansion of Location to also hold direction and an action.
        Used for searching graphs. 
        """
        Location.__init__(self, x, z)
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


class EntityPosition(Location):
    """
    Expansion of Location to also hold direction and name of an entity. 
    """
    def __init__(self, name, yaw, x, z):
        Location.__init__(self, x, z)
        self.name = name
        self.direction = self.find_direction(yaw)

    @staticmethod
    def _angle_float(angle):
        return ((angle % 360) // 90) - 1

    def find_direction(self, angle):
        angle = self._angle_float(angle - 45)
        return int(angle % 4)

    def update(self, name, yaw, x, z):
        self.name = name
        self.z = z
        self.x = x

        # Ensure consistency in angles
        self.direction = self.find_direction(yaw)

    def to_neighbour(self):
        return Neighbour(x=self.x, z=self.z, direction=self.direction, action="")

    def __str__(self):
        return "{: >16s}(x={:d}, z={:d}, direction={})".format(self.name + "_Position",
                                                               self.x,
                                                               self.z,
                                                               self.direction)

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


class GamePlanner:
    north, east, south, west = range(4)

    def __init__(self):
        pass

    @staticmethod
    def move_inc_x(x, direction, delta):
        if direction == GamePlanner.east:
            return x + delta
        if direction == GamePlanner.west:
            return x - delta
        return x

    @staticmethod
    def move_inc_z(z, direction, delta):
        if direction == GamePlanner.south:
            return z + delta
        if direction == GamePlanner.north:
            return z - delta
        return z

    @staticmethod
    def strafe_inc_x(x, direction, delta):
        if direction == GamePlanner.north:
            return x + delta
        if direction == GamePlanner.south:
            return x - delta
        return x

    @staticmethod
    def strafe_inc_z(z, direction, delta):
        if direction == GamePlanner.east:
            return z + delta
        if direction == GamePlanner.west:
            return z - delta
        return z

    @staticmethod
    def neighbors(pos, actions, state=None):
        # State information
        state_width = state.shape[1]
        state_height = state.shape[0]

        # Initialize list of neighbours
        neighbors = []

        # Add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in actions:
            sign = AllActions.sign(action)

            # Turning actions
            if AllActions.is_turn(action):
                new_direction = (pos.direction + sign) % 4
                new_state = Neighbour(x=pos.x,
                                      z=pos.z,
                                      direction=new_direction,
                                      action=action)
                neighbors.append(new_state)

            # Strafing actions
            if AllActions.is_strafe(action):
                neighbors.append(
                    Neighbour(x=GamePlanner.strafe_inc_x(pos.x, pos.direction, sign),
                              z=GamePlanner.strafe_inc_z(pos.z, pos.direction, sign),
                              direction=pos.direction,
                              action=action))

            # Moving actions
            if AllActions.is_move(action):
                neighbors.append(
                    Neighbour(x=GamePlanner.move_inc_x(pos.x, pos.direction, sign),
                              z=GamePlanner.move_inc_z(pos.z, pos.direction, sign),
                              direction=pos.direction,
                              action=action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            0 <= n.x < state_width and 0 <= n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours

    @staticmethod
    def heuristic(a, b):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def matches(a, b):
        return a.x == b.x and a.z == b.z  # don't worry about dir and action

    @staticmethod
    def astar_multi_search(start, goals, state, actions):
        """
        Searches the entire graph for the shortest path from one location to any of a list of goals. 
        :param Location |Position | Neighbour start: 
        :param list goals:  
        :return: 
        """
        # Ensure uniqueness of goal-positions
        goals_remaining = list(set(Location(item.x, item.z) for item in goals))

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

        # Keep fetching nodes from queue
        while len(explorer) > 0:
            current_priority, current = heappop(explorer)

            # Check if any goal is reached
            for idx, goal in enumerate(goals_remaining):
                if GamePlanner.matches(current, goal):
                    goal_nodes.append(current)
                    del goals_remaining[idx]

            if not goals_remaining:
                break

            # Go through neighbours
            for nb in GamePlanner.neighbors(current, actions=actions, state=state):
                # Compute new cost
                cost = nb.cost if hasattr(nb, "cost") else 1
                new_cost = cost_so_far[current] + cost

                # Check if node has never been seen before or if cost is lower than
                # previously found path (shouldn't happen)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost

                    # Find minimum heuristic
                    heuristic = min([GamePlanner.heuristic(goal, nb) for goal in goals_remaining])

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
