import numpy as np

from utility.constants import CellGoalType


class Features:
    def __init__(self, dist_challenger_pig, dist_me_pig, dist_challenger_exit, dist_me_exit,
                 delta_dist_challenger_pig, delta_dist_challenger_exit, helmet, compliance):
        # Distances
        self.dist_challenger_pig = dist_challenger_pig
        self.dist_me_pig = dist_me_pig
        self.dist_challenger_exit = dist_challenger_exit
        self.dist_me_exit = dist_me_exit

        # Number of non-delta features
        self.n_non_delta = 4

        # Delta-distance
        self.delta_dist_challenger_pig = delta_dist_challenger_pig
        self.delta_dist_challenger_exit = delta_dist_challenger_exit

        # Helmet
        self.helmet = helmet
        self.compliance = compliance

    def compute_deltas(self, challenger_pig_distance, challenger_exit_distance):
        delta_challenger_pig_distance = np.sign(challenger_pig_distance - self.dist_challenger_pig)
        delta_challenger_exit_distance = np.sign(challenger_exit_distance - self.dist_challenger_exit)

        # Compliance
        if self.dist_challenger_pig == 0:
            compliance = 1
        else:
            if delta_challenger_pig_distance < 0:
                compliance = 1
            else:
                compliance = 0

        return delta_challenger_pig_distance, delta_challenger_exit_distance, compliance

    def to_list(self):
        return [
            self.helmet,
            self.delta_dist_challenger_pig,
            self.delta_dist_challenger_exit
        ]

    def to_named_list(self):
        items = self.to_list()
        names = [
            "helmet",
            "delta_challenger_pig_distance",
            "delta_challenger_exit_distance",
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

    def reset(self):
        self.features = []

    def _compute_distances(self, own_plans, challengers_plans):
        # Pig distances
        dist_me_pig = min([plan.plan_length() for plan in own_plans
                           if plan.target == CellGoalType.PigCatch])
        dist_challenger_pig = min([plan.plan_length() for plan in challengers_plans
                                   if plan.target == CellGoalType.PigCatch])

        # Exit distances
        dist_me_exit = min([plan.plan_length() for plan in own_plans
                            if plan.target == CellGoalType.Exit])
        dist_challenger_exit = min([plan.plan_length() for plan in challengers_plans
                                    if plan.target == CellGoalType.Exit])

        return dist_me_pig, dist_challenger_pig, dist_me_exit, dist_challenger_exit

    def update(self, own_plans, challengers_plans, current_challenger):

        # Get distances in game
        dist_me_pig, dist_challenger_pig, dist_me_exit, dist_challenger_exit = \
            self._compute_distances(own_plans=own_plans, challengers_plans=challengers_plans)

        # Check if first iteration in game
        if len(self) == 0:
            c_features = Features(dist_challenger_pig=dist_challenger_pig,
                                  dist_me_pig=dist_me_pig,
                                  dist_challenger_exit=dist_challenger_exit,
                                  dist_me_exit=dist_me_exit,
                                  delta_dist_challenger_pig=0,
                                  delta_dist_challenger_exit=0,
                                  helmet=current_challenger,
                                  compliance=1)
            self.features.append(c_features)

        # Otherwise compute deltas
        else:

            # Get last features and compute deltas
            last_features = self.last_features()  # type: Features
            deltas = last_features.compute_deltas(challenger_pig_distance=dist_challenger_pig,
                                                  challenger_exit_distance=dist_challenger_exit)
            delta_challenger_pig_distance, delta_challenger_exit_distance, compliance = deltas

            # Make new features
            c_features = Features(dist_challenger_pig=dist_challenger_pig,
                                  dist_me_pig=dist_me_pig,
                                  dist_challenger_exit=dist_challenger_exit,
                                  dist_me_exit=dist_me_exit,
                                  delta_dist_challenger_pig=delta_challenger_pig_distance,
                                  delta_dist_challenger_exit=delta_challenger_exit_distance,
                                  helmet=current_challenger,
                                  compliance=compliance)

            # Add features
            self.features.append(c_features)

    def to_matrix(self):
        return np.array([feature.to_vector() for feature in self.features])

    def last_features(self):
        return self.features[-1]

    def __len__(self):
        return len(self.features)

    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return str(self)
