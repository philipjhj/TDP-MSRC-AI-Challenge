import numpy as np


class Features:
    def __init__(self, challenger_pig_distance, own_pig_distance, challenger_exit_distance, own_exit_distance,
                 delta_challenger_pig_distance, delta_challenger_exit_distance, helmet, compliance):
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

        # Helmet
        self.helmet = helmet
        self.compliance = compliance

    def compute_deltas(self, challenger_pig_distance, challenger_exit_distance):
        delta_challenger_pig_distance = np.sign(challenger_pig_distance - self.challenger_pig_distance)
        delta_challenger_exit_distance = np.sign(challenger_exit_distance - self.challenger_exit_distance)

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
            self.helmet,
            self.delta_challenger_pig_distance,
            self.delta_challenger_exit_distance
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

    def update(self, features):
        self.features.append(features)

    def to_matrix(self):
        return np.array([feature.to_vector() for feature in self.features])

    def last_features(self):
        return self.features[-1]

    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return str(self)
