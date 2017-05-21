import random
import warnings
from itertools import product

import numpy as np
from hmmlearn.hmm import MultinomialHMM

from ai import Plan
from constants import CellGoalType
from ml import FeatureSequence


class Strategies:
    """
    Denotes the assumed possible strategies observable from a challenger. 
    """
    initial_round = 0
    random_walker = 1
    naive_cooperative = 2
    optimal_cooperative = 3
    bad_guy = 4
    irrelevant = 5  # If pig can be caught by one person


class Brain:
    """
    Makes decisions for the DanishPuppet-agent.
    """
    def __init__(self, use_markov, helmets):
        """
        :param bool use_markov: 
            True: Use a Hidden-Markov-Model to train on data and infer challengers strategies.
        :param list[int] helmets: List of possible helmets (needed for encoding hmm-model emissions).
        """
        self._use_markov = use_markov

        # Initializations
        self.markov_model = None
        self.helmets = None
        self.n_helmets = None
        self.possible_emissions = None
        self.decode_emission = None
        self.encode_emission = None
        self.default_sequences = None
        self.n_emissions = None

        # Generate encodings
        self._create_encodings(helmets=helmets)

    def train(self, game_features_history):
        """
        If the Brain is using a markov model, then the model is trained on the input data.
        Otherwise this method does nothing.
        :param list[FeatureSequence] game_features_history: List of features sequences observed in previous games.
        :return: 
        """
        if self._use_markov:
            # Default training data (ensure observing each emission at least once to avoid zero-probabilities)
            x = list(self.default_sequences)

            # Add history
            for game_features in game_features_history:
                x.append([self.encode_emission[tuple(val.to_vector())]
                          for val in game_features.features])

            # Ensure format
            lengths = [len(row) for row in x]
            x = np.atleast_2d(np.concatenate(x))

            # Create model
            self.markov_model = MultinomialHMM(n_components=2,
                                               n_iter=100)

            # Train on data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.markov_model.fit(x.T, lengths=lengths)

    def _create_encodings(self, helmets):
        """
        Creates the emission-encodings for the markov model, depending on the symbols of the helmets.
        :param list[int] helmets: List of possible helmets observable in games.
        """
        # Add symbol for no helmet
        self.helmets = sorted([-1] + helmets)

        # Data
        self.n_helmets = len(self.helmets) + 1
        _towards_pig = [-1, 0, 1]
        _towards_exit = [-1, 0, 1]

        # Possible seen emissions from model
        self.possible_emissions = []
        self.decode_emission = {}
        bad_defaults = []
        good_defaults = []
        for idx, (h, p, e) in enumerate(product(self.helmets,
                                                _towards_pig,
                                                _towards_exit)):
            self.possible_emissions.append(idx)
            feature = (h, p, e)
            self.decode_emission[idx] = feature

            if h == 0 and p == 1:
                if e < 1:
                    good_defaults.extend([idx] * 2)
                else:
                    good_defaults.extend([idx])
            if h == 0 and e == 1:
                if p < 1:
                    bad_defaults.extend([idx] * 2)
                else:
                    bad_defaults.extend([idx])

        # Encoding map
        self.encode_emission = {code: idx for idx, code in self.decode_emission.items()}

        # Default dataset
        self.default_sequences = [
            random.sample(self.possible_emissions, len(self.possible_emissions)),
            # good_defaults * 2,
            # bad_defaults * 2
        ]

        # Number of states and emissions
        self.n_emissions = len(self.possible_emissions)

    def infer_challenger_strategy(self, game_features, helmet, own_plans, verbose=False):
        """
        Infers the strategy of the challenger in order to make decisions from there on.
        :param FeatureSequence game_features: The sequence of features observed in this game.
        :param int helmet: Current inferred helmet. 
        :param list[Plan] own_plans: The possible plans for our agent to exercise.
        :param bool verbose: Want prints?
        :return: int
        """
        if self._use_markov:
            return self._decide_markov_decision(game_features=game_features,
                                                helmet=helmet,
                                                own_plans=own_plans,
                                                verbose=verbose)
        else:
            return self._decide_simple_heuristic(game_features=game_features,
                                                 own_plans=own_plans)

    @staticmethod
    def _decide_simple_heuristic(game_features, own_plans):
        """
        Uses a simple heuristic to decide what to do. 
        Does not use any information from the helmets and does not learn anything.
        :param FeatureSequence game_features: The sequence of features observed in this game.
        :param list[Plan] own_plans: The possible plans for our agent to exercise.
        :return: int
        """
        own_pig_plans = [plan for plan in own_plans if plan.target == CellGoalType.PigCatch]

        # Data on past
        compliances = np.array([feature.compliance for feature in game_features.features])

        # Check if pig can be caught alone
        if len(own_pig_plans) == 1:
            challenger_strategy = Strategies.irrelevant

        # Base predicted strategy on compliance of challenger
        elif compliances.mean() < 0.5:
            challenger_strategy = Strategies.random_walker
        else:
            challenger_strategy = Strategies.naive_cooperative

        return challenger_strategy

    def _decide_markov_decision(self, game_features, helmet, own_plans, verbose=False):
        """
        :param FeatureSequence game_features: The sequence of features observed in this game.
        :param int helmet: Current inferred helmet. 
        :param list[Plan] own_plans: The possible plans for our agent to exercise.
        :param bool verbose: Want prints?
        :return: int
        """
        own_pig_plans = [plan for plan in own_plans if plan.target == CellGoalType.PigCatch]

        # Check if pig can be caught alone (override)
        if len(own_pig_plans) == 1:
            return Strategies.irrelevant

        # Observed sequence
        observed = np.array([self.encode_emission[tuple(val.to_vector())]
                             for val in game_features.features])

        # Sequence emissions
        helping_emission = (helmet, -1, 0)
        backstabbing_emission = (helmet, 0, -1)

        # Generated sequences
        sequence_lengths = 5
        helping_sequence = np.hstack(
            [observed, np.array([self.encode_emission[helping_emission]] * sequence_lengths)])
        backstabbing_sequence = np.hstack(
            [observed, np.array([self.encode_emission[backstabbing_emission]] * sequence_lengths)])

        # Flipping sequence
        flip_sequence = []
        for ____ in range(sequence_lengths // 2):
            flip_sequence.extend([self.encode_emission[helping_emission],
                                  self.encode_emission[backstabbing_emission]])
        if (sequence_lengths % 2) > 0:
            flip_sequence.append(self.encode_emission[helping_emission])
        flip_sequence = np.hstack([observed, np.array(flip_sequence)])

        # Likelihoods
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            likelihoods = []
            for sequence in [helping_sequence, flip_sequence, backstabbing_sequence]:
                likelihoods.append(np.exp(self.markov_model.score(sequence)))
            likelihoods = np.array(likelihoods) / sum(likelihoods)

        if verbose:
            print("Challenger strategy: good {:.2%}, flipping {:.2%}, bad {:.2%}".format(*likelihoods))

        # Determine strategy
        if max(likelihoods[1:]) > likelihoods[0] * 2:
            challenger_strategy = Strategies.random_walker
        else:
            challenger_strategy = Strategies.naive_cooperative

        return challenger_strategy
