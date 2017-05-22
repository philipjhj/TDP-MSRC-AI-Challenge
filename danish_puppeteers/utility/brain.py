import random
import warnings
from itertools import product

import numpy as np
from hmmlearn.hmm import MultinomialHMM
from pathlib2 import Path
import cPickle as pickle

from utility.ai import Plan
from utility.constants import CellGoalType, PIG_CATCH_PRIZE, EXIT_PRICE
from utility.ml import FeatureSequence
from utility.util import Paths, ensure_folder


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

    def __init__(self, use_markov, helmets, iterations_per_training, load_markov_from_file=True, n_markov_components=2):
        """
        :param bool use_markov: 
            True: Use a Hidden-Markov-Model to train on data and infer challengers strategies.
        :param list[int] helmets: List of possible helmets (needed for encoding hmm-model emissions).
        """
        self._use_markov = use_markov
        self._iterations_per_training = iterations_per_training
        self._current_training_call = 0
        self._load_markov_from_file = load_markov_from_file

        # Initializations
        self.helmets = None
        self.n_helmets = None
        self.possible_emissions = None
        self.decode_emission = None
        self.encode_emission = None
        self.default_sequences = None
        self.n_emissions = None

        # Emissions types
        self.emissions_towards_pig = None
        self.emissions_away_from_pig = None
        self.emissions_indifferent_pig = None
        self.emissions_towards_exit = None
        self.emissions_away_from_exit = None
        self.emissions_indifferent_exit = None

        # Generate encodings
        self._create_encodings(helmets=helmets)

        # Initialize model by training with default data
        if use_markov:
            if load_markov_from_file:
                self.load_model()
                print("Brain loaded from file.")
            else:
                self._initialize_markov_model(n_markov_components)
                print("New brain initialized.")

            # Do not initialize parameters at next training-steps (just update)
            self.markov_model.init_params = ""
        else:
            self.markov_model = None  # type: MultinomialHMM
            print("No-brainer used.")

    def _initialize_markov_model(self, n_markov_components):
        self.markov_model = MultinomialHMM(n_components=n_markov_components,
                                           n_iter=100)
        x = [[val
              for sublist in self.default_sequences
              for val in sublist]]
        lengths = [len(row) for row in x]
        x = np.atleast_2d(np.concatenate(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.markov_model.fit(x.T, lengths=lengths)

    def train(self, game_features_history):
        """
        If the Brain is using a markov model, then the model is trained on the input data.
        Otherwise this method does nothing.
        :param list[FeatureSequence] game_features_history: List of features sequences observed in previous games.
        :return: 
        """

        if self._use_markov and not self._load_markov_from_file:
            self._current_training_call += 1
            if (self._current_training_call % self._iterations_per_training) == 0:
                return
            self._current_training_call = 0

            if len(game_features_history) < 1:
                return

            # Add history
            x = list(self.default_sequences)
            for game_features in game_features_history:
                x.append([self.encode_emission[tuple(val.to_vector())]
                          for val in game_features.features])

            # Ensure format
            lengths = [len(row) for row in x]
            x = np.atleast_2d(np.concatenate(x))

            # Train on data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.markov_model.fit(x.T, lengths=lengths)

    def save_model(self):
        ensure_folder(Paths.brain_models)
        file_name = "brain_helmets_{}.p".format("".join(str(val) for val in self.helmets))
        pickle.dump(self.markov_model, Path(Paths.brain_models, file_name).open("wb"))

    def load_model(self):
        file_name = "brain_helmets_{}.p".format("".join(str(val) for val in self.helmets))
        self.markov_model = pickle.load(Path(Paths.brain_models, file_name).open("rb"))

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
        self.default_sequences = [[val] for val in
                                  random.sample(self.possible_emissions, len(self.possible_emissions))
                                  # good_defaults * 2,
                                  # bad_defaults * 2
                                  ]

        # Number of states and emissions
        self.n_emissions = len(self.possible_emissions)

        # Determine emission-types
        self.emissions_towards_pig = [emission_code for emission_code in self.possible_emissions
                                      if self.decode_emission[emission_code][1] > 0]
        self.emissions_away_from_pig = [emission_code for emission_code in self.possible_emissions
                                        if self.decode_emission[emission_code][1] < 0]
        self.emissions_indifferent_pig = [emission_code for emission_code in self.possible_emissions
                                          if self.decode_emission[emission_code][1] == 0]
        self.emissions_towards_exit = [emission_code for emission_code in self.possible_emissions
                                       if self.decode_emission[emission_code][2] > 0]
        self.emissions_away_from_exit = [emission_code for emission_code in self.possible_emissions
                                         if self.decode_emission[emission_code][2] < 0]
        self.emissions_indifferent_exit = [emission_code for emission_code in self.possible_emissions
                                           if self.decode_emission[emission_code][2] == 0]

    def infer_challenger_strategy(self, game_features, own_plans, verbose=False):
        """
        Infers the strategy of the challenger in order to make decisions from there on.
        :param FeatureSequence game_features: The sequence of features observed in this game.
        :param list[Plan] own_plans: The possible plans for our agent to exercise.
        :param bool verbose: Want prints?
        :return: int
        """
        if self._use_markov:
            return self._decide_markov_decision(game_features=game_features,
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
        elif compliances.mean() < 0.4:
            challenger_strategy = Strategies.random_walker
        else:
            challenger_strategy = Strategies.naive_cooperative

        return challenger_strategy

    def _decide_markov_decision(self, game_features, own_plans, verbose=False):
        """
        :param FeatureSequence game_features: The sequence of features observed in this game.
        :param list[Plan] own_plans: The possible plans for our agent to exercise.
        :param bool verbose: Want prints?
        :return: int
        """
        own_pig_plans = [plan for plan in own_plans if plan.target == CellGoalType.PigCatch]

        # Check if pig can be caught alone (override)
        if len(own_pig_plans) == 1:
            return Strategies.irrelevant

        # Observed sequence
        observed = [[self.encode_emission[tuple(val.to_vector())]
                     for val in game_features.features]]

        # Converting to suitable format
        lengths = [len(row) for row in observed]
        observed = np.atleast_2d(np.concatenate(observed)).T

        # Predict state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prop_state = self.markov_model.predict_proba(observed, lengths=lengths)[-1, :]

        # Marginal helpful probability of emissions for each state
        prob_towards_pig = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                       for idx in self.emissions_towards_pig]), prop_state)
        prob_away_from_pig = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                         for idx in self.emissions_away_from_pig]), prop_state)
        prob_indifferent_pig = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                           for idx in self.emissions_indifferent_pig]), prop_state)
        prob_towards_exit = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                        for idx in self.emissions_towards_exit]), prop_state)
        prob_away_from_exit = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                          for idx in self.emissions_away_from_exit]), prop_state)
        prob_indifferent_exit = np.dot(sum([self.markov_model.emissionprob_[:, idx]
                                            for idx in self.emissions_indifferent_exit]), prop_state)
        
        bad_or_neutral = prob_indifferent_pig + prob_away_from_pig
        
        if verbose:
            print("Chellender state: {}".format(prop_state))
            print("Challenger strategy: good {:.2%}, bad-or-neutral {:.2%}".format(prob_towards_pig,
                                                                                bad_or_neutral))

        # Determine strategy
        if bad_or_neutral > prob_towards_pig * (PIG_CATCH_PRIZE / EXIT_PRICE):
            challenger_strategy = Strategies.random_walker
        else:
            challenger_strategy = Strategies.naive_cooperative

        return challenger_strategy


if __name__ == "__main__":
    # Folder with training data
    folder_path = Paths.brain_training_data

    # Get all files
    file_base_name = "game_"
    files_in_directory = [item for item in folder_path.glob(file_base_name + "*.p")]

    # Get all data
    data_history = [pickle.load(c_file_path.open("rb")) for c_file_path in files_in_directory]

    # Find number of helmets
    helmets = set()
    for data in data_history:
        if data:
            helmets.update(data.to_matrix()[:, 0])
    helmets.remove(-1)
    helmets = sorted(list(helmets))
    print("Helmets found in data: {}".format(helmets))

    # Make a brain
    brain = Brain(use_markov=True, helmets=helmets, iterations_per_training=1, load_markov_from_file=False,
                  n_markov_components=3)

    # Train brain
    brain.train(data_history)

    # Store brain
    brain.save_model()





