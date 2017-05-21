from __future__ import division

import pickle
import random
import time
import warnings
from itertools import product

import numpy as np
from hmmlearn.hmm import MultinomialHMM
from pathlib2 import Path
from queue import Queue

from constants import AllActions, KeysMapping
from malmopy.agent import BaseAgent
from utility.ai import Plan, Neighbour, Location, GamePlanner
from utility.helmet_detection import HelmetDetector, HELMET_NAMES
from utility.minecraft import map_view, GameSummary, GameObserver
from utility.ml import Features, FeatureSequence

P_FOCUSED = .75
CELL_WIDTH = 33

MEMORY_SIZE = 10

N_HELMETS = 4
N_MARKOV_STATES = 4

DEBUG_STORE_IMAGE = False
SECONDS_WAIT_BEFORE_IMAGE = 1

# Data
HELMETS = list(range(N_HELMETS + 1))
_towards_pig = [-1, 0, 1]
_towards_exit = [-1, 0, 1]

# Possible seen emissions from model
# TODO: Move to GameObserver
SAMPLES_IN_MEMORY = 10000
POSSIBLE_EMISSIONS = []
DECODE_EMISSION = {}
BAD_DEFAULTS = []
GOOD_DEFAULTS = []
for idx, (h, p, e) in enumerate(product(HELMETS, _towards_pig, _towards_exit)):
    POSSIBLE_EMISSIONS.append(idx)
    feature = (h, p, e)
    DECODE_EMISSION[idx] = feature

    if h == 0 and p == 1:
        if e < 1:
            GOOD_DEFAULTS.extend([idx] * 2)
        else:
            GOOD_DEFAULTS.extend([idx])
    if h == 0 and e == 1:
        if p < 1:
            BAD_DEFAULTS.extend([idx] * 2)
        else:
            BAD_DEFAULTS.extend([idx])
ENCODE_EMISSION = {code: idx for idx, code in DECODE_EMISSION.items()}

# Default dataset
DEFAULT_SEQUENCES = [
    random.sample(POSSIBLE_EMISSIONS, len(POSSIBLE_EMISSIONS)),
    GOOD_DEFAULTS * 2,
    BAD_DEFAULTS * 2
]

# Number of states and emissions
N_EMISSIONS = len(POSSIBLE_EMISSIONS)


def print_if(condition, string):
    """
    Prints if condition is true.
    :param bool condition: 
    :param str string: 
    """
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
    helmet_detection = True
    timing = False
    iteration_line = False
    map = False
    positions = False
    steps_to_other = False
    pig_neighbours = False
    own_plans = False
    challenger_plans = False
    detailed_plans = False
    feature_vector = False
    feature_matrix = False
    challenger_strategy = True
    waiting_info = True
    repeated_waiting_info = False

    # Post game
    game_summary = True


class DanishPuppet(BaseAgent):
    ACTIONS = AllActions()
    BASIC_ACTIONS = AllActions(move=(1,), strafe=())

    def __init__(self, name, visualizer=None, wait_for_pig=True, use_markov=True):
        super(DanishPuppet, self).__init__(name, len(DanishPuppet.ACTIONS), visualizer)
        self._previous_target_pos = None
        self._action_list = []

        # Other fields
        self.manual = False
        self.first_act_call = True

        # Features and memory
        self.game_observer = GameObserver()
        self.game_features = FeatureSequence()
        self.game_summary_history = Queue(maxsize=MEMORY_SIZE)

        # Timing and iterations
        self.n_moves = -1
        self.n_games = 0
        self.n_total_moves = 0
        self.time_start = time.time()

        # Pig stuff
        self.waiting_for_pig = False
        self.wait_for_pig_if_necessary = wait_for_pig
        self.pig_wait_counter = 0

        # Image analysis
        self.initial_waits = True
        self.helmet_detector = HelmetDetector(storage_path="../danish_puppeteers/storage")
        self.helmet_detector.load_classifier()

        # Decision making
        self.use_markov = use_markov
        self.markov_model = None
        self.game_features_history = []

    def time_alive(self):
        return time.time() - self.time_start

    def note_game_end(self, reward_sequence, state):
        if self.game_summary_history.full():
            self.game_summary_history.get()

        prize = int(max(reward_sequence))
        reward = int(sum(reward_sequence))

        game_summary = GameSummary(feature_matrix=self.game_features,
                                   reward=reward,
                                   prize=prize,
                                   final_state=state,
                                   pig_is_caught=GameObserver.was_pig_caught(prize=prize))

        if Print.game_summary:
            print("\nGame Summary:")
            print("   {}".format(game_summary))
            print("   From reward-sequence: {}".format(reward_sequence))

        self.game_summary_history.put(game_summary)
        self.game_features_history.append(self.game_features)
        if len(self.game_features_history) > SAMPLES_IN_MEMORY:
            self.game_features_history = random.sample(self.game_features_history, SAMPLES_IN_MEMORY)

    def act(self, state, reward, done, is_training=False, frame=None):

        if Print.act_reached:
            print("DanishPuppet.act() called")

        ###############################################################################
        # Initializations

        if done:
            print_if(Print.code_line_print, "CODE: Initialization")
            if self.first_act_call:

                if Print.history_length:
                    print("\nLength of history: {}".format(self.game_summary_history.qsize()))

                self._action_list = []
                self._previous_target_pos = None
                self.game_features = FeatureSequence()
                self.n_moves = -1
                self.initial_waits = True
                self.game_observer.reset()
                self.helmet_detector.reset()

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

        ###############################################################################
        # If pig is not in a useful place - wait
        print_if(Print.code_line_print, "CODE: Considering pregame-wait")

        if self.initial_waits:
            print("\nHold your horses.\n")
            time.sleep(SECONDS_WAIT_BEFORE_IMAGE)
            self.initial_waits = False

            ##############
            # Do markov model training

            # Training data
            X = list(DEFAULT_SEQUENCES)

            # Add history
            for game_features in self.game_features_history:
                X.append([ENCODE_EMISSION[tuple(val.to_vector())]
                          for val in game_features.features])

            # Ensure format
            lengths = [len(row) for row in X]
            X = np.atleast_2d(np.concatenate(X))

            # Create model
            self.markov_model = MultinomialHMM(n_components=2,
                                               n_iter=100)

            # Train on data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.markov_model.fit(X.T, lengths=lengths)

            # Send wait action to get a refreshed frame
            return DanishPuppet.ACTIONS.wait

        ###############################################################################
        # Get information from environment
        print_if(Print.code_line_print, "CODE: Information parsing")

        entities = state[1]
        state = state[0]
        self.n_moves += 1

        if Print.iteration_line and not self.waiting_for_pig:
            print("\n---------------------------\n")

        # Get positions from game
        me, challenger, pig = self.game_observer.create_entity_positions(state=state,
                                                                         entities=entities)

        if Print.map and not self.waiting_for_pig:
            print(map_view(state))

        if Print.positions and not self.waiting_for_pig:
            for item in self.game_observer.get_entities():
                print(item)

        if Print.steps_to_other:
            print("Steps to challenger: {}".format(GameObserver.directional_steps_to_other(me, challenger)))

        if DEBUG_STORE_IMAGE:
            HelmetDetector.store_snapshot(me=me,
                                          challenger=challenger,
                                          pig=pig,
                                          state=state,
                                          frame=frame)

        ###############################################################################
        # Determine possible targets
        print_if(Print.code_line_print, "CODE: Determining targets")

        # Specific targets
        pig_neighbours, exits = GameObserver.determine_targets(state=state,
                                                               pig=pig)

        if Print.pig_neighbours:
            print("\nPig neighbours:")
            for neighbour, cell in zip(pig_neighbours):
                print("   {}: {}".format(neighbour, cell))

        ###############################################################################
        # Compute possible plans for each player plans
        print_if(Print.code_line_print, "CODE: Computing plans")

        # Find own paths
        own_plans = GameObserver.search_for_plans(start=me,
                                                  exits=exits,
                                                  pig_neighbours=pig_neighbours,
                                                  moves=self.n_moves,
                                                  state=state,
                                                  actions=list(DanishPuppet.ACTIONS))

        # Find challengers paths
        challengers_plans = GameObserver.search_for_plans(start=challenger,
                                                          exits=exits,
                                                          pig_neighbours=pig_neighbours,
                                                          moves=self.n_moves,
                                                          state=state,
                                                          actions=list(DanishPuppet.BASIC_ACTIONS))

        if Print.own_plans:
            print("\nOwn {} plans:".format(len(own_plans)))
            for plan in own_plans:
                print("   {}".format(plan))
                if Print.detailed_plans:
                    print("      {}".format(plan.path_print()))

        if Print.challenger_plans:
            print("\nChallenger {} plans:".format(len(challengers_plans)))
            for plan in challengers_plans:
                print("   {}".format(plan))
                if Print.detailed_plans:
                    print("      {}".format(plan.path_print()))

        # Pig plans for both agents
        own_pig_plans = [plan for plan in own_plans if plan.target == Plan.PigCatch]
        challenger_pig_plans = [plan for plan in challengers_plans if plan.target == Plan.PigCatch]

        ###############################################################################
        # If pig is not in a useful place - wait
        print_if(Print.code_line_print, "CODE: Considering pig-wait")

        # Check if pig is in an area, where it can not be caught
        if len(pig_neighbours) > 2 and self.wait_for_pig_if_necessary:

            # Prints
            if Print.repeated_waiting_info or not self.waiting_for_pig:
                if Print.waiting_info:
                    print("\nWaiting for pig at {}...".format(pig))
                    if Print.map:
                        print(map_view(state))
                self.waiting_for_pig = True

            # Count number of waits and return wait command
            self.pig_wait_counter += 1
            return DanishPuppet.ACTIONS.wait

        # Back in action
        self.waiting_for_pig = False
        self.pig_wait_counter = 0

        ###############################################################################
        # Detect helmet

        # Check if on top of challenger
        current_challenger, helmet_probabilities, decision_made = \
            self.helmet_detector.detect_helmet(me=me,
                                               challenger=challenger,
                                               frame=frame)

        # Prints
        if Print.helmet_detection:
            if decision_made:
                print("\nHelmet detected with probability {:.2%}. Number {} ({})".format(
                    helmet_probabilities[current_challenger],
                    current_challenger,
                    HELMET_NAMES[current_challenger]))
            else:
                print("\nHelmet not detected.")

        # Format for features
        if current_challenger is None:
            current_challenger = -1

        ###############################################################################
        # Feature Extraction
        print_if(Print.code_line_print, "CODE: Extracting features")

        self.game_features.update(own_plans=own_plans,
                                  challengers_plans=challengers_plans,
                                  current_challenger=current_challenger)

        if Print.feature_vector:
            print("\nFeature vector:")
            print("   {}".format(self.game_features.last_features()))

        if Print.feature_matrix:
            print("\nFeature matrix:")
            print(self.game_features.to_matrix())

        ###############################################################################
        # This is a move

        self.n_total_moves += 1

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

        if not self.use_markov:

            # Data on past
            compliances = np.array([feature.compliance for feature in self.game_features.features])

            # Check if pig can be caught alone
            if len(own_pig_plans) == 1:
                challenger_strategy = 5

            # Base predicted strategy on compliance of challenger
            elif compliances.mean() < 0.5:
                challenger_strategy = 1
            else:
                challenger_strategy = 2

        ###############################################################################
        # Markov-modelling

        else:

            # Observed sequence
            observed = np.array([ENCODE_EMISSION[tuple(val.to_vector())]
                                 for val in self.game_features.features])

            # Get helmet
            helmet = current_challenger + 1

            # Sequence emissions
            helping_emission = (helmet, 1, 0)
            backstabbing_emission = (helmet, 0, 1)

            # Generated sequences
            sequence_lengths = 5
            helping_sequence = np.hstack([observed, np.array([ENCODE_EMISSION[helping_emission]] * sequence_lengths)])
            backstabbing_sequence = np.hstack(
                [observed, np.array([ENCODE_EMISSION[backstabbing_emission]] * sequence_lengths)])

            # Flipping sequence
            flip_sequence = []
            for ____ in range(sequence_lengths // 2):
                flip_sequence.extend([ENCODE_EMISSION[helping_emission], ENCODE_EMISSION[backstabbing_emission]])
            if (sequence_lengths % 2) > 0:
                flip_sequence.append(ENCODE_EMISSION[helping_emission])
            flip_sequence = np.hstack([observed, np.array(flip_sequence)])

            # Likelihoods
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                likelihoods = []
                for sequence in [helping_sequence, flip_sequence, backstabbing_sequence]:
                    likelihoods.append(np.exp(self.markov_model.score(sequence)))
                likelihoods = np.array(likelihoods) / sum(likelihoods)

            print("Challenger strategy: good {}, flipping {}, bad {}".format(*likelihoods))

            # Determine strategy
            if max(likelihoods[1:]) > likelihoods[0] * 2:
                challenger_strategy = 1
            else:
                challenger_strategy = 0

        ###############################################################################
        # Manual overwrite

        # Check if manual control is wanted
        print_if(Print.code_line_print, "CODE: Considering manual overwrite")
        if self.manual:
            while True:
                choice = raw_input("\nType action (AWSD + QE): ")
                if choice in KeysMapping:
                    print("   Choice {} translated to {}".format(choice, KeysMapping[choice]))
                    return KeysMapping[choice]

        ###############################################################################
        # Determine plan based on challengers strategy
        print_if(Print.code_line_print, "CODE: Selecting plan based on strategy")

        # If challenger is an idiot or a douche - backstab him!
        if challenger_strategy in {1, 4}:
            if Print.challenger_strategy:
                print("\nChallenger seems to be an idiot.")

            # Exit plan
            plan = sorted([plan for plan in own_plans if plan.target == Plan.Exit],
                          key=lambda x: -x.utility)[0]

            # Return next action (0th element is current position)
            action = plan[1].action
            return action

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
                return action

            # Find shortest plan from challenger
            challenger_pig_plan = sorted(challenger_pig_plans, key=lambda x: -x.utility)[0]

            # Find remaining spot for own agent
            own_pig_plan = next((plan for plan in own_pig_plans
                                 if not GamePlanner.matches(plan, challenger_pig_plan)))

            # Check if already at pig
            if len(own_pig_plan) < 2:
                if Print.waiting_info:
                    print("\nWaiting for challenger to help with pig ...")
                return GameObserver.directional_wait_action(entity=me,
                                                            other_position=challenger)

            # Return next action (0th element is current position)
            action = own_pig_plan[1].action
            return action
