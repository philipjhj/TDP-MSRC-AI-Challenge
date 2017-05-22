from __future__ import division

import pickle
import random
import time

from pathlib2 import Path
from queue import Queue

from utility.brain import Brain, Strategies
from utility.constants import AllActions, KeysMapping, CellGoalType, HELMET_NAMES
from malmopy.agent import BaseAgent
from utility.ai import GamePlanner
from utility.helmet_detection import HelmetDetector
from utility.minecraft import map_view, GameSummary, GameObserver, GameTimer
from utility.ml import FeatureSequence

# TODO: I think you can still get stuck!
#   TODO: Print times when waiting for the pig - do they not increment?
#   TODO: Possibly ask agent to do something irrelevant (fx. jump) if waining too may iterations for the pig.
#   TODO: It will cost an action, but may avoid the server crashing.

# TODO: Make the following input to the agent's constructor
from utility.util import ensure_folder

SUMMARY_HISTORY_SIZE = 10

SAMPLES_IN_MEMORY = 100
DEBUG_STORE_IMAGE = False
SECONDS_WAIT_BEFORE_IMAGE = 0.1


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

    # Niceness
    iteration_line = False

    # Environment
    map = False
    positions = False
    steps_to_other = False
    pig_neighbours = False

    # Searches
    own_plans = False
    challenger_plans = False
    detailed_plans = False

    # Timing
    real_time_timing = False
    game_timing = False

    # Features
    feature_vector = False
    feature_matrix = False

    # Decision making
    helmet_detection = True
    challenger_strategy = True
    waiting_info = True
    repeated_waiting_info = False

    # Post game
    game_summary = True


class DanishPuppet(BaseAgent):
    ACTIONS = AllActions()
    BASIC_ACTIONS = AllActions(move=(1,), strafe=())

    LEAVE_TIME = 100

    def __init__(self, name, helmets, visualizer=None, wait_for_pig=True, use_markov=True):
        super(DanishPuppet, self).__init__(name, len(DanishPuppet.ACTIONS), visualizer)
        self._previous_target_pos = None
        self._action_list = []

        # Other fields
        self.manual = False
        self.first_act_call = True

        # Features and memory
        self.game_features_history = []
        self.game_observer = GameObserver()
        self.game_features = FeatureSequence()
        self.game_summary_history = Queue(maxsize=SUMMARY_HISTORY_SIZE)

        # Timing and iterations
        self.n_moves = -1
        self.n_games = 0
        self.n_total_moves = 0
        self.time_start = time.time()

        # In-game timer
        self.game_timer = GameTimer()

        # Pig stuff
        self.waiting_for_pig = False
        self.wait_for_pig_if_necessary = wait_for_pig
        self.pig_wait_counter = 0

        # Image analysis
        self.initial_waits = True
        self.helmet_detector = HelmetDetector()
        self.helmet_detector.load_classifier()

        # Decision making
        self.brain = Brain(use_markov=use_markov,
                           helmets=helmets,
                           iterations_per_training=10)

    def time_alive(self):
        return time.time() - self.time_start

    def note_game_end(self, reward_sequence, state):
        if self.game_summary_history.full():
            self.game_summary_history.get()

        prize = int(max(reward_sequence))
        reward = int(sum(reward_sequence))

        game_summary = GameSummary(feature_sequence=self.game_features,
                                   reward=reward,
                                   prize=prize,
                                   final_state=state,
                                   pig_is_caught=GameObserver.was_pig_caught(prize=prize))

        self.game_summary_history.put(game_summary)
        if Print.game_summary:
            print("\nGame Summary {}:".format(len(self.game_features_history)))
            print("   {}".format(game_summary))
            print("   From reward-sequence: {}".format(reward_sequence))

        # self.game_features_history.append(self.game_features)
        # if len(self.game_features_history) > SAMPLES_IN_MEMORY:
        #     print("Storing data-history.")
        #     folder_path = Path("..", "data_dumps")
        #     ensure_folder(folder_path)
        #     pickle.dump(self.game_features_history, Path(folder_path, "game_features_history.p").open("wb"))
        #     self.game_features_history = random.sample(self.game_features_history,
        #                                                int(SAMPLES_IN_MEMORY * 0.95))

    def act(self, state, reward, done, total_time=None, is_training=False, frame=None):

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
                self.game_timer.reset()

                self.n_games += 1

                if Print.real_time_timing:
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

        # Game time
        self.game_timer.update(total_time=total_time)

        if Print.game_timing:
            print("\nTime left: {}".format(self.game_timer.time_left))

        ###############################################################################
        # If pig is not in a useful place - wait
        print_if(Print.code_line_print, "CODE: Considering pregame-wait")

        if self.initial_waits:
            print("\nHold your horses.\n")
            time.sleep(SECONDS_WAIT_BEFORE_IMAGE)
            self.initial_waits = False

            # Do training
            self.brain.train(game_features_history=self.game_features_history)

            # Send wait action to get a refreshed frame
            return DanishPuppet.ACTIONS.wait

        ###############################################################################
        # Get information from environment
        print_if(Print.code_line_print, "CODE: Information parsing")

        if state is None:
            return AllActions.jump

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
            print("Steps to challenger: {}".format(GamePlanner.directional_steps_to_other(me, challenger)))

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
                    print("      {}".format(plan.path_str()))

        if Print.challenger_plans:
            print("\nChallenger {} plans:".format(len(challengers_plans)))
            for plan in challengers_plans:
                print("   {}".format(plan))
                if Print.detailed_plans:
                    print("      {}".format(plan.path_str()))

        # Pig plans for both agents
        own_pig_plans = [plan for plan in own_plans if plan.target == CellGoalType.PigCatch]
        challenger_pig_plans = [plan for plan in challengers_plans if plan.target == CellGoalType.PigCatch]

        ###############################################################################
        # If no time left - leave game
        print_if(Print.code_line_print, "CODE: Considering to leave if clock runs out")

        if self.game_timer.time_left < DanishPuppet.LEAVE_TIME:
            print("Time's running out! Getting out of here!")

            # Exit plans
            exit_plans = [plan for plan in own_plans
                          if plan.target == CellGoalType.Exit]

            # Find nearest exit
            path = sorted(exit_plans, key=lambda x: -x.utility)[0]

            # Return next action
            action = path[1].action
            return action

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
        if current_challenger not in self.brain.helmets:
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

        challenger_strategy = self.brain.infer_challenger_strategy(game_features=self.game_features,
                                                                   own_plans=own_plans,
                                                                   verbose=Print.challenger_strategy,
                                                                   helmet_and_prob=(current_challenger,
                                                                                    helmet_probabilities[
                                                                                        current_challenger]))

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

        # If challenger is an idiot or a bad-guy - backstab him!
        if challenger_strategy in {Strategies.random_walker,
                                   Strategies.bad_guy}:
            if Print.challenger_strategy:
                print("Challenger seems to be an idiot.")

            # Exit plan
            plan = sorted([plan for plan in own_plans if plan.target == CellGoalType.Exit],
                          key=lambda x: -x.utility)[0]

            # Return next action (0th element is current position)
            action = plan[1].action
            return action

        # If optimistic - go to the pig
        elif challenger_strategy in {Strategies.initial_round,
                                     Strategies.naive_cooperative,
                                     Strategies.optimal_cooperative,
                                     Strategies.irrelevant}:
            if Print.challenger_strategy:
                if challenger_strategy == Strategies.initial_round:
                    print("First round. Challenger is assumed compliant.")
                elif challenger_strategy == Strategies.irrelevant:
                    print("Catching pig on my own!")
                else:
                    print("Challenger seems compliant.")

            # Check if pig can be caught by one person - then catch it!
            if challenger_strategy == Strategies.irrelevant:
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
                return GamePlanner.directional_wait_action(entity=me,
                                                           other_position=challenger)

            # Return next action (0th element is current position)
            action = own_pig_plan[1].action
            return action
