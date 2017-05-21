from __future__ import division

import pickle
import random

from pathlib2 import Path

from danish_puppet import DanishPuppet
from utility.constants import AllActions, CellGoalType
from utility.minecraft import GameObserver
from utility.ml import FeatureSequence
from utility.util import ensure_folder


class Jumper(DanishPuppet):
    ActionMap = AllActions(move=(), turn=(1, -1), strafe=(), jump=True, wait=False)

    def __init__(self, name, helmets, visualizer=None):
        super(Jumper, self).__init__(name, helmets=helmets, visualizer=visualizer,
                                     wait_for_pig=False, use_markov=False)
        self.actions = [Jumper.ActionMap.turn_r, Jumper.ActionMap.jump, Jumper.ActionMap.jump]

    def act(self, state, reward, done, total_time=None, is_training=False, frame=None):
        if done:

            # Path for file
            folder_path = Path("..", "danish_puppeteers", "results", "training_data")
            ensure_folder(folder_path)

            # Files in folder
            file_base_name = "game_"
            files_in_directory = [str(item.stem) for item in folder_path.glob("*.p")]
            try:
                next_file_number = max([int(item.replace(file_base_name, ""))
                                        for item in files_in_directory]) + 1
            except ValueError:
                next_file_number = 0

            # Data to be stored
            data_file = (self.game_features)

            # Make data-dump
            pickle.dump(data_file, Path(folder_path, file_base_name + "{}.p".format(next_file_number)).open("wb"))

            self._action_list = []
            self._previous_target_pos = None
            self.game_features = FeatureSequence()
            self.n_moves = -1
            self.initial_waits = True
            self.game_observer.reset()
            self.helmet_detector.reset()
            self.game_timer.reset()

            self.n_games += 1

            self.first_act_call = False
        else:
            self.first_act_call = True

        if state is None:
            return AllActions.jump

        entities = state[1]
        state = state[0]
        self.n_moves += 1

        # Get positions from game
        me, challenger, pig = self.game_observer.create_entity_positions(state=state,
                                                                         entities=entities)

        # Specific targets
        pig_neighbours, exits = GameObserver.determine_targets(state=state,
                                                               pig=pig)

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

        # Pig plans for both agents
        own_pig_plans = [plan for plan in own_plans if plan.target == CellGoalType.PigCatch]
        challenger_pig_plans = [plan for plan in challengers_plans if plan.target == CellGoalType.PigCatch]

        # Check if on top of challenger
        current_challenger, helmet_probabilities, decision_made = \
            self.helmet_detector.detect_helmet(me=me,
                                               challenger=challenger,
                                               frame=frame)

        if current_challenger is None:
            current_challenger = -1

        self.game_features.update(own_plans=own_plans,
                                  challengers_plans=challengers_plans,
                                  current_challenger=current_challenger)

        self.n_total_moves += 1

        # Return action
        return random.choice(self.actions)
