# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from threading import Thread, active_count
from time import sleep

from bad_guy import BadGuy
from challenger_factory import ChallengerFactory
from danish_puppet import DanishPuppet
from malmopy.agent import RandomAgent
from standstill_agent import StandstillAgent

try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

BASELINES_FOLDER = 'results/baselines/pig_chase/%s/%s'
EPOCH_SIZE = 100

MANUAL = False
HUMAN_SPEED = False
WAIT_FOR_PIG = True

PASS_FRAME = True

AGENT_TYPE = {
    RandomAgent: PigChaseEnvironment.AGENT_TYPE_0,
    StandstillAgent: PigChaseEnvironment.AGENT_TYPE_1,
    BadGuy: PigChaseEnvironment.AGENT_TYPE_2
}


def get_agent_type(agent):
    return AGENT_TYPE.get(type(agent),
                          PigChaseEnvironment.AGENT_TYPE_3)


def agent_factory(name, role, clients, max_epochs,
                  logdir, visualizer, manual=False):
    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)

    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder,
                              actions=DanishPuppet.ACTIONS.all_commands(),
                              role=role,
                              human_speed=HUMAN_SPEED,
                              randomize_positions=True)

    # Challenger  (Agent_1)
    if role == 0:
        agent = ChallengerFactory(name, focused=True, random=True, bad_guy=False)

        agent_type = get_agent_type(agent.current_agent)
        state = env.reset(agent_type)

        reward = 0
        agent_done = False

        while True:

            # select an action
            action = agent.act(state, reward, agent_done, is_training=True)

            # reset if needed
            if env.done:
                agent_type = get_agent_type(agent.current_agent)
                state = env.reset(agent_type)

            # take a step
            state, reward, agent_done = env.do(action)

    # Our Agent (Agent_2)
    else:
        agent = DanishPuppet(name=name,
                             wait_for_pig=WAIT_FOR_PIG)

        # Manual overwrite!
        if manual:
            agent.manual = True

        state = env.reset()
        reward = 0
        agent_done = False
        viz_rewards = []

        max_training_steps = EPOCH_SIZE * max_epochs
        for step in range(1, max_training_steps + 1):

            # check if env needs reset
            if env.done:
                agent.note_game_end(reward_sequence=viz_rewards,
                                    state=state[0])
                print("")
                visualize_training(visualizer, step, viz_rewards)
                viz_rewards = []
                state = env.reset()

            # select an action
            action = None
            frame = None if not PASS_FRAME else env.frame
            while action is None:
                # for key, item in env.world_observations.items():
                #     print(key, ":", item)

                action = agent.act(state, reward, agent_done,
                                   is_training=True,
                                   frame=frame)

                # 'wait'
                if action == DanishPuppet.ACTIONS.wait:
                    action = None
                    sleep(4e-3)
                    state = env.state

            # take a step
            state, reward, agent_done = env.do(action)
            viz_rewards.append(reward)

            agent.inject_summaries(step)


def run_experiment(agents_def):
    assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)' \
                                 % len(agents_def)

    processes = []
    for agent in agents_def:
        p = Thread(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')


if __name__ == '__main__':
    arg_parser = ArgumentParser('Danish Puppet Pig Chase experiment')
    arg_parser.add_argument('-t', '--type', type=str, default='danish',
                            choices=['astar', 'random'],
                            help='The type of baseline to run.')
    arg_parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')

    # Manual overwrite!
    manual = MANUAL

    args = arg_parser.parse_args()

    logdir = BASELINES_FOLDER % (args.type, datetime.utcnow().isoformat())
    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
    else:
        visualizer = ConsoleVisualizer()

    agents = [{'name': agent, 'role': role, 'manual': manual,
               'clients': args.clients, 'max_epochs': args.epochs,
               'logdir': logdir, 'visualizer': visualizer}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_experiment(agents)
