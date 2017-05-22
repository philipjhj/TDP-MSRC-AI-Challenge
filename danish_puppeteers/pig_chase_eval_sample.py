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
from pathlib2 import Path

from common import ENV_AGENT_NAMES
from danish_puppet import DanishPuppet
from environment import PigChaseSymbolicStateBuilder
from evaluation import PigChaseEvaluator
from utility.util import ensure_folder

if __name__ == '__main__':
    # Warn for Agent name !!!

    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    agent = DanishPuppet(ENV_AGENT_NAMES[1], helmets=[0, 1], use_markov=False)

    eval = PigChaseEvaluator(clients, agent, agent, PigChaseSymbolicStateBuilder())
    eval.run()

    folder_path = Path("..", "evaluations")
    ensure_folder(folder_path)
    eval.save('My Exp 1', Path(folder_path, "pig_chase_results.json"))
