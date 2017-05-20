import random

from hmmlearn import hmm
from hmmlearn.base import check_array, ConvergenceMonitor, iter_from_X_lengths
from hmmlearn.hmm import MultinomialHMM
import numpy as np
from itertools import product


# model = MultinomialHMM(n_components=2,
#                        n_iter=1)
#
# X = np.array([[1, 1, 1], [0, 0, 0, 0]])
# lengths = [len(row) for row in X]
# X = np.atleast_2d(np.concatenate(X)).T
#
# model.fit(X, lengths)


###############################################################################

# Data
n_helmets = 4
helmets = list(range(n_helmets + 1))
towards_pig = [0, 1]
towards_exit = [0, 1]

# Possible seen emissions from model
possible_emissions = []
decode_emission_idx = {}
for idx, (h, p, e) in enumerate(product(helmets, towards_pig, towards_exit)):
    possible_emissions.append(idx)
    decode_emission_idx[idx] = (h, p, e)
encode_emission = {code: idx for idx, code in decode_emission_idx.items()}

# Number of states and emissions
n_states = 2
n_emissions = len(possible_emissions)

# Training data
X = np.array([
    random.sample(possible_emissions, len(possible_emissions)),
    [1, 1, 2, 1],
    [6, 5, 5, 4, 7, 7]
])
lengths = [len(row) for row in X]
X = np.atleast_2d(np.concatenate(X))

# Create randomly initialized model
model = MultinomialHMM(n_components=2,
                           n_iter=100)

# Train on data
model.fit(X.T, lengths=lengths)

# Trained parameters
transition_matrix = model.transmat_
emission_matrix = model.emissionprob_
initial_stat_probability = model.startprob_

# Observed sequence
observed = np.array([1, 1, 1])

# Get helmet
helmet = 0

# Sequence emissions
helping_emission = (helmet, 1, 0)
backstabbing_emission = (helmet, 0, 1)

# Generated sequences
sequence_lengths = 9
helping_sequence = np.hstack([observed, np.array([encode_emission[helping_emission]] * sequence_lengths)])
backstabbing_sequence = np.hstack([observed, np.array([encode_emission[backstabbing_emission]] * sequence_lengths)])

# Flipping sequence
flip_sequence = []
for ____ in range(sequence_lengths // 2):
    flip_sequence.extend([encode_emission[helping_emission], encode_emission[backstabbing_emission]])
if (sequence_lengths % 2) > 0:
    flip_sequence.append(encode_emission[helping_emission])
flip_sequence = np.array(flip_sequence)

# Likelihoods
helping_likelihood = np.exp(model.score(helping_sequence))
backstabbing_likelihood = np.exp(model.score(backstabbing_sequence))
flip_likelihood = np.exp(model.score(flip_sequence))


