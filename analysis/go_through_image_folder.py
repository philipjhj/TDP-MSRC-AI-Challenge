import random
import time
from pathlib2 import Path
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import sklearn
from sklearn.linear_model import LogisticRegression


def to_matrix(a_frame):
    if not isinstance(a_frame, np.ndarray):
        return np.asarray(a_frame.convert(matrix='RBG'))
    return a_frame


def get_sky(a_frame):
    matrix = to_matrix(a_frame=a_frame)
    return matrix[:50, :, :]


hats = list(range(4))

print("Working directory: {}".format(Path.cwd()))

main_data_path = Path("..", "data_dumps")

plt.close("all")

files = [c_file for c_file in main_data_path.glob("*.p")]

# Create idx-generator
all_idxs = range(len(files))
random.shuffle(all_idxs)
all_idxs = (val for val in all_idxs)

###########################################################
# Go through folder

fig = plt.figure()

idx = next(all_idxs)
print(files[idx])
blaf = pickle.load(files[idx].open("rb"))
frame = blaf[-1]
frame = to_matrix(frame)
sky = get_sky(frame)

plt.subplot(2, 1, 1)
plt.imshow(frame)
plt.title("Image")

plt.subplot(2, 1, 2)
plt.imshow(sky)
plt.title("Sky")

