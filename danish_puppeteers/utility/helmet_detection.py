import cPickle as pickle
import colorsys
import random

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
from sklearn import svm
import warnings


# Helmet names
HELMET_NAMES = [
    "iron_helmet",
    "golden_helmet",
    "diamond_helmet",
    "leather_helmet"
]


class HelmetDetector:
    def __init__(self, retrain=False, storage_path="../danish_puppeteers/storage"):
        self.hats = list(range(4))
        self.classifier = None

        # Get base-sky
        path = Path(storage_path, "base_sky.p")
        (_, _, _), _, _, frame = pickle.load(path.open("rb"))
        self.base_frame = self.to_matrix(frame)
        self.base_sky = self.get_sky(frame)

        # If set to train
        if retrain:
            self.train_from_path()

    def train_from_path(self, data_path=None):
        if data_path is None:
            data_path = Path("..", "data")

        # Folders with data
        hat_folders = [Path(data_path, "hat_{}".format(idx))
                       for idx in self.hats]

        # Get features of all helmets
        hat_class = []
        hat_vectors = []
        for hat in self.hats:
            folder = hat_folders[hat]

            # Get data-files
            file_paths = [path for path in folder.glob("*.p")]

            # Mean feature-vector
            mean_vector = None

            # Go through files
            for path in file_paths:
                _, _, _, frame = pickle.load(path.open("rb"))

                # Get helmet features
                current_features = self.get_helmet_features(frame, self.base_sky)

                # Check if helmet was found
                if current_features is not None:
                    hat_class.append(hat)
                    hat_vectors.append(current_features)

        # Turn into array
        hat_vectors = np.array(hat_vectors)

        # Train classifier
        self.classifier = svm.SVC(probability=True)
        self.classifier.fit(hat_vectors, hat_class)

    def store_classifier(self, path=None):
        if path is None:
            path = Path("..", "danish_puppeteers", "storage", "helmet_classifier.p")
        pickle.dump(self.classifier, path.open("wb"))

    def load_classifier(self, path=None):
        if path is None:
            path = Path("..", "danish_puppeteers", "storage", "helmet_classifier.p")
        self.classifier = pickle.load(path.open("rb"))

    def predict_helmet(self, frame):
        # Get features
        current_features = self.get_helmet_features(frame, self.base_sky)

        # Check if helmet was found in frame
        if current_features is None:
            return None, None

        else:
            # Ensure shape
            current_features = np.expand_dims(current_features, axis=0)

            # Predict with probabilities
            probabilities = self.classifier.predict_proba(current_features)
            most_probable_hat = np.argmax(probabilities)

            # Return
            return most_probable_hat, probabilities

    @staticmethod
    def to_matrix(a_frame):
        if not isinstance(a_frame, np.ndarray):
            return np.asarray(a_frame.convert(matrix='RBG'))
        return a_frame

    @staticmethod
    def get_sky(a_frame):
        matrix = HelmetDetector.to_matrix(a_frame=a_frame)
        return matrix[:50, :, :]

    @staticmethod
    def get_helmet_pixels(frame, base_sky):
        frame = HelmetDetector.to_matrix(frame)
        sky = HelmetDetector.get_sky(frame)

        helmet_detected = False

        # Get sky-difference
        sky_diff = sky - base_sky
        abs_sky_diff = abs(sky_diff)
        c_diff = abs_sky_diff.mean(0).mean(0)

        # Detect object in image
        if sum(c_diff) > 1:
            helmet_detected = True

        if helmet_detected:
            # Get region of helmet
            region = np.where(abs_sky_diff.sum(axis=2) > 10)

            helmet_pixels = sky_diff[region]
        else:
            helmet_pixels = None

        return helmet_pixels

    @staticmethod
    def get_helmet_features(frame, base_sky):
        helmet_pixels = HelmetDetector.get_helmet_pixels(frame, base_sky)

        if helmet_pixels is not None:
            helmet_rgb = helmet_pixels.mean(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                helmet_hsv = np.array([colorsys.rgb_to_hsv(*item) for item in helmet_pixels])
            helmet_hsv = helmet_hsv.mean(axis=0)
            helmet_features = np.hstack([helmet_rgb, helmet_hsv])

        else:
            helmet_features = None

        return helmet_features


if __name__ == "__main__":

    ################################################################################################
    # Training classifier

    detector = HelmetDetector(retrain=True)
    detector.store_classifier()

    ###########################################################
    # Testing

    plt.close("all")
    main_data_path = Path("..", "data")
    hat_folders = [Path(main_data_path, "hat_{}".format(idx))
                   for idx in range(4)]
    files = [c_file
             for path in hat_folders
             for c_file in path.glob("*.p")]

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(detector.base_frame)
    plt.title("Base Image")

    plt.subplot(2, 2, 3)
    plt.imshow(detector.base_sky)
    plt.title("Base Sky")

    # Create idx-generator
    all_idxs = range(len(files))
    random.shuffle(all_idxs)
    all_idxs = (val for val in all_idxs)

    # Helmet examples
    helmet_examples = [
        Path(r"..\data\hat_0\file_21.p"),
        Path(r"..\data\hat_1\file_8.p"),
        Path(r"..\data\hat_2\file_16.p"),
        Path(r"..\data\hat_3\file_3.p")
    ]

    #########################
    # Next frame

    # Plot next detected image and sky
    idx = next(all_idxs)
    print(files[idx])
    blaf = pickle.load(files[idx].open("rb"))
    frame = blaf[-1]
    frame = detector.to_matrix(frame)
    sky = detector.get_sky(frame)

    most_probable_hat, probabilities = detector.predict_helmet(frame)

    # Plot image
    ax = plt.subplot(2, 2, 2)  # type: plt.Axes
    ax.cla()
    plt.imshow(frame)

    title = "Image without helmet"
    plt.title(title)
    if most_probable_hat is not None:
        title = "Image with helmet {} \n({})".format(most_probable_hat, HELMET_NAMES[most_probable_hat])

        plt.title(title)

        plt.subplot(2, 2, 4)  # type: plt.Axes
        blaf = pickle.load(helmet_examples[most_probable_hat].open("rb"))
        frame = blaf[-1]
        frame = detector.to_matrix(frame)
        plt.imshow(frame)
        plt.title("Example of same hat.")

    else:
        ax = plt.subplot(2, 2, 4)  # type: plt.Axes
        ax.cla()

