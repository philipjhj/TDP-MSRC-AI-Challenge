import cPickle as pickle
import colorsys
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
from sklearn import svm

from utility.ai import GamePlanner, EntityPosition
from utility.constants import HELMET_NAMES
from utility.util import Paths


class HelmetDetector:
    """
    Class used for detecting helmet in the game.
    Very domain-specific and trained on data from this game.
    Uses an SVM for classifying helmets and can only detect the four helmet found in the pig-chase environment. 
    """
    def __init__(self, retrain=False):
        self.hats = list(range(4))
        self.classifier = None
        self.helmet_probabilities = np.ones(4)

        # Get base-sky
        path = Path(Paths.helmet_data, "base_sky.p")
        (_, _, _), _, _, frame = pickle.load(path.open("rb"))
        self.base_frame = self.to_matrix(frame)
        self.base_sky = self.get_sky(frame)

        # If set to train
        if retrain:
            self.train_from_path()

    def reset(self):
        """
        Resets helmet probabilities. 
        """
        self.helmet_probabilities = np.ones(4)

    def train_from_path(self):
        """
        Trains helmet-detector from data-files in the designated path.
        """
        data_path = Paths.helmet_training_data

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
        """
        Stores currently trained classifier.
        :param Path path: 
        """
        if path is None:
            path = Path(Paths.helmet_data, "helmet_classifier.p")
        pickle.dump(self.classifier, path.open("wb"))

    def load_classifier(self, path=None):
        """
        Loads classifier from file.
        :param Path path: 
        """
        if path is None:
            path = Path(Paths.helmet_data, "helmet_classifier.p")
        self.classifier = pickle.load(path.open("rb"))

    def _helmet_probabilities(self, frame):
        """
        Computes the probabilities as given by the used classifier, for the various helmets. 
        :param np.array frame: 
        :return: int, np.array
        """
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
    def store_snapshot(me, challenger, pig, state, frame):
        """
        Stores a snapshot of the current situation (used for creating data for training model).
        :param EntityPosition me: 
        :param EntityPosition challenger: 
        :param EntityPosition pig: 
        :param np.array state: 
        :param np.array frame: 
        """
        storage_path = Paths.helmet_training_data

        # Find next file-id
        files_in_directory = [str(item.stem) for item in storage_path.glob("*.p")]
        try:
            next_file_number = max([int(item.replace("file_", ""))
                                    for item in files_in_directory]) + 1
        except ValueError:
            next_file_number = 0

        # Data to be stored
        data_file = ((me, challenger, pig), GamePlanner.directional_steps_to_other(me, challenger), state, frame)

        # Make data-dump
        pickle.dump(data_file, Path(storage_path, "file_{}.p".format(next_file_number)).open("wb"))

    @staticmethod
    def to_matrix(a_frame):
        """
        Converts PIL-object to numpy array.
        :param a_frame: 
        :return: np.array
        """
        if not isinstance(a_frame, np.ndarray):
            return np.asarray(a_frame.convert(matrix='RBG'))
        return a_frame

    @staticmethod
    def get_sky(a_frame):
        """
        
        :param a_frame: 
        :return: 
        """
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

    def detect_helmet(self, me, challenger, frame):
        if not GamePlanner.matches(me, challenger):
            # Detect helmet
            _, probabilities = self._helmet_probabilities(frame)

            # Check if helmet was seen
            if probabilities is not None:
                probabilities = np.array(probabilities)

                # Update probabilities of round
                self.helmet_probabilities = np.squeeze(self.helmet_probabilities * probabilities, axis=0)
                self.helmet_probabilities /= self.helmet_probabilities.sum()

        # Determine most probably helmet
        sorted_probabilities = sorted([val for val in self.helmet_probabilities])
        decision_made = not np.isclose(sorted_probabilities[-1], sorted_probabilities[-2])

        # Get most likely helmet
        if decision_made:
            current_challenger = np.argmax(self.helmet_probabilities)
        else:
            current_challenger = None

        return current_challenger, self.helmet_probabilities, decision_made




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
        Path(Paths.helmet_training_data, "hat_0", "file_21.p"),
        Path(Paths.helmet_training_data, "hat_1", r"file_8.p"),
        Path(Paths.helmet_training_data, "hat_2", r"file_16.p"),
        Path(Paths.helmet_training_data, "hat_3", r"file_3.p")
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

    most_probable_hat, probabilities = detector._helmet_probabilities(frame)

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

