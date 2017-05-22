from pathlib2 import Path


class Paths:
    brain_training_data = Path("..", "danish_puppeteers", "results", "training_data")
    brain_models = Path("..", "danish_puppeteers", "results", "brains")
    helmet_data = Path("..", "danish_puppeteers", "results", "helmet_data")
    helmet_training_data = Path("..", "danish_puppeteers", "results", "helmet_data", "raw_data")


def get_dir(path):
    """
    Returns the directory of a file, or simply the original path if the path is a directory (has no extension)
    :param Path path:
    :return: Path
    """
    extension = path.suffix
    if extension == '':
        return path
    else:
        return path.parent


def ensure_folder(*arg):
    """
    Ensures the existence of a folder. If the folder does not exist it is created, otherwise nothing happens.
    :param str | Path arg: Any number of strings of Path-objects which can be combined to a path.
    """
    if len(arg) == 0:
        raise Exception("No input to ensure_folder")
    path = get_dir(Path(*arg))
    path.mkdir(parents=True, exist_ok=True)
