from pathlib2 import Path


class Paths:
    results_root_path = Path("..","danish_puppeteers","results")
    resources_root_path = Path("..","danish_puppeteers","resources")

    brain_training_data_save = Path(results_root_path, "training_data")
    brain_training_data_load = Path(results_root_path, "remoteResults")

    brain_models = Path(resources_root_path, "brains")
    helmet_data = Path(resources_root_path, "helmet_data")
    helmet_training_data = Path(resources_root_path, "helmet_data", "raw_data")


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
