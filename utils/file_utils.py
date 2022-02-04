import os


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)
