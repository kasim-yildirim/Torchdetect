import os
import gdown


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def download(url: str, save_path: str):
    """
    Downloads file from gdrive, shows progress.
    Example inputs:
        url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
        save_path: 'data/file.zip'
    """

    # create save_dir if not present
    create_dir(os.path.dirname(save_path))
    # download file
    gdown.download(url, save_path, quiet=False)
