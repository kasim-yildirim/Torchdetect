import cv2
import numpy as np


def read_image(img):
    if type(img) == str:
        img = cv2.imread(img)

    elif type(img) == bytes:
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(img) == np.ndarray:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = img

        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

    return img


def save_image(img, path):
    from file_utils import create_dir
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    create_dir(path)
    cv2.imwrite(path, img)


def resize_image(image, long_size, interpolation=cv2.INTER_LINEAR):
    height, width, channel = image.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    return image
