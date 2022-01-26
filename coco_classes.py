import cv2
import numpy as np
import torch

classes = (
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)


def read_image(image, img_size=416):
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (img_size, img_size))

        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
            img = cv2.resize(img, (img_size, img_size))

        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.resize(img, (img_size, img_size))

    img = numpy_to_torch(img)

    return img


def numpy_to_torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img
