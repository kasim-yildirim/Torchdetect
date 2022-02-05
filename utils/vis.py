import random
import cv2
import numpy as np
from utils.object_counter import counter


def object_visulization(image, box, prediction_class, prediction_score, i):
    score = prediction_score[i]
    counter(image, prediction_class)
    labels = '%{} {}'.format(float(int(score * 100)), prediction_class[i])
    x, y, w, h = box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(image, (x, y), (int(x + 100 + (len(prediction_class[i]) - 4) * 10), int(y - 30)), (0, 0, 255), -1)
    cv2.putText(image, labels, (int(x + 5), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image


def segmentation_visulization(image, masks, boxes, prediction_class, prediction_score, text_size, text_th, rect_th, i):
    score = prediction_score[i]
    counter(image, prediction_class)
    labels = '%{} {}'.format(float(int(score * 100)), prediction_class[i])
    rgb_mask = get_coloured_mask(masks[i])
    img = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
    cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img, labels, boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img


def get_coloured_mask(mask):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask
