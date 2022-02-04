import cv2


def object_visulization(image, box, class_names, score, i):
    labels = '%{} {}'.format(float(int(score * 100)), class_names[i])
    x, y, w, h = box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(image, (x, y), (int(x + 100 + (len(class_names[i]) - 4) * 10), int(y - 30)), (0, 0, 255), -1)
    cv2.putText(image, labels, (int(x + 5), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
