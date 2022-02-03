from collections import Counter
import cv2


def counter(image, class_names):
    class_counts = Counter(class_names)
    class_counts = dict(class_counts)
    class_names2 = list(dict.fromkeys(class_names))
    for j in range(len(class_names2)):
        cv2.putText(image, str(class_names2[j]) + ":" + str(class_counts[class_names2[j]]), (j, j * 40 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


