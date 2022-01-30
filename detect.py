from datasets import *
import cv2
import torchvision
import torch
from collections import Counter


def predictions(image, model, threshold=0.5):
    prediction = model([image])
    image = torch_to_numpy(image)  # görselleştirme için resimler numpy formatında olmalıdır.
    prediction_class = [coco_classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]

    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        int_score = int(score * 100)
        percent_score = str(int_score)

        #Counting Classes
        class_names = prediction_class[:prediction_thresh + 1]
        class_counts=Counter(class_names)
        class_counts = dict(class_counts)
        res = list(sum(sorted(class_counts.items(), key = lambda x:x[1]), ()))
        class_names2 = list(dict.fromkeys(class_names))


        labels = '%{} {}'.format(float(percent_score), class_names[i])
        x, y, w, h = box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (int(x+100+(len(class_names[i])-4)*10) , int(y-30) ), (0, 0, 255), -1)
        cv2.putText(image, labels, (int(x+5), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    for j in range(len(class_names2)):
        cv2.putText(image, str(class_names2[j])+":"+str(class_counts[class_names2[j]]), (j,j*40+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    image = read_image("images/3.jpg")
    image = numpy_to_torch(image) # modele giren resimler tensor formatında olmalıdır.
    predictions(image, model, threshold=0.7)