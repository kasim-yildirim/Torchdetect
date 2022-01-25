import glob
import os
import torch
import coco_classes
import cv2
import torchvision


def detect(image_path, model, threshold=0.5):
    prediction = model([image_path])
    prediction_class = [coco_classes.classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        int_score = int(score * 100)
        percent_score = str(int_score)
        class_names = prediction_class[:prediction_thresh + 1]
        labels = '%{} {}'.format(percent_score, class_names[i])
        x, y, w, h = box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, labels, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def transforms_img(image_path, img_size=416):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_path = transform(image_path)
    image_path = image_path.unsqueeze(0)
    return image_path


if __name__ == '__main__':
    image_path = './images/1.jpg'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image_path = cv2.imread(image_path)
    image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image_path = transforms_img(image_path)
    detect(image_path, model)