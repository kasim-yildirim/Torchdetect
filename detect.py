from coco_classes import classes, read_image
import cv2
import torchvision


def detect(image, model, threshold=0.5):
    prediction = model([image])
    prediction_class = [classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]

    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        int_score = int(score * 100)
        percent_score = str(int_score)
        class_names = prediction_class[:prediction_thresh + 1]
        labels = '%{} {}'.format(percent_score, class_names[i])
        x, y, w, h = box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]
        image = image.numpy()
        image = image.transpose((1, 2, 0))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, labels, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = read_image("images/2.jpg", 512)
    detect(image, model, threshold=0.5)
