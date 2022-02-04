from utils.datasets import numpy_to_torch, torch_to_numpy, coco_classes
from utils.object_counter import counter
from utils.vis import object_visulization
from utils.img_utils import read_image, save_image, resize_image
import cv2
import torchvision
import torch


def predictions(image, model, threshold=0.5):
    prediction = model([image])
    image = torch_to_numpy(image)  # görselleştirme için resimler numpy formatında olmalıdır.
    prediction_class = [coco_classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]
    class_names = prediction_class[:prediction_thresh + 1]
    return image, prediction_boxes, class_names, prediction_score


def object_detect(image, model, threshold=0.5):
    image, prediction_boxes, class_names, prediction_score = predictions(image, model, threshold)
    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        counter(image, class_names)
        object_visulization(image, box, class_names, score, i)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    image = read_image("images/3.jpg")
    image = numpy_to_torch(image)  # modele giren resimler tensor formatında olmalıdır.
    object_detect(image, model, threshold=0.7)
