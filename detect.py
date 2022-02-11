import torch
from utils.datasets import numpy_to_torch, torch_to_numpy, coco_classes
from utils.vis import object_visulization, segmentation_visulization
from utils.img_utils import read_image
import cv2
import torchvision


def object_detections(image, model, threshold=0.5):
    image = read_image(image)
    image = numpy_to_torch(image)
    prediction = model([image])
    image = torch_to_numpy(image)
    prediction_class = [coco_classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]
    return image, prediction_boxes, prediction_class, prediction_score


def instance_segmentations(img_path, confidence):
    img = read_image(img_path)
    img = numpy_to_torch(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    prediction_class = [coco_classes[i] for i in list(pred[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    prediction_boxes = prediction_boxes[:pred_t + 1]
    prediction_class = prediction_class[:pred_t + 1]
    prediction_score = pred_score[:pred_t + 1]
    return masks, prediction_boxes, prediction_class, prediction_score


def detections_predictions(image, threshold=0.7):

    image, prediction_boxes, class_names, prediction_score = object_detections(image, model, threshold)
    for i, box in enumerate(prediction_boxes):
        image = object_visulization(image, box, class_names, prediction_score, i)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentations_predictions(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    masks, boxes, prediction_class, prediction_score = instance_segmentations(img_path, confidence)
    img = read_image(img_path)
    for i in range(len(masks)):
        img = segmentation_visulization(img, masks, boxes, prediction_class,
                                        prediction_score, text_size, text_th, rect_th, i)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    from utils.file_utils import download
    RetinaNet:
    url = "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
    save_path = "../models/retinanet_resnet50_fpn_coco.pth"
    download(url, save_path)
    
    Faster-RCNN:
    from utils.file_utils import download
    url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    save_path = "../models/retinanet_resnet50_fpn_coco.pth"
    download(url, save_path)
    
    Mask-RCNN:
    from utils.file_utils import download
    url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    save_path = "../models/maskrcnn_resnet50_fpn_coco.pth"
    download(url, save_path)
    
    Download the model:

    """
    from utils.file_utils import download
    url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    save_path = "models/maskrcnn_resnet50_fpn_coco.pth"
    download(url, save_path)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load('models/maskrcnn_resnet50_fpn_coco.pth'))
    model.eval()
    #segmentations_predictions("images/3.jpg")  # instance segmentations
    #detections_predictions("images/3.jpg")  # object detections
