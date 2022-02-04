import random
from PIL import Image
from matplotlib import pyplot as plt
from utils.datasets import numpy_to_torch, torch_to_numpy, coco_classes
from utils.object_counter import counter
from utils.vis import object_visulization
from utils.img_utils import read_image, save_image, resize_image
import cv2
import torchvision
import torch
import numpy as np
from torchvision import transforms as T


def object_detect(image, model, threshold=0.5):
    prediction = model([image])
    print(prediction)
    image = torch_to_numpy(image)  # görselleştirme için resimler numpy formatında olmalıdır.
    prediction_class = [coco_classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_boxes = prediction_boxes[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]
    class_names = prediction_class[:prediction_thresh + 1]
    return image, prediction_boxes, class_names, prediction_score


def instance_segmentation_detect(image, model, threshold=0.5):
    prediction = model([image])
    print(prediction)
    image = torch_to_numpy(image)  # görselleştirme için resimler numpy formatında olmalıdır.
    prediction_class = [coco_classes[i] for i in list(prediction[0]['labels'].numpy())]
    prediction_segmentation = list(prediction[0]['masks'].detach().numpy())
    prediction_score = list(prediction[0]['scores'].detach().numpy())
    prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
    prediction_segmentation = prediction_segmentation[:prediction_thresh + 1]
    prediction_class = prediction_class[:prediction_thresh + 1]
    class_names = prediction_class[:prediction_thresh + 1]
    return image, prediction_segmentation, class_names, prediction_score


def get_coloured_mask(mask):
    """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [coco_classes[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class


def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
  segment_instance
    parameters:
      - img_path - path to input image
      - confidence- confidence to keep the prediction or not
      - rect_th - rect thickness
      - text_size
      - text_th - text thickness
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def predictions(image, model, threshold=0.5):
    image, prediction_boxes, class_names, prediction_score = object_detect(image, model, threshold)
    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        counter(image, class_names)
        object_visulization(image, box, class_names, score, i)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load('mask.pth'))
    model.eval()
    image = read_image("images/3.jpg")
    image = numpy_to_torch(image)  # modele giren resimler tensor formatında olmalıdır.
    # predictions(image, model, threshold=0.7)
    segment_instance("images/3.jpg")
