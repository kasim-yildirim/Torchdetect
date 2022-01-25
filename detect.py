import torchvision
import cv2
from PIL import Image
from torchvision import transforms as T
import coco_classes

model_pretrained = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Inference
model_pretrained.eval()

print(coco_classes.classes)


def get_prediction(img_path, threshold):
    img = Image.open(img_path)  # Load the image
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    pred = model_pretrained([img])  # Pass the image to the model
    pred_class = [coco_classes.classes[i] for i in list(pred[0]['labels'].numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
        -1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=2, text_th=3):
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
    cv2.imshow('image', img)  # Display the output image
    cv2.waitKey(0)
    cv2.destroyAllWindows()


object_detection_api('images/1.jpg', threshold=0.8)
