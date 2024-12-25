import torch
import torchvision.models.detection as detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# COCO class labels (from COCO dataset)
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def load_model():
    model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    return model

def analyze_frame(frame, model):
    # Convert frame to tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    return predictions


def get_object_details(predictions, object_threshold):
    object_details = []
    
    # Ensure predictions is a dictionary
    if isinstance(predictions, dict):
        labels = predictions.get('labels', [])
        scores = predictions.get('scores', [])
        boxes = predictions.get('boxes', [])
        
        # Loop through the predictions and collect object details based on score threshold
        for label, score, box in zip(labels, scores, boxes):
            if score > object_threshold:
                object_details.append({
                    "label": label.item(),  
                    "score": score.item(),
                    "box": box.tolist() 
                })
                
    return object_details

