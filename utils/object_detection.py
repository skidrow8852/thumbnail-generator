import torch
import torchvision.models.detection as detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

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
