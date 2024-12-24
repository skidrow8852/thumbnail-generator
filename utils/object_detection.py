import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def analyze_frame(frame, model):
    # Convert frame to tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    return predictions
