import torch 
import cv2
import numpy as np
device='cpu'
import torch.nn as nn 
class Convolutional_NN(nn.Module):
    def __init__(self, channels):
        super().__init__()  # ALWAYS FIRST
        self.Convolutional_Block = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),   # 128 -> 64
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),   # 64 -> 32
        )
        # 64 channels * 32 * 32 = 65536
        self.Classification_Block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),  # CORRECT size for 128x128 input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
            # NO Sigmoid here — CrossEntropyLoss handles it internally
        )

    def forward(self, x):
        x = self.Convolutional_Block(x)
        x = self.Classification_Block(x)
        return x
Model=Convolutional_NN(3)    
def load_model():
    Model.load_state_dict(torch.load("chest_xray_model.pth",map_location='cpu'))
    Model.eval()
    return Model

def pipeline(pil_image, model):
    # Ensure PIL image is RGB (model expects 3 channels)
    try:
        mode = pil_image.mode
    except Exception:
        raise ValueError("pipeline expects a PIL.Image input")
    if mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image = np.array(pil_image)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    # Match input dtype to model parameters to avoid dtype mismatch
    param_dtype = next(model.parameters()).dtype
    image = torch.tensor(image, dtype=param_dtype)
    image = image.unsqueeze(0).permute(0, 3, 1, 2).to(device=device, dtype=param_dtype)
    with torch.no_grad():
        logits = model.forward(image)
        probs = nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    return prediction.item(), confidence.item() * 100
    