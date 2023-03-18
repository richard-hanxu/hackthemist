PATH_TO_WEIGHTS = './checkpoint.pth'
PATH_TO_LABELS = './labels.txt'

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict_label(fileobject):
    # Crop, center, normalize the image input
    transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    # The model is pretrained on Resnet-50
    resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(in_features=2048, out_features=101, bias=True)

    weights = torch.load(PATH_TO_WEIGHTS)
    resnet50.load_state_dict(weights)

    input = Image.open(fileobject)
    prediction = torch.argmax(resnet50(input.unsqueeze(0)), 1)

    with open (PATH_TO_LABELS, 'r') as file:
        lines = file.readlines()
        predicted_label = lines[prediction]

    return predicted_label

