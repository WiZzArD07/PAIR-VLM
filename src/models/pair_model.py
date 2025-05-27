import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class PAIRModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PAIRModel, self).__init__()
        
        # Load pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Perturbation analysis layers
        self.perturbation_net = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features from backbone
        features = self.feature_extractor(x)
        
        # Apply perturbation analysis
        perturbation_features = self.perturbation_net(features)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        perturbation_features = perturbation_features.view(perturbation_features.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([features, perturbation_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

class TextExtractor(nn.Module):
    def __init__(self):
        super(TextExtractor, self).__init__()
        
        # CNN layers for text detection
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Text recognition layers
        self.text_recognition = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        # Extract features
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        # Text recognition
        text_features = self.text_recognition(features)
        
        return text_features

def detect_adversarial_image(model, image, threshold=0.5):
    """
    Detect if an image is adversarial using the PAIR model
    
    Args:
        model: PAIRModel instance
        image: Input image tensor
        threshold: Classification threshold
        
    Returns:
        bool: True if image is adversarial, False otherwise
        float: Confidence score
    """
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
        is_adversarial = prediction.item() == 1
        confidence_score = confidence.item()
        
        return is_adversarial, confidence_score

def extract_hidden_text(text_extractor, image):
    """
    Extract hidden text from an image
    
    Args:
        text_extractor: TextExtractor instance
        image: Input image tensor
        
    Returns:
        str: Extracted text
    """
    text_extractor.eval()
    with torch.no_grad():
        text_features = text_extractor(image)
        # Convert features to text (simplified version)
        # In a real implementation, this would use a more sophisticated text decoding method
        return text_features.cpu().numpy() 