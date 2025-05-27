import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    """
    Load and preprocess an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(image)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

def apply_perturbation(image, epsilon=0.03):
    """
    Apply random perturbation to an image
    
    Args:
        image: Input image tensor
        epsilon: Maximum perturbation magnitude
        
    Returns:
        torch.Tensor: Perturbed image
    """
    noise = torch.randn_like(image) * epsilon
    perturbed_image = image + noise
    return torch.clamp(perturbed_image, 0, 1)

def detect_edges(image):
    """
    Detect edges in an image using Canny edge detection
    
    Args:
        image: Input image tensor
        
    Returns:
        numpy.ndarray: Edge map
    """
    # Convert tensor to numpy array
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    return edges

def extract_text_regions(image):
    """
    Extract potential text regions from an image
    
    Args:
        image: Input image tensor
        
    Returns:
        list: List of text region coordinates
    """
    # Convert tensor to numpy array
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        
        if 0.1 < aspect_ratio < 10 and area > 100:
            text_regions.append((x, y, w, h))
    
    return text_regions

def visualize_results(image, is_adversarial, confidence, text_regions=None):
    """
    Visualize detection results on the image
    
    Args:
        image: Input image tensor
        is_adversarial: Boolean indicating if image is adversarial
        confidence: Confidence score
        text_regions: List of text region coordinates
        
    Returns:
        numpy.ndarray: Image with visualization
    """
    # Convert tensor to numpy array and ensure proper format
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image for drawing
    vis_image = image_np.copy()
    
    # Draw detection result
    color = (0, 0, 255) if is_adversarial else (0, 255, 0)
    text = f"Adversarial: {is_adversarial} ({confidence:.2f})"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(vis_image, (10, 30 - text_height), (10 + text_width, 30 + 5), (255, 255, 255), -1)
    
    # Draw text
    cv2.putText(vis_image, text, (10, 30), font, font_scale, color, thickness)
    
    # Draw text regions if provided
    if text_regions:
        for x, y, w, h in text_regions:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert back to RGB for display
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    return vis_image 