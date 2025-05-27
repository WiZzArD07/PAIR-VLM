import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from src.models.pair_model import PAIRModel
from src.utils.image_processing import apply_perturbation

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def generate_clean_image(size=(224, 224), text="Clean Image"):
    """Generate a clean image with text"""
    # Create a white background
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Convert to PIL Image for text
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position for center alignment
    text_width = draw.textlength(text, font=font)
    text_position = ((size[1] - text_width) // 2, size[0] // 2)
    
    # Draw text
    draw.text(text_position, text, fill=(0, 0, 0), font=font)
    
    return np.array(pil_image)

def generate_adversarial_image(size=(224, 224), epsilon=0.03):
    """Generate an adversarial image using FGSM attack"""
    # Create a clean image
    clean_image = generate_clean_image(size, "Adversarial Image")
    
    # Convert to tensor
    image_tensor = torch.from_numpy(clean_image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Apply perturbation
    perturbed_image = apply_perturbation(image_tensor, epsilon)
    
    # Convert back to numpy
    perturbed_image = perturbed_image.squeeze(0).permute(1, 2, 0).numpy()
    perturbed_image = (perturbed_image * 255).astype(np.uint8)
    
    return perturbed_image

def generate_hidden_text_image(size=(224, 224)):
    """Generate an image with hidden text using steganography"""
    # Create base image
    image = generate_clean_image(size, "Hidden Text")
    
    # Convert to float for manipulation
    image_float = image.astype(np.float32) / 255.0
    
    # Create hidden text pattern
    text_pattern = np.zeros_like(image_float)
    text_pattern[50:100, 50:150, :] = 0.1  # Subtle pattern
    
    # Add hidden text
    image_with_hidden = image_float + text_pattern
    image_with_hidden = np.clip(image_with_hidden, 0, 1)
    
    # Convert back to uint8
    return (image_with_hidden * 255).astype(np.uint8)

def generate_noise_image(size=(224, 224), noise_level=0.1):
    """Generate an image with random noise"""
    # Create base image
    image = generate_clean_image(size, "Noisy Image")
    
    # Add random noise
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def generate_test_dataset():
    """Generate a complete test dataset"""
    # Create directories
    base_dir = "data/test_images"
    clean_dir = os.path.join(base_dir, "clean")
    adversarial_dir = os.path.join(base_dir, "adversarial")
    hidden_text_dir = os.path.join(base_dir, "hidden_text")
    noisy_dir = os.path.join(base_dir, "noisy")
    
    for directory in [clean_dir, adversarial_dir, hidden_text_dir, noisy_dir]:
        create_directory(directory)
    
    # Generate clean images
    for i in range(5):
        image = generate_clean_image()
        cv2.imwrite(os.path.join(clean_dir, f"clean_{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate adversarial images
    for i in range(5):
        image = generate_adversarial_image(epsilon=0.03 + i*0.01)
        cv2.imwrite(os.path.join(adversarial_dir, f"adversarial_{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate hidden text images
    for i in range(5):
        image = generate_hidden_text_image()
        cv2.imwrite(os.path.join(hidden_text_dir, f"hidden_text_{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate noisy images
    for i in range(5):
        image = generate_noise_image(noise_level=0.1 + i*0.05)
        cv2.imwrite(os.path.join(noisy_dir, f"noisy_{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print("Test dataset generated successfully!")
    print(f"Total images generated: 20")
    print(f"Clean images: 5")
    print(f"Adversarial images: 5")
    print(f"Hidden text images: 5")
    print(f"Noisy images: 5")
    print(f"\nImages are saved in: {base_dir}")

if __name__ == "__main__":
    generate_test_dataset() 