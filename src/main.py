import os
import argparse
import torch
from models.pair_model import PAIRModel, TextExtractor, detect_adversarial_image, extract_hidden_text
from utils.image_processing import (
    load_image,
    apply_perturbation,
    detect_edges,
    extract_text_regions,
    visualize_results
)

def process_image(image_path, model, text_extractor, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process a single image through the adversarial detection pipeline
    
    Args:
        image_path: Path to the input image
        model: PAIRModel instance
        text_extractor: TextExtractor instance
        device: Device to run inference on
        
    Returns:
        dict: Results containing detection and text extraction information
    """
    # Load and preprocess image
    image = load_image(image_path)
    image = image.to(device)
    
    # Detect if image is adversarial
    is_adversarial, confidence = detect_adversarial_image(model, image)
    
    # Extract text regions
    text_regions = extract_text_regions(image)
    
    # Extract hidden text if image is adversarial
    hidden_text = None
    if is_adversarial:
        text_features = extract_hidden_text(text_extractor, image)
        hidden_text = text_features  # In a real implementation, this would be decoded to text
    
    # Visualize results
    visualization = visualize_results(image, is_adversarial, confidence, text_regions)
    
    return {
        'is_adversarial': is_adversarial,
        'confidence': confidence,
        'text_regions': text_regions,
        'hidden_text': hidden_text,
        'visualization': visualization
    }

def main():
    parser = argparse.ArgumentParser(description='Adversarial Image Detection using PAIR Method')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_path', type=str, default='models/pair_model.pth', help='Path to saved model')
    parser.add_argument('--text_extractor_path', type=str, default='models/text_extractor.pth',
                       help='Path to saved text extractor model')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAIRModel().to(device)
    text_extractor = TextExtractor().to(device)
    
    # Load saved models if they exist
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    if os.path.exists(args.text_extractor_path):
        text_extractor.load_state_dict(torch.load(args.text_extractor_path, map_location=device))
    
    # Process single image or directory
    if os.path.isfile(args.input_path):
        results = process_image(args.input_path, model, text_extractor, device)
        
        # Save visualization
        output_file = os.path.join(args.output_path, 'result_' + os.path.basename(args.input_path))
        cv2.imwrite(output_file, results['visualization'])
        
        # Print results
        print(f"\nResults for {args.input_path}:")
        print(f"Adversarial: {results['is_adversarial']}")
        print(f"Confidence: {results['confidence']:.2f}")
        if results['hidden_text'] is not None:
            print("Hidden text detected!")
            print(f"Text features: {results['hidden_text']}")
    
    else:
        # Process all images in directory
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input_path, filename)
                results = process_image(image_path, model, text_extractor, device)
                
                # Save visualization
                output_file = os.path.join(args.output_path, 'result_' + filename)
                cv2.imwrite(output_file, results['visualization'])
                
                # Print results
                print(f"\nResults for {filename}:")
                print(f"Adversarial: {results['is_adversarial']}")
                print(f"Confidence: {results['confidence']:.2f}")
                if results['hidden_text'] is not None:
                    print("Hidden text detected!")
                    print(f"Text features: {results['hidden_text']}")

if __name__ == '__main__':
    main() 