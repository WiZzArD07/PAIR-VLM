# Adversarial Image Detection using PAIR Method

This project implements an advanced machine learning model for detecting adversarial images that could be used for jailbreaking AI models. The system uses the PAIR (Perturbation Analysis and Image Recognition) method to identify potential adversarial images and extract any hidden text within them.

## Features

- Detection of adversarial images
- Extraction of hidden text from images
- Robust perturbation analysis
- Support for various image formats
- Real-time detection capabilities

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── pair_model.py
│   │   └── text_extractor.py
│   ├── utils/
│   │   ├── image_processing.py
│   │   └── visualization.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── model_analysis.ipynb
├── tests/
│   └── test_detection.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `data/raw` directory
2. Run the detection script:
   ```bash
   python src/main.py --input_path data/raw --output_path data/processed
   ```

## Model Architecture

The PAIR method combines:
- Perturbation analysis for detecting adversarial patterns
- Deep learning-based image classification
- Text extraction and analysis
- Robust feature extraction

## License

MIT License 