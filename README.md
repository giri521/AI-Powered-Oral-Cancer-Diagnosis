# AI-Powered Oral Cancer Diagnosis

## Overview

AI-Powered Oral Cancer Diagnosis is an artificial intelligence-driven application designed to assist in the early detection of oral cancer using image analysis. It leverages deep learning models to classify oral images as cancerous or non-cancerous, providing clinicians with a supportive diagnostic tool.

## Features

* Upload oral images for AI-based classification.
* Trained convolutional neural network models (e.g., EfficientNet, ConvNeXt).
* Display predictions with confidence scores.
* Web interface using Flask for easy user interaction.
* Optional visual explainability (Grad-CAM) highlighting important regions.

## Technologies Used

* Python
* Flask (web framework)
* PyTorch / TensorFlow / Keras (deep learning)
* OpenCV / PIL (image processing)
* NumPy, Pandas
* Grad-CAM or similar for visualization (optional)

## File Structure

```
AI-Powered-Oral-Cancer-Diagnosis/
│
├── app.py                 # Flask web application
├── train_efficientnet.py  # Script to train EfficientNet model
├── train_convnext.py      # Script to train ConvNeXt model
├── test.py                # Script for testing and evaluation
├── model/                 # Trained model weights
├── templates/             # HTML templates for UI
├── static/                # CSS, JS, images
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## Getting Started

### Prerequisites

* Python 3.x
* pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/giri521/AI-Powered-Oral-Cancer-Diagnosis.git
   ```
2. Navigate into the project folder:

   ```bash
   cd AI-Powered-Oral-Cancer-Diagnosis
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Ensure trained model weights are in the `model/` folder.

### Training the Models

* To train EfficientNet:

```bash
python train_efficientnet.py
```

* To train ConvNeXt:

```bash
python train_convnext.py
```

### Running the Flask App

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

## Usage

1. Upload an oral image on the web interface.
2. Submit the image to get AI prediction (Cancerous / Non-cancerous).
3. Optionally view attention/heatmap regions.
4. Use predictions as supportive information for clinical decisions.

## Future Enhancements

* Multi-class classification (stages or subtypes).
* Advanced data augmentation.
* Ensemble learning or model fusion for improved accuracy.
* Mobile app or REST API integration.
* Expand dataset with diverse demographics and imaging types.
* User authentication and logging for clinical workflow.

