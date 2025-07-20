from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Model architecture (same as your training code)


class ConvNeXtFashionClassifier(nn.Module):
    def __init__(self, num_classes_dict, pretrained=True):
        super(ConvNeXtFashionClassifier, self).__init__()

        # Load pre-trained ConvNeXt-Tiny
        self.backbone = models.convnext_tiny(pretrained=pretrained)

        # Get the number of features from ConvNeXt-Tiny
        # 768 features for ConvNeXt-Tiny
        num_features = self.backbone.classifier[2].in_features

        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()

        # Freeze backbone for feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Shared dense layers optimized for ConvNeXt features
        self.shared_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )

        # Classification heads for each target variable
        self.classifiers = nn.ModuleDict()
        for target, num_classes in num_classes_dict.items():
            # Adjusted head sizes based on target complexity
            if target == 'articleType':  # Most complex target (143 classes)
                head_size = 256
            elif target == 'baseColour':  # Medium complexity (46 classes)
                head_size = 128
            else:  # season (4 classes) and gender (5 classes)
                head_size = 64

            self.classifiers[target] = nn.Sequential(
                nn.Linear(256, head_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(head_size),
                nn.Dropout(0.2),
                nn.Linear(head_size, num_classes)
            )

    def forward(self, x):
        # Extract features using ConvNeXt-Tiny backbone
        features = self.backbone(x)

        features = torch.flatten(features, 1)
        # Shared representation
        shared_features = self.shared_layers(features)

        # Classification outputs
        outputs = {}
        for target, classifier in self.classifiers.items():
            outputs[target] = classifier(shared_features)

        return outputs



# Global variables for model and preprocessing
model = None
transform = None
label_encoders = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """Load the trained model and preprocessing components"""
    global model, transform, label_encoders

    # Define class counts (update with your actual values)
    num_classes_dict = {
        'baseColour': 11,    # After color grouping
        'articleType': 142,  # After pruning <10 samples
        'season': 4,
        'gender': 5
    }

    # Initialize model
    model = ConvNeXtFashionClassifier(num_classes_dict)
    model.load_state_dict(torch.load(
        'best_convnext_fashion_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Load label encoders (you'll need to save these during training)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Load and preprocess image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)

        # Process predictions
        predictions = {}
        confidences = {}

        for target in outputs.keys():
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs[target], dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

            # Decode label
            predicted_class = label_encoders[target].inverse_transform(
                [predicted_idx.cpu().numpy()[0]])[0]

            predictions[target] = predicted_class
            confidences[target] = float(confidence.cpu().numpy()[0])

        return jsonify({
            'predictions': predictions,
            'confidences': confidences,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Prediction endpoint for base64 encoded images"""
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)

        # Process predictions (same as above)
        predictions = {}
        confidences = {}

        for target in outputs.keys():
            probs = torch.nn.functional.softmax(outputs[target], dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_class = label_encoders[target].inverse_transform(
                [predicted_idx.cpu().numpy()[0]])[0]

            predictions[target] = predicted_class
            confidences[target] = float(confidence.cpu().numpy()[0])

        return jsonify({
            'predictions': predictions,
            'confidences': confidences,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)