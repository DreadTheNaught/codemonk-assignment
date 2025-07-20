import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pickle
import io
import base64
import requests
from sklearn.preprocessing import LabelEncoder

# Configure page
st.set_page_config(
    page_title="Fashion Classification Demo",
    page_icon="👗",
    layout="wide"
)

# Model architecture (same as API)


class ConvNeXtFashionClassifier(nn.Module):
    def __init__(self, num_classes_dict, pretrained=False):
        super(ConvNeXtFashionClassifier, self).__init__()

        self.backbone = models.convnext_tiny(pretrained=pretrained)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

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

        self.classifiers = nn.ModuleDict()
        for target, num_classes in num_classes_dict.items():
            if target == 'articleType':
                head_size = 256
            elif target == 'baseColour':
                head_size = 128
            else:
                head_size = 64

            self.classifiers[target] = nn.Sequential(
                nn.Linear(256, head_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(head_size),
                nn.Dropout(0.2),
                nn.Linear(head_size, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        shared_features = self.shared_layers(features)

        outputs = {}
        for target, classifier in self.classifiers.items():
            outputs[target] = classifier(shared_features)

        return outputs


@st.cache_resource
def load_model():
    """Load model and preprocessing components (cached for efficiency)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define class counts
    num_classes_dict = {
        'baseColour': 11,
        'articleType': 142,
        'season': 4,
        'gender': 5
    }

    # Load model
    model = ConvNeXtFashionClassifier(num_classes_dict)
    model.load_state_dict(torch.load(
        'best_convnext_fashion_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load label encoders
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    return model, label_encoders, transform, device


def predict_fashion(image, model, label_encoders, transform, device):
    """Make fashion predictions"""
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process results
    predictions = {}
    confidences = {}

    for target in outputs.keys():
        probs = torch.nn.functional.softmax(outputs[target], dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = label_encoders[target].inverse_transform(
            [predicted_idx.cpu().numpy()[0]])[0]

        predictions[target] = predicted_class
        confidences[target] = float(confidence.cpu().numpy()[0])

    return predictions, confidences


def main():
    # Title and description
    st.title("👗 Fashion Classification Demo")
    st.markdown("""
    ### ConvNeXt-Tiny Multi-Label Fashion Classifier
    
    Upload a fashion image to predict:
    - **Color Group**: Dominant color category
    - **Article Type**: Type of clothing item
    - **Season**: Seasonal appropriateness
    - **Gender**: Target demographic
    
    *Built with ConvNeXt-Tiny backbone and trained on 44K+ fashion samples*
    """)

    # Load model
    try:
        model, label_encoders, transform, device = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Model information
    with st.sidebar.expander("Model Information"):
        st.write("**Architecture**: ConvNeXt-Tiny")
        st.write("**Training Samples**: 44,424")
        st.write("**Targets**: 4 fashion attributes")
        st.write("**Approach**: Feature extraction")

    # Target information
    with st.sidebar.expander("Prediction Targets"):
        st.write("**Color Groups**: 8 categories")
        st.write("**Article Types**: 140+ categories")
        st.write("**Seasons**: 4 categories")
        st.write("**Gender**: 5 categories")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📷 Upload Image")

        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a fashion image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a fashion item"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict button
            if st.button("🔍 Classify Fashion Item", type="primary"):
                with st.spinner("Analyzing fashion item..."):
                    try:
                        predictions, confidences = predict_fashion(
                            image, model, label_encoders, transform, device
                        )

                        # Store results in session state
                        st.session_state.predictions = predictions
                        st.session_state.confidences = confidences

                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    with col2:
        st.header("🎯 Predictions")

        if hasattr(st.session_state, 'predictions'):
            predictions = st.session_state.predictions
            confidences = st.session_state.confidences

            # Create prediction cards
            for target in ['baseColour', 'articleType', 'season', 'gender']:
                if target in predictions:
                    # Target name mapping
                    target_names = {
                        'baseColour': 'Color Group',
                        'articleType': 'Article Type',
                        'season': 'Season',
                        'gender': 'Gender'
                    }

                    # Confidence color
                    conf = confidences[target]
                    if conf >= 0.8:
                        conf_color = "green"
                    elif conf >= 0.6:
                        conf_color = "orange"
                    else:
                        conf_color = "red"

                    # Display prediction card
                    st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd; margin: 0.5rem 0;">
                        <h4 style="margin-top: 0;">{target_names[target]}</h4>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>{predictions[target]}</strong></p>
                        <p style="color: {conf_color}; margin: 0;">Confidence: {conf:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Summary section
            st.markdown("---")
            st.subheader("📊 Classification Summary")

            avg_confidence = np.mean(list(confidences.values()))

            if avg_confidence >= 0.8:
                st.success(
                    f"High confidence predictions (avg: {avg_confidence:.1%})")
            elif avg_confidence >= 0.6:
                st.warning(
                    f"Moderate confidence predictions (avg: {avg_confidence:.1%})")
            else:
                st.error(
                    f"Low confidence predictions (avg: {avg_confidence:.1%})")

            # Detailed results in expandable section
            with st.expander("View Detailed Results"):
                st.json({
                    'predictions': predictions,
                    'confidences': {k: f"{v:.3f}" for k, v in confidences.items()}
                })

        else:
            st.info(
                "Upload an image and click 'Classify Fashion Item' to see predictions")

            # Example predictions for demonstration
            st.markdown("**Example Output:**")
            st.code("""
            Color Group: Blue_Group (85% confidence)
            Article Type: Tshirts (92% confidence)
            Season: Summer (78% confidence)
            Gender: Men (89% confidence)
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Fashion Classification Demo | Powered by ConvNeXt-Tiny | Multi-Label Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
