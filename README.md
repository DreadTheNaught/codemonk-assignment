# Fashion Product Image Classification with ConvNeXt-Tiny

## Project Overview

This assignment for CodeMonk implements a multi-label fashion image classification system using ConvNeXt-Tiny as the backbone architecture. The model predicts four key fashion attributes from product images:

- **Color Group**: 8 color categories (after grouping strategy)
- **Article Type**: 140+ clothing categories
- **Season**: 4 seasonal categories
- **Gender**: 5 demographic categories

The solution includes comprehensive EDA, model training, and deployment through both API and Streamlit interfaces.

## File Structure

```
fashion_deployment/
├── api.py                      # Flask API for model serving
├── streamlit_gui.py                     # Streamlit demonstration app
├── best_convnext_fashion_model.pth     # Trained model weights
├── label_encoders.pkl                  # Label encoders for predictions
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── images/                     # Sample fashion images
    ├── fashion_image.jpg
```


## Dataset Information

- **Source**: Fashion Product Images Dataset (Kaggle)
- **Total Samples**: 44,424 fashion items
- **Images**: 224x224 RGB product images
- **Targets**: Multi-label classification (Color, Type, Season, Gender)
- **Data Quality**: 99.9% completeness with minimal missing values


## Exploratory Data Analysis (EDA)

### Step 1: Data Loading and Quality Assessment

- Successfully loaded 44,424 fashion samples with 10 features
- Identified minimal missing data (36 missing values across target variables)
- Confirmed all 4 target variables present with high data completeness


### Step 2: Target Variable Distribution Analysis

**Class Imbalance Assessment:**

- **ArticleType**: 143 classes, 7067:1 ratio (extreme imbalance)
- **BaseColour**: 46 classes, 1945:1 ratio (severe imbalance)
- **Gender**: 5 classes, 33:1 ratio (moderate imbalance)
- **Season**: 4 classes, 7:1 ratio (manageable)

**Key Findings:**

- Tshirts dominate article types (7,067 samples)
- Black is the most common color (9,728 samples)
- Men's clothing represents 50% of dataset
- Summer items are most prevalent


### Step 3: Visual Distribution Analysis

- Created comprehensive visualizations showing class distributions
- Identified long-tail patterns in ArticleType and BaseColour
- Confirmed seasonal balance and gender distribution patterns
- Used log-scale plots to visualize extreme imbalances


### Step 4: Multi-Label Relationship Analysis

**Key Relationships Discovered:**

- **Gender-Article Type**: Men prefer Tshirts/Shirts, Women prefer Kurtas/Tops
- **Season-Color**: Black dominates all seasons, Pink peaks in Spring
- **Cross-variable patterns**: Clear logical relationships
- **Unisex Categories**: Functional items like backpacks and sunglasses


### Step 5: Class Imbalance Strategy

**Color Grouping Implementation:**

- Reduced 46 individual colors to 11 logical color groups
- **Groups**: Black, White, Blue, Red, Brown, Green, Grey, Purple, oraneh, yellow, other
- **Impact**: Improved color classification from 39% to 60-75% accuracy

**ArticleType Class Pruning Implementation:**

- Removed 8-13 classes with fewer than 10 samples from 143 total categories
- **Classes Removed:** Singleton items (Ipad, Suits, Body Wash and Scrub, Mens Grooming Kit, Shoe Laces) and ultra-rare specialty items
- **Impact:** Improved training stability and expected accuracy from 65-70% to 70-75%



## Model Architecture

### ConvNeXt-Tiny Multi-Label Classifier

**Backbone Architecture:**

- **Base Model**: ConvNeXt-Tiny (pre-trained on ImageNet)
- **Feature Extraction**: Frozen backbone (28M parameters frozen)
- **Approach**: Transfer learning with feature extraction
- **Output Features**: 768-dimensional feature vectors

**Classification Heads:**

- **Shared Dense Layers**: 768 → 512 → 256 features
- **Task-Specific Heads**:
    - ArticleType: 256 neurons (most complex, 140+ classes)
    - BaseColour: 128 neurons (medium complexity, 8 groups)
    - Season/Gender: 64 neurons (fewer classes)

**Training Configuration:**

- **Optimizer**: AdamW with cosine annealing
- **Learning Rate**: 1e-3 with scheduling
- **Batch Size**: 32
- **Epochs**: 15 with early stopping
- **Loss Function**: Weighted CrossEntropyLoss for class imbalance


### Key Functions

1. **Class Weight Alignment**: Proper alignment between label encoders and class weights
2. **Color Grouping Strategy**: Reduced color complexity while maintaining business relevance
3. **Hierarchical Head Design**: Variable head sizes based on classification complexity
4. **Robust Error Handling**: Handling of missing images during training

## Model Performance

### Final Accuracy Results

- **Gender**: 84.40% (Excellent - 5 classes)
- **ArticleType**: 79.18% (Very Good - 140+ classes with extreme imbalance)
- **Season**: 64.02% (Moderate - 4 classes)
- **BaseColour**: 67.88% (Expected after color grouping - 8 groups)


### Training Characteristics

- **Minimal Overfitting**: Train/validation loss gap < 0.02
- **Stable Convergence**: Consistent improvement across epochs
- **Efficient Training**: 2.35 it/s on Kaggle GPU environment


## API Deployment

### Flask API Features

- **Health Check**: `/health` endpoint for service monitoring
- **File Upload**: `/predict` for direct image file uploads
- **Base64 Support**: `/predict_base64` for encoded image data
- **JSON Responses**: Structured predictions with confidence scores


### API Usage

**Start the API:**

```bash
python api.py
```

**Test with curl:**

```bash
# Health check
curl http://localhost:5000/health

# Predict with image file
curl -X POST -F "image=@images/fashion_image.jpg" http://localhost:5000/predict
```

or 

```bash
python test.py
```

**Expected Response:**

```json
{
  "predictions": {
    "baseColour": "Blue_Group",
    "articleType": "Tshirts", 
    "season": "Summer",
    "gender": "Men"
  },
  "confidences": {
    "baseColour": 0.85,
    "articleType": 0.92,
    "season": 0.78,
    "gender": 0.89
  },
  "status": "success"
}
```


## Streamlit Demo

### Interactive Features

- **Image Upload Interface**: Drag-and-drop or browse functionality
- **Real-time Predictions**: Instant classification with confidence scores
- **Batch Processing**: Multiple image analysis capability
- **Model Analytics**: Performance metrics visualization


### Demo Usage

**Run Streamlit App:**

```bash
streamlit run fashion_demo.py
```

**Access Interface:**

- Navigate to `http://localhost:8501`
- Upload fashion images (PNG, JPG, JPEG)
- View multi-label predictions with confidence scores


## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM
- 2GB+ disk space for model weights


### Installation Steps

1. **Clone or Download the Project Files**

```bash
# Ensure all files from the file structure are present
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify Model Files**

```bash
# Ensure these files exist:
# - best_convnext_fashion_model.pth
# - label_encoders.pkl
```

4. **Test Installation**

```bash
# Test API
python fashion_api.py

# Test Streamlit (in new terminal)
streamlit run fashion_demo.py
```


### Requirements

```
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
streamlit>=1.28.0
pillow>=8.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
requests>=2.25.0
pandas>=1.3.0
```


## Model Limitations and Future Work

### Current Limitations

- **BaseColour Accuracy**: Color classification remains challenging due to subjective labeling
- **Rare Classes**: Some article types with very few samples may be underperforming
- **Image Quality Dependency**: Performance varies with image quality and lighting


### Future Improvements

- **Fine-Tuning**: Unfreeze backbone layers for potential accuracy gains
- **Data Augmentation**: Implement fashion-specific augmentation strategies
- **Ensemble Methods**: Combine multiple models for improved performance
- **Real-Time Inference**: Optimize for faster prediction speeds

