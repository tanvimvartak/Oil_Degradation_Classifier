"""
Flask web application for Oil Degradation Classifier.
Provides a user-friendly interface for uploading images and getting predictions.
"""

from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.preprocessing import ImagePreprocessor
from src.model import OilDegradationCNN
import numpy as np

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
device = None
config = None

def load_model():
    """Load the trained model."""
    global model, preprocessor, device, config
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(image_size=config.IMAGE_SIZE)
    
    # Load model
    model = OilDegradationCNN(num_classes=config.NUM_CLASSES)
    
    # Load trained weights - check both possible model paths
    model_path = Path(config.MODEL_SAVE_PATH)
    
    # Try best_model_best.pth if best_model.pth doesn't exist
    if not model_path.exists():
        model_path = Path('models/best_model_best.pth')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found. Please train the model first using: python main.py")
    
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path} on {device}")

def predict_image(image):
    """
    Predict oil degradation stage from image.
    
    Args:
        image: PIL Image
    
    Returns:
        dict: Prediction results
    """
    # Preprocess image
    transform = preprocessor.get_val_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    # Get class name
    class_names = config.CLASS_NAMES
    predicted_label = class_names[predicted_class.item()]
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    # Map to usage description
    usage_map = {
        "fresh.oil": "Fresh Oil (Never Used)",
        "1.time.use.oil": "Used Once",
        "2.time.use.oil": "Used Twice",
        "3.time.use.oil": "Used Three Times"
    }
    
    # Recommendation based on usage
    recommendations = {
        "fresh.oil": "✓ Safe to use. Oil is fresh and suitable for cooking.",
        "1.time.use.oil": "✓ Safe to use. Oil can be reused once more.",
        "2.time.use.oil": "⚠ Caution. Consider replacing soon. Oil quality is degrading.",
        "3.time.use.oil": "✗ Not recommended. Replace oil immediately for health safety."
    }
    
    # Detailed health and safety information
    health_info = {
        "fresh.oil": {
            "status": "safe",
            "why_cannot_use": None,
            "health_risks": [],
            "chemical_changes": [
                "• Oil maintains optimal nutritional value",
                "• No harmful compounds detected",
                "• Antioxidants and vitamins intact"
            ],
            "if_used": [
                "✓ Safe for consumption",
                "✓ Optimal flavor and texture",
                "✓ No health concerns"
            ],
            "technical_details": [
                "• Free Fatty Acid (FFA) content: < 0.5% (optimal)",
                "• Peroxide Value: < 10 meq/kg (fresh)",
                "• Color: Clear, golden yellow",
                "• No visible degradation particles"
            ]
        },
        "1.time.use.oil": {
            "status": "safe",
            "why_cannot_use": None,
            "health_risks": [],
            "chemical_changes": [
                "• Slight increase in Free Fatty Acids (FFA)",
                "• Minor breakdown of antioxidants",
                "• Some polymerization begins"
            ],
            "if_used": [
                "✓ Generally safe for one more use",
                "✓ Minimal health impact",
                "⚠ Monitor for darkening or off-odors"
            ],
            "technical_details": [
                "• Free Fatty Acid (FFA) content: 0.5-1.0% (acceptable)",
                "• Peroxide Value: 10-20 meq/kg (moderate)",
                "• Color: Slightly darker than fresh",
                "• Some food particles may be present"
            ]
        },
        "2.time.use.oil": {
            "status": "caution",
            "why_cannot_use": [
                "• High levels of harmful compounds forming",
                "• Significant loss of nutritional value",
                "• Increased risk of free radical formation"
            ],
            "health_risks": [
                "• Increased oxidative stress in the body",
                "• Potential inflammation from oxidized compounds",
                "• Reduced absorption of essential nutrients"
            ],
            "chemical_changes": [
                "• Free Fatty Acids (FFA) significantly increased",
                "• Formation of polar compounds and polymers",
                "• Breakdown of beneficial antioxidants",
                "• Acrolein and other aldehydes forming"
            ],
            "if_used": [
                "⚠ May cause digestive discomfort",
                "⚠ Increased risk of inflammation",
                "⚠ Potential long-term health effects",
                "⚠ Food may have off-flavors"
            ],
            "technical_details": [
                "• Free Fatty Acid (FFA) content: 1.0-2.0% (high)",
                "• Peroxide Value: 20-30 meq/kg (elevated)",
                "• Color: Noticeably darker, brownish tint",
                "• Visible degradation particles present"
            ]
        },
        "3.time.use.oil": {
            "status": "danger",
            "why_cannot_use": [
                "• Extremely high levels of toxic compounds",
                "• Formation of carcinogenic substances (PAHs, aldehydes)",
                "• Complete loss of nutritional benefits",
                "• High concentration of free radicals",
                "• Polymerized compounds that are difficult to digest"
            ],
            "health_risks": [
                "• Increased cancer risk from polycyclic aromatic hydrocarbons (PAHs)",
                "• Cardiovascular problems from oxidized lipids",
                "• Digestive issues and inflammation",
                "• Accelerated aging due to oxidative stress",
                "• Potential liver and kidney damage with prolonged exposure"
            ],
            "chemical_changes": [
                "• Free Fatty Acids (FFA) > 2.0% (dangerous levels)",
                "• High peroxide value indicating severe oxidation",
                "• Formation of trans fats and harmful aldehydes",
                "• Polymerization creating indigestible compounds",
                "• Breakdown of all beneficial compounds"
            ],
            "if_used": [
                "✗ Immediate: Digestive discomfort, nausea",
                "✗ Short-term: Inflammation, oxidative stress",
                "✗ Long-term: Increased cancer risk, cardiovascular issues",
                "✗ Food quality: Poor taste, dark color, unpleasant odor"
            ],
            "technical_details": [
                "• Free Fatty Acid (FFA) content: > 2.0% (critical)",
                "• Peroxide Value: > 30 meq/kg (dangerous)",
                "• Color: Dark brown/black, cloudy",
                "• Heavy particle accumulation visible",
                "• Viscosity increased due to polymerization"
            ]
        }
    }
    
    info = health_info[predicted_label]
    
    return {
        'predicted_class': predicted_label,
        'usage_description': usage_map[predicted_label],
        'confidence': float(confidence.item() * 100),
        'recommendation': recommendations[predicted_label],
        'status': info['status'],
        'why_cannot_use': info['why_cannot_use'],
        'health_risks': info['health_risks'],
        'chemical_changes': info['chemical_changes'],
        'if_used': info['if_used'],
        'technical_details': info['technical_details'],
        'all_probabilities': {
            'Fresh Oil': float(all_probs[0] * 100),
            'Used Once': float(all_probs[1] * 100),
            'Used Twice': float(all_probs[2] * 100),
            'Used Three Times': float(all_probs[3] * 100)
        }
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        result = predict_image(image)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        result['image'] = f"data:image/jpeg;base64,{img_str}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=8000)
