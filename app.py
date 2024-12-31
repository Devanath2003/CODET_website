from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model setup
device = torch.device('cpu')
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    global model
    try:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

        model_path = os.path.join(MODEL_FOLDER, r'C:\Users\devan\Desktop\website\real_website\betterone - Copy\backend\model\coconut_detector_final.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        model = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/detect', methods=['POST'])
def detect_coconut():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500

    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save and process image
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(image_tensor)[0]
            
            scores = prediction['scores']
            if len(scores) == 0 or scores[0] < 0.5:
                result = {
                    "prediction": "No coconut detected",
                    "confidence": 0
                }
            else:
                label = prediction['labels'][0].item()
                confidence = scores[0].item()
                coconut_type = "immature" if label == 1 else "mature"
                result = {
                    "prediction": f"{coconut_type.capitalize()} coconut detected",
                    "confidence": round(confidence * 100, 2)
                }
        
        # Clean up
        os.remove(image_path)
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': str(e)}), 500

# Initialize model when starting the server
@app.before_first_request
def before_first_request():
    initialize_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
