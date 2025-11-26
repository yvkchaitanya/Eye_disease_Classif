import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import os

# Initialize Flask App
app = Flask(__name__)

# Model Definition
class HybridEyeDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridEyeDiseaseModel, self).__init__()
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.swin_transformer = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        
        efficientnet_out = self.efficientnet.num_features
        swin_out = self.swin_transformer.num_features
        
        self.fc = nn.Sequential(
            nn.Linear(efficientnet_out + swin_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x1 = self.efficientnet(x)
        x2 = self.swin_transformer(x)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# Load Model
def load_model():
    num_classes = 4  # Update this based on your model
    model = HybridEyeDiseaseModel(num_classes=num_classes)
    model_path = os.path.join(os.path.dirname(__file__), "eye_disease_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize model and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model().to(device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_LABELS = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Verify the file is an image
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": "Invalid image file", "details": str(e)}), 400
            
        # Transform and predict
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return jsonify({
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": float(confidence),
            "class_probabilities": {
                cls: float(prob) for cls, prob in zip(CLASS_LABELS, probabilities[0].tolist())
            }
        })
        
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)