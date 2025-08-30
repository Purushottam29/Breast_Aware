from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

# Define folders
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = './models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Load models with error handling
try:
    neural_network_model = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'densenet121_model.h5'))
    print("Neural network model loaded successfully")
except Exception as e:
    print(f"Error loading neural network model: {e}")
    neural_network_model = None

try:
    svm_model = joblib.load(os.path.join(MODELS_FOLDER, 'breast_cancer_svm_model1.pkl'))
    print("SVM model loaded successfully")
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

# Define class labels
CLASS_LABELS = ["Benign", "Malignant", "Normal"]

def preprocess_for_densenet(image_path):
    """Preprocess image for DenseNet121 model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Resize for DenseNet
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def preprocess_for_svm(image_path):
    """Preprocess image for SVM model (grayscale + HOG features)."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to match training size
    img_array = np.array(img)

    # Extract HOG features
    features, _ = hog(
        img_array,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True
    )
    return features.reshape(1, -1)  # Reshape for prediction

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'API is running',
        'neural_network_loaded': neural_network_model is not None,
        'svm_loaded': svm_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request!")  # Debug log

        if 'file' not in request.files or 'model' not in request.form:
            return jsonify({'success': False, 'error': 'Missing file or model selection'})

        file = request.files['file']
        model_id = request.form['model']
        
        print(f"Model Selected: {model_id}")  # Debug log

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        print(f"File saved at: {image_path}")

        # Predict
        if model_id in ['DenseNet121', 'neural-network']:
            if neural_network_model is None:
                return jsonify({'success': False, 'error': 'Neural network model not available'})
                
            preprocessed_data = preprocess_for_densenet(image_path)
            print(f"Preprocessed Data Shape: {preprocessed_data.shape}")  # Debug log

            prediction = neural_network_model.predict(preprocessed_data)[0]
            print(f"Raw Prediction: {prediction}")  # Debug log

            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            probabilities = {CLASS_LABELS[i]: float(prob) * 100 for i, prob in enumerate(prediction)}

        elif model_id in ['svm', 'SVM']:
            if svm_model is None:
                return jsonify({'success': False, 'error': 'SVM model not available'})
                
            preprocessed_data = preprocess_for_svm(image_path)
            print(f"SVM Features Shape: {preprocessed_data.shape}")  # Debug log

            prediction = svm_model.predict(preprocessed_data)[0]
            print(f"SVM Prediction: {prediction}")  # Debug log

            if hasattr(svm_model, "predict_proba"):
                proba = svm_model.predict_proba(preprocessed_data)[0]
                probabilities = {CLASS_LABELS[i]: float(proba[i]) * 100 for i in range(len(proba))}
                predicted_class = np.argmax(proba)  # Get the class with the highest probability
            else:
                probabilities = {CLASS_LABELS[prediction]: 100}
                predicted_class = prediction  # Directly use prediction as class label

            confidence = float(max(probabilities.values())) if probabilities else None

        else:
            return jsonify({'success': False, 'error': 'Unknown model ID'})

        predicted_label = CLASS_LABELS[predicted_class]

        response = {
            'success': True,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'modelName': model_id
        }
        print("Flask Response:", response)
        return jsonify(response)

    except Exception as e:
        print("Error:", str(e))  # Log error
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
