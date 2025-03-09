from flask import Blueprint, render_template, request, jsonify
import torch
import os
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer
from app.utils.tokenizer import tokenize_input

# Create blueprint
main_bp = Blueprint('main', __name__)

# Path to the model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'models', 'biobert_model')

# Load model and disease classes
model = None
tokenizer = None
disease_classes = []

def load_model():
    global model, tokenizer, disease_classes
    try:
        # Load the model if it exists
        if os.path.exists(MODEL_DIR):
            print(f"Loading model from {MODEL_DIR}")
            model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model.eval()  # Set to evaluation mode
            
            # Load disease classes if available
            disease_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'data', 'processed', 'disease_classes.txt')
            if os.path.exists(disease_path):
                with open(disease_path, 'r') as f:
                    disease_classes = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(disease_classes)} disease classes")
            else:
                print("Disease classes file not found")
            
            return True
        else:
            print(f"Model directory not found at {MODEL_DIR}. Please train the model first.")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Routes
@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    # Load model if not loaded
    if model is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model not available. Please train the model first.',
                'details': 'Run python app/models/train_model.py to train the model'
            }), 500
    
    # Get symptoms from request
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        # Tokenize input
        inputs = tokenizer(
            symptoms,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Get disease name
        if disease_classes and predicted_class < len(disease_classes):
            disease = disease_classes[predicted_class]
        else:
            disease = f"Disease_{predicted_class}"
        
        # Get confidence score
        confidence = torch.nn.functional.softmax(predictions, dim=1)[0][predicted_class].item()
        
        return jsonify({
            'disease': disease,
            'confidence': round(confidence * 100, 2),
            'symptoms': symptoms
        })
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback_str
        }), 500