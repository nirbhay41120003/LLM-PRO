import torch
import os
import logging
import re
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthModelHandler:
    """Handler for the health model predictions"""
    
    def __init__(self, model_dir=None, model_type="bio_clinical_bert"):
        """Initialize the health model handler
        
        Args:
            model_dir: Directory where the model is stored
            model_type: Type of model to use ('bio_clinical_bert', 'pubmed_bert', 'biogpt', 'biobert')
        """
        self.model = None
        self.tokenizer = None
        self.disease_classes = []
        self.model_dir = model_dir
        self.model_type = model_type
        self.disease_info = {}
        self._load_disease_info()
        
    def _load_disease_info(self):
        """Load comprehensive disease information from the JSON database"""
        try:
            # Find the path to the disease info JSON file
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            disease_info_path = os.path.join(os.path.dirname(app_dir), 'data', 'processed', 'disease_info.json')
            
            if os.path.exists(disease_info_path):
                with open(disease_info_path, 'r') as f:
                    self.disease_info = json.load(f)
                logger.info(f"Loaded detailed info for {len(self.disease_info)} diseases")
            else:
                logger.warning("Disease information database not found at: " + disease_info_path)
        except Exception as e:
            logger.error(f"Error loading disease information: {str(e)}")
            
    def load_model(self, model_dir=None):
        """Load the model and tokenizer"""
        try:
            # Use provided model_dir or the one from config
            model_dir = model_dir or self.model_dir
            
            if not model_dir or not os.path.exists(model_dir):
                logger.error(f"Model directory not found at {model_dir}")
                return False
            
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Determine which model to load based on model_type
            if self.model_type == "bio_clinical_bert":
                logger.info("Loading Bio_ClinicalBERT model...")
                # If we have a fine-tuned model in the model_dir, use that
                if os.path.exists(os.path.join(model_dir, "config.json")):
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                else:
                    # Otherwise, use the pre-trained model (we'll need to fine-tune it)
                    self.model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                    self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            
            elif self.model_type == "pubmed_bert":
                logger.info("Loading PubMedBERT model...")
                # If we have a fine-tuned model in the model_dir, use that
                if os.path.exists(os.path.join(model_dir, "config.json")):
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                else:
                    # Otherwise, use the pre-trained model
                    self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            
            elif self.model_type == "biogpt":
                logger.info("Loading BioGPT model...")
                # If we have a fine-tuned model in the model_dir, use that
                if os.path.exists(os.path.join(model_dir, "config.json")):
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                else:
                    # Otherwise, use the pre-trained model
                    self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/biogpt")
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            
            else:  # Default to BioBERT
                logger.info("Loading BioBERT model...")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            self.model.eval()  # Set to evaluation mode
            
            # Load disease classes if available
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            disease_path = os.path.join(os.path.dirname(app_dir), 'data', 'processed', 'disease_classes.txt')
            
            if os.path.exists(disease_path):
                with open(disease_path, 'r') as f:
                    self.disease_classes = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.disease_classes)} disease classes")
            else:
                logger.warning("Disease classes file not found")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def is_model_loaded(self):
        """Check if the model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def predict(self, symptoms):
        """Make a prediction based on symptoms"""
        if not self.is_model_loaded():
            success = self.load_model()
            if not success:
                return {
                    'error': 'Model not available. Please train the model first.',
                    'details': 'Run python app/models/train_model.py to train the model'
                }
        
        try:
            logger.info(f"Making prediction for symptoms: {symptoms}")
            
            # Check if this is a non-health query that somehow made it to prediction
            if not self._is_symptom_description(symptoms):
                logger.info("Non-symptom description passed to predict, returning error")
                return {
                    'error': 'Not a symptom description',
                    'message': 'The text does not appear to be describing symptoms.'
                }
            
            # Special case handling for common conditions that might not be in the training data
            symptoms_lower = symptoms.lower()
            
            # Enhanced UTI detection - check for common UTI-related terms
            uti_terms = ['urine infection', 'urinary infection', 'uti', 'bladder infection', 
                        'urinary tract', 'painful urination', 'burning when i pee', 
                        'frequent urination', 'burning sensation when urinating']
                        
            for term in uti_terms:
                if term in symptoms_lower:
                    logger.info(f"Detected urinary tract infection keyword: '{term}'")
                    
                    # Skip checking disease classes - directly return UTI
                    logger.info("Returning manual UTI prediction")
                    return {
                        'disease': 'Urinary Tract Infection',
                        'confidence': 95.0,  # High confidence for this specific case
                        'symptoms': symptoms,
                        'alternatives': [],
                        'manual_prediction': True
                    }
            
            # Tokenize input
            inputs = self.tokenizer(
                symptoms,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
                
                # Get probabilities for top 3 predictions
                probabilities = torch.nn.functional.softmax(predictions, dim=1)[0]
                top_3_probs, top_3_indices = torch.topk(probabilities, 3)
                
                # Get disease names and probabilities
                top_3_diseases = []
                for idx, (prob, index) in enumerate(zip(top_3_probs, top_3_indices)):
                    index_item = index.item()
                    if self.disease_classes and index_item < len(self.disease_classes):
                        disease = self.disease_classes[index_item]
                    else:
                        disease = f"Disease_{index_item}"
                    
                    confidence = prob.item()
                    top_3_diseases.append({
                        'disease': disease,
                        'confidence': round(confidence * 100, 2)
                    })
                
                # Main prediction is the top disease
                primary_prediction = top_3_diseases[0]
                disease = primary_prediction['disease']
                confidence = primary_prediction['confidence']
                
                # Check if the confidence is too low - below this we shouldn't make predictions
                CONFIDENCE_THRESHOLD = 30  # Only make predictions with at least 30% confidence
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.warning(f"Confidence too low ({confidence}%) for prediction, below threshold {CONFIDENCE_THRESHOLD}%")
                    return {
                        'error': 'Could not determine a specific condition with enough confidence',
                        'confidence': confidence,
                        'message': 'The symptoms provided are too vague or don\'t match any specific condition in my database with sufficient confidence.'
                    }
            
            logger.info(f"Prediction result: Disease={disease}, Confidence={round(confidence, 2)}%")
            
            return {
                'disease': disease,
                'confidence': round(confidence, 2),
                'symptoms': symptoms,
                'alternatives': top_3_diseases[1:],  # Return the 2nd and 3rd predictions as alternatives
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def get_health_response(self, query, include_disclaimer=True):
        """Get a response for health-related queries"""
        logger.info(f"Processing health query: {query}")
        
        # First check if this is a symptom description for prediction
        is_symptom = self._is_symptom_description(query)
        logger.info(f"Is symptom description: {is_symptom}")
        
        if is_symptom:
            logger.info("Treating as symptom description, making prediction...")
            prediction = self.predict(query)
            
            if 'error' in prediction:
                logger.error(f"Prediction error: {prediction['error']}")
                # Give a more helpful response when confidence is too low
                if 'confidence' in prediction:
                    return {
                        'error': prediction['error'],
                        'response': f"I'm sorry, but I couldn't determine a specific condition based on the symptoms you described. The information provided might be too general or might not match conditions in my database. Please provide more specific symptoms or consult a healthcare provider for a proper diagnosis."
                    }
                else:
                    return {
                        'error': prediction['error'],
                        'response': f"I'm sorry, but I couldn't analyze your symptoms. {prediction['error']}"
                    }
            
            # Format a nice response with the prediction
            disease = prediction['disease']
            confidence = prediction['confidence']
            
            # Generate a comprehensive response with detailed disease information
            response = f"Based on your symptoms, I think you might have: **{disease}**.\n\n"
            
            # Add comprehensive information about the disease
            disease_info = self._get_comprehensive_disease_info(disease)
            if disease_info:
                response += disease_info
            
            # Add alternative possibilities if confidence is not very high
            if confidence < 70 and 'alternatives' in prediction and prediction['alternatives']:
                response += "\n\n**Other possibilities to consider:**\n"
                for alt in prediction['alternatives']:
                    response += f"- {alt['disease']} (confidence: {alt['confidence']}%)\n"
            
            if include_disclaimer:
                response += "\n\n**Important:** This is not a medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
            
            logger.info(f"Returning disease prediction: {prediction['disease']}")
            return {
                'response': response,
                'disease': prediction['disease'],
                'confidence': prediction['confidence']
            }
        else:
            logger.info("Not a symptom description, providing general health response")
            # For general health questions, provide a helpful response
            response = self._get_general_health_response(query)
            return {
                'response': response
            }
    
    def _is_symptom_description(self, text):
        """Check if the text is describing symptoms"""
        # List of common greetings and non-medical queries to filter out
        non_symptom_phrases = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what\'s up', 'how is it going', 'nice to meet you',
            'thanks', 'thank you', 'help', 'what can you do', 'who are you',
            'bye', 'goodbye', 'see you', 'talk to you later'
        ]
        
        text_lower = text.lower().strip()
        logger.info(f"Checking if text is symptom description: '{text_lower}'")
        
        # First check if this is just a greeting or non-medical query
        for phrase in non_symptom_phrases:
            if text_lower == phrase or text_lower.startswith(phrase + " ") or text_lower.endswith(" " + phrase):
                logger.info(f"Detected non-symptom phrase: '{text_lower}'")
                return False
        
        # Very basic check for symptom-like content
        symptom_indicators = [
            'symptom', 'feel', 'pain', 'ache', 'hurt', 'sore', 'fever', 'cough', 
            'headache', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue', 'sick',
            'I have', 'I am experiencing', 'I\'ve been', 'My', 'suffering', 'disease',
            # Add more medical conditions and symptoms
            'cold', 'flu', 'infection', 'rash', 'swelling', 'inflammation', 
            'burning', 'itching', 'diarrhea', 'constipation', 'breathing', 
            'blood', 'pressure', 'diabetes', 'cancer', 'heart', 'chest', 
            'stomach', 'back', 'joint', 'muscle', 'throat', 'ear', 'eye',
            'nose', 'skin', 'allergy', 'sneeze', 'runny', 'congestion',
            'urine', 'urinary', 'uti', 'bladder', 'kidney', 'burning sensation'
        ]
        
        # Check for presence of symptom indicators
        for indicator in symptom_indicators:
            if indicator.lower() in text_lower:
                logger.info(f"Found symptom indicator: '{indicator}'")
                return True
        
        # If no specific symptom indicators were found, it's not a symptom description
        logger.info("No symptom indicators found in text")
        return False
    
    def _get_general_health_response(self, query):
        """Handle general health-related questions"""
        # Simple template responses for common health questions
        health_templates = {
            'covid': "COVID-19 symptoms may include fever, cough, and shortness of breath. If you're experiencing these symptoms, please get tested and consult a healthcare provider.",
            'vaccine': "Vaccines are an important way to protect against serious diseases. They work by triggering an immune response in your body. Always consult with your healthcare provider about which vaccines are right for you.",
            'headache': "Headaches can be caused by stress, dehydration, lack of sleep, or more serious conditions. For persistent or severe headaches, please consult a healthcare provider.",
            'diet': "A balanced diet typically includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Individual dietary needs may vary based on age, activity level, and health conditions.",
            'exercise': "Regular physical activity offers numerous health benefits including weight control, reduced risk of heart disease, and improved mental health. Aim for at least 150 minutes of moderate activity per week.",
            'sleep': "Adults typically need 7-9 hours of quality sleep per night. Consistent sleep schedules, a comfortable sleep environment, and limiting screen time before bed can improve sleep quality."
        }
        
        query_lower = query.lower()
        logger.info(f"Checking for template match in general health query: '{query_lower}'")
        
        # Try to match the query to a template
        for key, response in health_templates.items():
            if key in query_lower:
                logger.info(f"Found template match: '{key}'")
                return response
        
        # Default response if no specific template matches
        logger.info("No template match found, returning default response")
        return "I'm a health assistant trained to provide general health information and analyze symptoms. I can help identify possible health conditions based on symptoms you describe, but I'm not a replacement for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."

    def _get_disease_info(self, disease):
        """Get basic information about a disease (legacy method)"""
        # Simple dictionary of common diseases and their descriptions
        disease_lower = disease.lower()
        disease_info = {
            'common cold': "The common cold is a viral infection of your nose and throat. It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Symptoms usually include a runny nose, sore throat, cough, and congestion.",
            
            'flu': "The flu is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. Symptoms include fever, cough, sore throat, runny nose, body aches, headaches, fatigue, and sometimes vomiting and diarrhea.",
            
            'pneumonia': "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.",
            
            'bronchitis': "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. People who have bronchitis often cough up thickened mucus, which can be discolored.",
            
            'tuberculosis': "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects your lungs. The bacteria that cause tuberculosis are spread from one person to another through tiny droplets released into the air via coughs and sneezes.",
            
            'asthma': "Asthma is a condition in which your airways narrow and swell and produce extra mucus. This can make breathing difficult and trigger coughing, wheezing and shortness of breath.",
            
            'covid': "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Common symptoms include fever, cough, and tiredness. Other symptoms may include loss of taste or smell, sore throat, headache, body aches, and breathing difficulties.",
            
            'diabetes': "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose get into your cells to be used for energy.",
            
            'hypertension': "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            
            'heart disease': "Heart disease refers to several types of heart conditions. The most common type is coronary artery disease, which can cause heart attack. Other kinds include heart valve disease, heart failure, and arrhythmia.",
            
            'stroke': "A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes.",
            
            'migraine': "A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
            
            'allergies': "Allergies occur when your immune system reacts to a foreign substance that doesn't cause a reaction in most people. Common allergens include certain foods, pollen, bee venom, pet dander and specific medications.",
            
            'eczema': "Eczema is a condition where patches of skin become inflamed, itchy, red, cracked, and rough. Blisters may sometimes occur. Different stages may affect people of different ages."
        }
        
        # Check for partial matches
        for known_disease, info in disease_info.items():
            if known_disease in disease_lower or disease_lower in known_disease:
                return info
                
        # If no specific info is available
        return f"I don't have detailed information about {disease}, but I recommend consulting a healthcare provider for accurate diagnosis and treatment options."

    def _get_comprehensive_disease_info(self, disease):
        """
        Get comprehensive information about a disease from the disease info database
        
        Args:
            disease: Name of the disease to look up
            
        Returns:
            str: Formatted disease information or None if not found
        """
        # Try exact match first
        if disease in self.disease_info:
            info = self.disease_info[disease]
            return self._format_disease_info(disease, info)
        
        # Try case-insensitive match
        disease_lower = disease.lower()
        for disease_name, info in self.disease_info.items():
            if disease_name.lower() == disease_lower:
                return self._format_disease_info(disease_name, info)
        
        # Try partial match
        for disease_name, info in self.disease_info.items():
            if disease_lower in disease_name.lower() or disease_name.lower() in disease_lower:
                return self._format_disease_info(disease_name, info)
        
        # Fall back to basic disease info
        return self._get_disease_info(disease)
    
    def _format_disease_info(self, disease_name, info):
        """Format the disease information into a readable string"""
        result = f"**{disease_name}**\n\n"
        
        if "description" in info:
            result += f"{info['description']}\n\n"
        
        if "symptoms" in info:
            result += "**Common Symptoms:**\n"
            for symptom in info["symptoms"]:
                result += f"- {symptom}\n"
            result += "\n"
        
        if "causes" in info:
            result += "**Causes:**\n"
            for cause in info["causes"]:
                result += f"- {cause}\n"
            result += "\n"
        
        if "risk_factors" in info:
            result += "**Risk Factors:**\n"
            for factor in info["risk_factors"]:
                result += f"- {factor}\n"
            result += "\n"
        
        if "treatment" in info:
            result += "**Treatment Options:**\n"
            for treatment in info["treatment"]:
                result += f"- {treatment}\n"
            result += "\n"
        
        if "when_to_see_doctor" in info:
            result += f"**When to See a Doctor:**\n{info['when_to_see_doctor']}\n"
        
        return result

    def classify_message(self, message):
        """
        Classify a message using the trained model
        
        Args:
            message: The user's message
            
        Returns:
            dict: Classification result with label and score
        """
        # First check if this is even a symptom-related message
        if not self._is_symptom_description(message):
            logger.info(f"Message '{message}' is not a symptom description")
            return {
                'label': 'Not Health Related',
                'score': 0.0,
                'explanation': "I'm here to help with health-related questions and symptoms. If you'd like me to analyze symptoms or provide health information, please describe how you're feeling or ask a specific health question."
            }
            
        try:
            # Make prediction using the existing predict method
            prediction = self.predict(message)
            
            if 'error' in prediction:
                logger.error(f"Error in prediction: {prediction['error']}")
                # Give a more helpful response for low confidence
                if 'confidence' in prediction:
                    return {
                        'label': 'Uncertain',
                        'score': prediction.get('confidence', 0) / 100,
                        'explanation': "I'm sorry, but I couldn't determine a specific condition based on the symptoms you described. The information provided might be too general or might not match conditions in my database. Please provide more specific symptoms or consult a healthcare provider for a proper diagnosis."
                    }
                return {
                    'label': 'Error',
                    'score': 0.0,
                    'explanation': f"Could not classify the message: {prediction['error']}"
                }
            
            # Check if the confidence is too low
            if prediction.get('confidence', 0) < 30:
                logger.info(f"Confidence too low ({prediction.get('confidence', 0)}%), returning uncertain response")
                return {
                    'label': 'Uncertain',
                    'score': prediction.get('confidence', 0) / 100,
                    'explanation': "I'm sorry, but I couldn't determine a specific condition with enough confidence based on the symptoms you described. The information provided might be too general or might not match conditions in my database with sufficient certainty. Please provide more specific symptoms or consult a healthcare provider for a proper diagnosis."
                }
            
            # Get comprehensive disease information
            disease_info = self._get_comprehensive_disease_info(prediction['disease'])
            
            # For UTI or other manually predicted diseases that might not have info
            if prediction.get('manual_prediction', False) and not disease_info:
                if prediction['disease'] == 'Urinary Tract Infection':
                    disease_info = """
**Urinary Tract Infection (UTI)**

A urinary tract infection affects the urinary system, which includes the kidneys, ureters, bladder, and urethra.

**Common Symptoms:**
- Burning sensation when urinating
- Frequent urination
- Feeling the need to urinate despite having an empty bladder
- Cloudy or strong-smelling urine
- Pelvic pain in women
- Rectal pain in men

**Causes:**
- Bacteria entering the urinary tract through the urethra
- E. coli bacteria from the digestive tract
- Sexual activity (particularly in women)
- Urinary tract abnormalities or blockages

**Risk Factors:**
- Female anatomy
- Sexual activity
- Certain types of birth control
- Menopause
- Urinary tract abnormalities
- Weakened immune system
- Catheter use

**Treatment Options:**
- Antibiotics
- Pain medication
- Increased fluid intake
- Cranberry products (for prevention)
- Probiotics

**When to See a Doctor:**
See a doctor if you experience painful urination, persistent urge to urinate, or pain in your side, back or lower abdomen. Seek immediate medical attention if you have fever, chills, back pain, nausea or vomiting alongside urinary symptoms.
"""
            
            # Build response based on prediction
            explanation = f"Based on your symptoms, I think you might have: **{prediction['disease']}**.\n\n{disease_info or ''}"
            
            # Add alternative possibilities if confidence is not very high
            if prediction['confidence'] < 70 and 'alternatives' in prediction and prediction['alternatives']:
                explanation += "\n\n**Other possibilities to consider:**\n"
                for alt in prediction['alternatives']:
                    explanation += f"- {alt['disease']} (confidence: {alt['confidence']}%)\n"
            
            explanation += "\n\n**Important:** This is not a medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
            
            # Transform prediction output to the expected format
            return {
                'label': prediction['disease'],
                'score': prediction['confidence'] / 100,  # Convert percentage to 0-1 scale
                'explanation': explanation
            }
        except Exception as e:
            logger.error(f"Error in classify_message: {str(e)}")
            return {
                'label': 'Error',
                'score': 0.0,
                'explanation': f"An error occurred while processing your message: {str(e)}"
            }

class SimpleHealthModelHandler:
    """A simple fallback implementation when the real health model is not available"""
    
    def __init__(self):
        """Initialize the simple health model handler"""
        logger.info("Initializing Simple Health Model Handler (fallback version)")
        
        # Define some condition patterns for basic rule-based matching
        self.condition_patterns = [
            (r'\b(headache|migraine|head\s+pain)\b', 'Possible Headache', 
             "Headaches can be caused by various factors including stress, dehydration, lack of sleep, or underlying conditions."),
            
            (r'\b(fever|high\s+temperature|chills)\b', 'Possible Fever', 
             "Fever is often a sign that your body is fighting an infection. It can be associated with flu, COVID-19, or other infections."),
            
            (r'\b(cough|coughing)\b', 'Respiratory Symptoms', 
             "Coughing can be due to various causes including allergies, infections, or irritants."),
            
            (r'\b(tired|fatigue|exhaustion|low\s+energy)\b', 'Fatigue', 
             "Fatigue can result from many factors including lack of sleep, stress, medical conditions, or lifestyle factors."),
            
            (r'\b(rash|itchy\s+skin|skin\s+irritation)\b', 'Skin Condition', 
             "Skin rashes may be caused by allergies, infections, heat, or underlying medical conditions."),
            
            (r'\b(nausea|vomit|queasy|upset\s+stomach)\b', 'Digestive Issues', 
             "Nausea can be caused by digestive disorders, infections, motion sickness, or certain medications."),
            
            (r'\b(diarrhea|loose\s+stool)\b', 'Digestive Issues', 
             "Diarrhea is often caused by viral or bacterial infections, food intolerances, or digestive disorders."),
            
            (r'\b(dizzy|lightheaded|vertigo)\b', 'Dizziness', 
             "Dizziness can be due to inner ear problems, low blood pressure, anemia, or other conditions."),
            
            (r'\b(sore\s+throat|throat\s+pain)\b', 'Sore Throat', 
             "Sore throats are commonly caused by viral infections like colds or flu, but can also result from allergies or bacterial infections."),
            
            (r'\b(back\s+pain|backache)\b', 'Back Pain', 
             "Back pain can be caused by muscle or ligament strain, bulging discs, arthritis, or skeletal irregularities."),
            
            (r'\b(chest\s+pain|chest\s+tightness)\b', 'Chest Pain', 
             "Chest pain can be serious and may be caused by heart problems, lung conditions, or digestive issues. Seek immediate medical attention for unexplained chest pain.")
        ]
        
        # Load disease info if available
        self.disease_info = {}
        try:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            disease_info_path = os.path.join(os.path.dirname(app_dir), 'data', 'processed', 'disease_info.json')
            
            if os.path.exists(disease_info_path):
                with open(disease_info_path, 'r') as f:
                    self.disease_info = json.load(f)
                logger.info(f"SimpleHealthModelHandler: Loaded detailed info for {len(self.disease_info)} diseases")
        except Exception as e:
            logger.warning(f"SimpleHealthModelHandler: Could not load disease info: {str(e)}")
        
    def is_model_loaded(self):
        """Simple model is always 'loaded'"""
        return True
        
    def predict(self, symptoms):
        """Make a simple rule-based prediction"""
        symptoms_lower = symptoms.lower()
        matches = []
        
        # Check for matches with our patterns
        for pattern, condition, description in self.condition_patterns:
            if re.search(pattern, symptoms_lower):
                matches.append((condition, description))
        
        # If we have matches, return the first one
        if matches:
            condition, description = matches[0]
            # Pick a random confidence between 60-85% for the matched condition
            confidence = random.uniform(60.0, 85.0)
            return {
                'disease': condition,
                'confidence': round(confidence, 2),
                'symptoms': symptoms,
                'explanation': description
            }
        
        # If no match, return a generic response
        return {
            'disease': 'Unspecified Condition',
            'confidence': 50.0,
            'symptoms': symptoms,
            'explanation': "I couldn't determine a specific condition from your symptoms. Please consult a healthcare provider for proper evaluation."
        }
    
    def get_health_response(self, query, include_disclaimer=True):
        """Get a response for health-related queries using the simple model"""
        # Analyze the query
        prediction = self.predict(query)
        
        # Generate response
        response = f"{prediction['explanation']}"
        
        if include_disclaimer:
            response += "\n\n**Important:** This is not a medical diagnosis. Please consult with a healthcare professional for proper evaluation and treatment."
        
        return {
            'response': response,
            'disease': prediction.get('disease'),
            'confidence': prediction.get('confidence', 0)
        }
        
    def classify_message(self, message):
        """Classify a message with the simple model"""
        prediction = self.predict(message)
        
        return {
            'label': prediction.get('disease', 'Unknown'),
            'score': prediction.get('confidence', 0) / 100,
            'explanation': prediction.get('explanation', "I couldn't determine a specific condition from your symptoms.")
        }


def create_health_model_handler(model_dir=None, model_type="bio_clinical_bert"):
    """Factory function to create a health model handler
    
    Args:
        model_dir: Directory where the model is stored
        model_type: Type of model to use ('bio_clinical_bert', 'pubmed_bert', 'biogpt', 'biobert')
        
    Returns:
        A health model handler instance
    """
    try:
        # Create the main model
        handler = HealthModelHandler(model_dir=model_dir, model_type=model_type)
        
        # Try to load the model
        if model_dir:
            success = handler.load_model(model_dir)
            if success:
                logger.info(f"Successfully loaded health model from {model_dir}")
                return handler
            else:
                logger.warning(f"Failed to load health model from {model_dir}, falling back to simple handler")
        else:
            logger.warning("No model directory specified, falling back to simple handler")
        
        # Fall back to simple handler
        return SimpleHealthModelHandler()
        
    except Exception as e:
        logger.error(f"Error creating health model handler: {str(e)}")
        # Fall back to the simple version
        return SimpleHealthModelHandler() 