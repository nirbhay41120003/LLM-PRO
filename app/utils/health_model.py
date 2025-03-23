import torch
import os
import logging
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthModelHandler:
    """Handler for the health model predictions"""
    
    def __init__(self, model_dir=None):
        """Initialize the health model handler"""
        self.model = None
        self.tokenizer = None
        self.disease_classes = []
        self.model_dir = model_dir
        
    def load_model(self, model_dir=None):
        """Load the model and tokenizer"""
        try:
            # Use provided model_dir or the one from config
            model_dir = model_dir or self.model_dir
            
            if not model_dir or not os.path.exists(model_dir):
                logger.error(f"Model directory not found at {model_dir}")
                return False
            
            from transformers import BertForSequenceClassification, AutoTokenizer
            
            logger.info(f"Loading model from {model_dir}")
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
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
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Get disease name
            if self.disease_classes and predicted_class < len(self.disease_classes):
                disease = self.disease_classes[predicted_class]
            else:
                disease = f"Disease_{predicted_class}"
            
            # Get confidence score
            confidence = torch.nn.functional.softmax(predictions, dim=1)[0][predicted_class].item()
            
            logger.info(f"Prediction result: Disease={disease}, Confidence={round(confidence * 100, 2)}%")
            
            return {
                'disease': disease,
                'confidence': round(confidence * 100, 2),
                'symptoms': symptoms
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
                return {
                    'error': prediction['error'],
                    'response': f"I'm sorry, but I couldn't analyze your symptoms. {prediction['error']}"
                }
            
            # Format a nice response with the prediction
            disease = prediction['disease']
            confidence = prediction['confidence']
            
            # Generate a more comprehensive response
            response = f"Based on your symptoms, I think you might have: **{disease}** (confidence: {confidence}%).\n\n"
            
            # Add some common information about the disease
            disease_info = self._get_disease_info(disease)
            if disease_info:
                response += disease_info + "\n\n"
            
            if include_disclaimer:
                response += "**Important:** This is not a medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
            
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
            'nose', 'skin', 'allergy', 'sneeze', 'runny', 'congestion'
        ]
        
        text_lower = text.lower()
        logger.info(f"Checking if text is symptom description: '{text_lower}'")
        
        # Check for presence of symptom indicators
        for indicator in symptom_indicators:
            if indicator in text_lower:
                logger.info(f"Found symptom indicator: '{indicator}'")
                return True
        
        # Check if it's a longer description (symptom descriptions tend to be longer)
        word_count = len(text.split())
        logger.info(f"Word count: {word_count}")
        
        if word_count >= 3:  # Lower threshold from 5 to 3 words to catch more symptom descriptions
            logger.info("Word count >= 3, treating as symptom description")
            return True
        
        logger.info("Not detected as symptom description")
        return False
    
    def _get_general_health_response(self, query):
        """Handle general health-related questions"""
        # Simple template responses for common health questions
        health_templates = {
            'covid': "COVID-19 symptoms may include fever, cough, and shortness of breath. If you're experiencing these symptoms, please get tested and consult a healthcare provider.",
            'vaccine': "Vaccines are an important way to protect against serious diseases. They work by triggering an immune response in your body. Always consult with your healthcare provider about which vaccines are right for you.",
            'headache': "Headaches can be caused by stress, dehydration, lack of sleep, or more serious conditions. For persistent or severe headaches, please consult a healthcare provider.",
            'diet': "A balanced diet typically includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Individual dietary needs may vary based on age, activity level, and health conditions.",
            'exercise': "Regular physical activity has numerous health benefits, including improved cardiovascular health, weight management, and better mental health. It's recommended to get at least 150 minutes of moderate activity per week.",
            'sleep': "Adults typically need 7-9 hours of quality sleep per night. Poor sleep can impact your physical and mental health. If you're having persistent sleep issues, consider consulting a healthcare provider.",
            'stress': "Chronic stress can have negative effects on your health. Stress management techniques include deep breathing, meditation, physical activity, and maintaining social connections."
        }
        
        # Check for matches with templates
        query_lower = query.lower()
        logger.info(f"Checking for general health template matches for: '{query_lower}'")
        
        for keyword, response in health_templates.items():
            if keyword in query_lower:
                logger.info(f"Found template match for keyword: '{keyword}'")
                return response
        
        # Default response if no specific template matches
        logger.info("No template match found, returning default response")
        return "I'm a health assistant trained to provide general health information and analyze symptoms. I can help identify possible health conditions based on symptoms you describe, but I'm not a replacement for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."

    def _get_disease_info(self, disease):
        """Get additional information about a disease"""
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

    def classify_message(self, message):
        """
        Classify a message using the trained model
        
        Args:
            message: The user's message
            
        Returns:
            dict: Classification result with label and score
        """
        try:
            # Make prediction using the existing predict method
            prediction = self.predict(message)
            
            if 'error' in prediction:
                logger.error(f"Error in prediction: {prediction['error']}")
                return {
                    'label': 'Error',
                    'score': 0.0,
                    'explanation': f"Could not classify the message: {prediction['error']}"
                }
            
            # Transform prediction output to the expected format
            return {
                'label': prediction['disease'],
                'score': prediction['confidence'] / 100,  # Convert percentage to 0-1 scale
                'explanation': f"Based on your symptoms, I think you might have: **{prediction['disease']}**.\n\n{self._get_disease_info(prediction['disease']) or ''}\n\n**Important:** This is not a medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
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
             "Fatigue can be caused by lack of sleep, stress, poor diet, or various medical conditions."),
            
            (r'\b(pain|ache|sore)\b', 'General Pain', 
             "Pain can be caused by injury, inflammation, or various medical conditions."),
            
            (r'\b(stomach|nausea|vomit|diarrhea)\b', 'Gastrointestinal Issues', 
             "Stomach issues can be caused by food poisoning, viruses, stress, or digestive conditions.")
        ]
        
        logger.info("Simple Health Model Handler initialized successfully")
    
    def is_model_loaded(self):
        """Check if model is loaded - always returns False for fallback model"""
        return False
    
    def classify_message(self, message):
        """
        Classify a message using simple rule-based matching
        
        Args:
            message: The user's message
            
        Returns:
            dict: Classification result with label and confidence
        """
        message = message.lower()
        
        # Try to match patterns
        matched_conditions = []
        for pattern, label, explanation in self.condition_patterns:
            if re.search(pattern, message):
                matched_conditions.append((label, explanation))
        
        if matched_conditions:
            # Pick the first match (could be enhanced to pick the best match)
            label, explanation = matched_conditions[0]
            confidence = random.uniform(0.4, 0.6)  # Random confidence to indicate uncertainty
            
            return {
                'label': label,
                'score': confidence,
                'explanation': f"{explanation}\n\n**Important Note**: This is a very basic assessment and should not be considered a medical diagnosis. The AI model that provides more accurate health classifications is not currently available. Please consult a healthcare professional for proper diagnosis and treatment."
            }
        else:
            # No match found
            return {
                'label': 'Unrecognized Symptoms',
                'score': 0.3,
                'explanation': "I couldn't confidently identify specific health conditions from your message.\n\n**Important Note**: The AI model that provides more accurate health classifications is not currently available. Please consult a healthcare professional for proper diagnosis and treatment."
            }
    
    def get_health_response(self, message):
        """
        Get a response for a health-related message
        
        Args:
            message: The user's message
            
        Returns:
            dict: Response data
        """
        classification = self.classify_message(message)
        
        response = f"Health Classification (Simplified): {classification['label']} (Confidence: {classification['score']:.2f})\n\n{classification['explanation']}\n\n**Please note**: This is a fallback response as the full health classification model is not available."
        
        return {
            'response': response,
            'label': classification['label'],
            'score': classification['score'],
            'explanation': classification['explanation'],
            'is_fallback': True
        }

def create_health_model_handler():
    """Create and return a health model handler instance"""
    try:
        # Try to import the required libraries
        from transformers import BertForSequenceClassification, AutoTokenizer
        logger.info("Using real health model with transformers")
        
        # If import succeeds, create and return the real model
        # Don't use current_app here - it's only available inside request context
        # Get model directory from a more predictable location
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(os.path.dirname(app_dir), 'app', 'models', 'biobert_model')
        
        if not model_dir or not os.path.exists(model_dir):
            logger.warning(f"Model directory not found at {model_dir}, using fallback model")
            return SimpleHealthModelHandler()
            
        # Create real model handler
        handler = HealthModelHandler(model_dir=model_dir)
        load_success = handler.load_model()
        
        if load_success:
            logger.info(f"Successfully loaded health model from {model_dir}")
            return handler
        else:
            logger.warning("Failed to load health model, using fallback model")
            return SimpleHealthModelHandler()
    
    except ImportError as e:
        # Fall back to simple implementation if transformers is not available
        logger.warning(f"Could not import transformers: {str(e)}. Using simple fallback health model.")
        return SimpleHealthModelHandler() 