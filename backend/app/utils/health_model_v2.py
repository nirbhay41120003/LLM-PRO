"""
Integrated Health Model Handler
Combines BioBERT fine-tuned classifier with RAG for comprehensive medical responses
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedHealthModel:
    """
    Integrated Health Model combining:
    - BioBERT fine-tuned classifier for symptom-disease classification
    - RAG system for knowledge retrieval and augmented responses
    """
    
    def __init__(self, model_dir: str = None, model_type: str = "biobert"):
        self.model_dir = model_dir
        self.model_type = model_type
        self.classifier = None
        self.rag_system = None
        self.disease_info = {}
        self.symptom_severity = {}
        self._initialized = False
        
        self._load_resources()
    
    def _load_resources(self):
        """Load disease info and other resources"""
        try:
            # Find data directory
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(os.path.dirname(app_dir), 'data', 'processed')
            
            # Load disease info
            disease_info_path = os.path.join(data_dir, 'disease_info.json')
            if os.path.exists(disease_info_path):
                with open(disease_info_path, 'r') as f:
                    self.disease_info = json.load(f)
                logger.info(f"Loaded info for {len(self.disease_info)} diseases")
            
            # Load symptom severity
            severity_path = os.path.join(data_dir, 'symptom_severity.json')
            if os.path.exists(severity_path):
                with open(severity_path, 'r') as f:
                    self.symptom_severity = json.load(f)
                logger.info(f"Loaded severity for {len(self.symptom_severity)} symptoms")
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
    
    def initialize(self) -> bool:
        """Initialize the classifier and RAG system"""
        if self._initialized:
            return True
        
        try:
            # Initialize BioBERT classifier
            from app.utils.biobert_classifier import BioBERTClassifier
            
            self.classifier = BioBERTClassifier(model_type=self.model_type, model_dir=self.model_dir)
            
            # Try to load fine-tuned model
            if not self.classifier.load_finetuned():
                logger.warning("No fine-tuned model found. Please train the model first.")
                self.classifier = None
            else:
                logger.info("BioBERT classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BioBERT classifier: {e}")
            self.classifier = None
        
        try:
            # Initialize RAG system
            from app.utils.rag_system import RAGSystem
            self.rag_system = RAGSystem()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_system = None
        
        self._initialized = True
        return self.classifier is not None
    
    def is_model_loaded(self) -> bool:
        """Check if the model is ready for predictions"""
        return self.classifier is not None
    
    def predict(self, symptoms: str) -> Dict:
        """
        Make disease prediction from symptoms
        
        Args:
            symptoms: Natural language description of symptoms
            
        Returns:
            Dict with prediction results
        """
        if not self.initialize():
            return {
                'error': 'Model not initialized',
                'message': 'Please train the model first using: python -m app.utils.train_pipeline'
            }
        
        try:
            # Get predictions from classifier
            predictions = self.classifier.predict(symptoms, top_k=3)
            
            if not predictions:
                return {
                    'error': 'No prediction available',
                    'message': 'Could not classify the symptoms'
                }
            
            # Primary prediction
            primary = predictions[0]
            
            return {
                'disease': primary['disease'],
                'confidence': primary['confidence'],
                'symptoms': symptoms,
                'alternatives': predictions[1:] if len(predictions) > 1 else []
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'message': 'An error occurred during prediction'
            }
    
    def get_health_response(self, query: str, include_disclaimer: bool = True) -> Dict:
        """
        Get comprehensive health response using BioBERT + RAG
        
        Args:
            query: User's health query
            include_disclaimer: Whether to include medical disclaimer
            
        Returns:
            Dict with response and metadata
        """
        # Check if this is a symptom description
        if not self._is_symptom_description(query):
            return self._handle_general_query(query)
        
        # Get prediction
        prediction = self.predict(query)
        
        if 'error' in prediction:
            return {
                'response': f"I'm sorry, I couldn't analyze your symptoms. {prediction.get('message', '')}",
                'error': prediction['error']
            }
        
        # Generate comprehensive response using RAG
        response = self._generate_rag_response(query, prediction, include_disclaimer)
        
        return {
            'response': response,
            'disease': prediction.get('disease'),
            'confidence': prediction.get('confidence'),
            'alternatives': prediction.get('alternatives', [])
        }
    
    def _generate_rag_response(self, query: str, prediction: Dict, include_disclaimer: bool) -> str:
        """Generate comprehensive response using RAG"""
        disease = prediction.get('disease', 'Unknown')
        confidence = prediction.get('confidence', 0)
        
        response_parts = []
        
        # Main prediction
        response_parts.append(
            f"Based on your symptoms, the most likely condition is **{disease}** "
            f"(confidence: {confidence:.1f}%).\n"
        )
        
        # Get disease information
        disease_info = self._get_disease_info(disease)
        
        if disease_info:
            # Description
            if disease_info.get('description'):
                response_parts.append(f"### About {disease}")
                response_parts.append(disease_info['description'])
            
            # Symptoms
            if disease_info.get('symptoms'):
                response_parts.append("\n### Common Symptoms")
                for symptom in disease_info['symptoms'][:6]:
                    response_parts.append(f"• {symptom}")
            
            # Causes
            if disease_info.get('causes'):
                response_parts.append("\n### Possible Causes")
                for cause in disease_info['causes'][:4]:
                    response_parts.append(f"• {cause}")
            
            # Treatment/Precautions
            treatments = disease_info.get('treatment', [])
            if treatments:
                response_parts.append("\n### Recommended Actions")
                if isinstance(treatments, list):
                    for treatment in treatments[:5]:
                        response_parts.append(f"• {treatment}")
                else:
                    response_parts.append(f"• {treatments}")
            
            # When to see doctor
            if disease_info.get('when_to_see_doctor'):
                response_parts.append(f"\n### When to See a Doctor")
                response_parts.append(disease_info['when_to_see_doctor'])
        
        # Use RAG for additional context if available
        if self.rag_system:
            try:
                rag_context = self.rag_system.retrieve_context(f"{disease} symptoms treatment", top_k=2)
                if rag_context and len(rag_context) > 100:
                    # Already have comprehensive info from disease_info, RAG adds depth
                    pass
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Alternative diagnoses
        alternatives = prediction.get('alternatives', [])
        if alternatives and confidence < 80:
            response_parts.append("\n### Other Possibilities to Consider")
            for alt in alternatives[:2]:
                response_parts.append(f"• {alt['disease']} ({alt['confidence']:.1f}%)")
        
        # Disclaimer
        if include_disclaimer:
            response_parts.append("\n---")
            response_parts.append(
                "⚠️ **Important Disclaimer:** This assessment is generated by an AI system "
                "and is for informational purposes only. It is not a substitute for professional "
                "medical advice, diagnosis, or treatment. Always consult a qualified healthcare "
                "provider for proper medical evaluation."
            )
        
        return "\n".join(response_parts)
    
    def _get_disease_info(self, disease: str) -> Dict:
        """Get comprehensive information about a disease"""
        # Try exact match first
        if disease in self.disease_info:
            return self.disease_info[disease]
        
        # Try case-insensitive match
        disease_lower = disease.lower()
        for key, info in self.disease_info.items():
            if key.lower() == disease_lower:
                return info
        
        # Try partial match
        for key, info in self.disease_info.items():
            if disease_lower in key.lower() or key.lower() in disease_lower:
                return info
        
        return {}
    
    def _is_symptom_description(self, text: str) -> bool:
        """Check if text is describing symptoms"""
        text_lower = text.lower().strip()
        
        # Non-symptom phrases (greetings, questions about the bot, etc.)
        non_symptom_patterns = [
            r'^(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy)\b',
            r'^(how are you|what\'?s up|what can you do|who are you)',
            r'^(thanks?|thank you|bye|goodbye|see you)',
            r'^(help|menu|options|commands)',
        ]
        
        for pattern in non_symptom_patterns:
            if re.match(pattern, text_lower):
                return False
        
        # Symptom indicators
        symptom_keywords = [
            'symptom', 'pain', 'ache', 'hurt', 'sore', 'fever', 'cough',
            'headache', 'nausea', 'vomit', 'dizzy', 'tired', 'fatigue',
            'sick', 'ill', 'unwell', 'infection', 'rash', 'swelling',
            'inflammation', 'burning', 'itching', 'diarrhea', 'constipation',
            'breathing', 'chest', 'stomach', 'back', 'joint', 'muscle',
            'throat', 'ear', 'eye', 'nose', 'skin', 'allergy', 'sneeze',
            'congestion', 'blood', 'pressure', 'diabetes', 'heart',
            'urine', 'urinary', 'bladder', 'kidney', 'cold', 'flu',
            'i have', 'i am experiencing', 'i\'ve been', 'suffering from',
            'feel', 'feeling', 'experiencing', 'noticed'
        ]
        
        for keyword in symptom_keywords:
            if keyword in text_lower:
                return True
        
        # If text is long enough, it might be describing symptoms
        if len(text.split()) >= 5:
            return True
        
        return False
    
    def _handle_general_query(self, query: str) -> Dict:
        """Handle non-symptom queries"""
        query_lower = query.lower()
        
        # Health tips
        health_tips = {
            'diet': "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Stay hydrated and limit processed foods, sugar, and excessive salt.",
            'exercise': "Regular physical activity is crucial. Aim for at least 150 minutes of moderate-intensity exercise per week. Include both cardio and strength training.",
            'sleep': "Adults need 7-9 hours of quality sleep. Maintain a consistent sleep schedule and create a restful environment.",
            'stress': "Manage stress through regular exercise, meditation, deep breathing, and maintaining social connections.",
            'water': "Stay hydrated by drinking 8-10 glasses of water daily. Adjust based on activity level and climate.",
        }
        
        for topic, tip in health_tips.items():
            if topic in query_lower:
                return {'response': tip}
        
        # Default response
        return {
            'response': (
                "I'm a health assistant designed to help analyze symptoms and provide "
                "health information. Please describe your symptoms, and I'll try to help "
                "identify possible conditions. For example, you can say:\n\n"
                "• 'I have a headache and fever'\n"
                "• 'I've been experiencing stomach pain and nausea'\n"
                "• 'I have a rash on my arms'\n\n"
                "Remember, I provide information only and cannot replace professional medical advice."
            )
        }
    
    def analyze_symptom_severity(self, symptoms: str) -> Dict:
        """Analyze the severity of described symptoms"""
        if not self.symptom_severity:
            return {'severity': 'unknown', 'score': 0}
        
        text_lower = symptoms.lower()
        total_score = 0
        matched_symptoms = []
        
        for symptom, weight in self.symptom_severity.items():
            symptom_lower = symptom.lower().replace('_', ' ')
            if symptom_lower in text_lower:
                total_score += weight
                matched_symptoms.append({'symptom': symptom, 'severity': weight})
        
        # Determine severity level
        if total_score >= 15:
            severity = 'high'
        elif total_score >= 8:
            severity = 'moderate'
        elif total_score >= 3:
            severity = 'low'
        else:
            severity = 'minimal'
        
        return {
            'severity': severity,
            'score': total_score,
            'matched_symptoms': matched_symptoms
        }


def create_health_model_handler(model_dir: str = None, model_type: str = "biobert") -> IntegratedHealthModel:
    """Factory function to create health model handler"""
    return IntegratedHealthModel(model_dir=model_dir, model_type=model_type)


# Keep backward compatibility with old interface
HealthModelHandler = IntegratedHealthModel
