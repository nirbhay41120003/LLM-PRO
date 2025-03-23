import logging
from flask import current_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """Class for handling general queries with a rule-based approach"""
    
    def __init__(self):
        """Initialize the rule-based LLM handler"""
        # Initialize response templates
        self.templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Greetings! How may I be of service?"
            ],
            "farewell": [
                "Goodbye! Take care.",
                "See you later! Have a great day.",
                "Farewell! Feel free to return if you have more questions."
            ],
            "thanks": [
                "You're welcome! Is there anything else I can help with?",
                "My pleasure! Let me know if you need anything else.",
                "Happy to help! Feel free to ask more questions."
            ],
            "default": [
                "I'm a simple rule-based assistant without external API access. I can help with basic information, but my knowledge is limited to what's been programmed. For more advanced queries, please try the health assistant mode which uses our medical model.",
                "As a standalone assistant without API access, I have limited knowledge. I can try to help with some questions, but for medical inquiries, please use the health assistant mode which leverages our specialized model.",
                "I'm operating without external API access, so my responses are based on pre-defined patterns. For health-related questions, I recommend switching to the health assistant mode for more specialized information."
            ]
        }
        
        # Initialize common knowledge base
        self.knowledge_base = {
            "weather": "I don't have access to real-time weather data, but I can suggest checking a weather service or app for the most current information.",
            "time": "I don't have access to your current time. Your device should display the current time in your location.",
            "date": "I don't have access to the current date. Your device should display today's date.",
            "name": "I'm a simple rule-based assistant created to help with basic questions and health inquiries.",
            "age": "I'm just a computer program, so I don't have an age in the human sense.",
            "creator": "I was created by a development team as part of a health chatbot project.",
            "purpose": "My purpose is to provide assistance with health-related questions and general inquiries.",
            "capabilities": "I can help answer basic questions and provide health information using our medical model. However, I operate without external API access, so my knowledge is limited to what's been programmed.",
            "limitations": "Since I don't use external APIs, I can't access real-time data, search the web, or learn from new information unless I'm updated by my developers.",
            "help": "You can ask me general questions or switch to the health assistant mode for medical inquiries. Just type your question, and I'll do my best to assist you.",
            "food": "I don't have personal preferences for food since I'm a computer program. But a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins is generally recommended for good health.",
            "exercise": "Regular physical activity is important for health. The WHO recommends at least 150 minutes of moderate-intensity exercise per week for adults.",
            "sleep": "Most adults need 7-9 hours of quality sleep per night. Good sleep hygiene includes maintaining a regular sleep schedule and creating a restful environment.",
            "stress": "Managing stress is important for overall health. Techniques like deep breathing, meditation, physical activity, and maintaining social connections can help reduce stress."
        }
        
        # Initialize topic handlers
        self.topic_handlers = {
            "health": self._handle_health_topic,
            "technology": self._handle_technology_topic,
            "science": self._handle_science_topic,
            "entertainment": self._handle_entertainment_topic,
            "history": self._handle_history_topic
        }
    
    def get_response(self, query, chat_history=None):
        """Get response based on the query using rule-based matching"""
        query_lower = query.lower().strip()
        
        # Check for greetings
        if self._is_greeting(query_lower):
            return self._get_random_template("greeting")
        
        # Check for farewells
        if self._is_farewell(query_lower):
            return self._get_random_template("farewell")
        
        # Check for thanks/gratitude
        if self._is_thanks(query_lower):
            return self._get_random_template("thanks")
        
        # Check knowledge base for direct matches
        for key, response in self.knowledge_base.items():
            if key in query_lower or self._is_asking_about(query_lower, key):
                return response
        
        # Detect topic and use appropriate handler
        topic = self._detect_topic(query_lower)
        if topic in self.topic_handlers:
            return self.topic_handlers[topic](query_lower)
        
        # If no specific rule matches, return a default response
        return self._get_random_template("default")
    
    def _is_greeting(self, text):
        """Check if the text is a greeting"""
        greeting_phrases = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"]
        return any(phrase in text for phrase in greeting_phrases)
    
    def _is_farewell(self, text):
        """Check if the text is a farewell"""
        farewell_phrases = ["bye", "goodbye", "see you", "farewell", "see you later", "take care", "have a good day", "have a nice day"]
        return any(phrase in text for phrase in farewell_phrases)
    
    def _is_thanks(self, text):
        """Check if the text expresses gratitude"""
        thanks_phrases = ["thank", "thanks", "appreciate", "grateful", "gratitude"]
        return any(phrase in text for phrase in thanks_phrases)
    
    def _is_asking_about(self, text, topic):
        """Check if the text is asking about a specific topic"""
        question_patterns = [
            f"what is {topic}",
            f"what's {topic}",
            f"tell me about {topic}",
            f"know about {topic}",
            f"information on {topic}",
            f"info about {topic}"
        ]
        return any(pattern in text for pattern in question_patterns)
    
    def _get_random_template(self, template_type):
        """Get a random response template for a given type"""
        import random
        templates = self.templates.get(template_type, self.templates["default"])
        return random.choice(templates)
    
    def _detect_topic(self, text):
        """Detect the main topic of the query"""
        topic_keywords = {
            "health": ["health", "disease", "medical", "doctor", "hospital", "symptom", "treatment", "medicine", "drug", "pill", "therapy", "diet", "exercise", "nutrition"],
            "technology": ["computer", "software", "hardware", "internet", "app", "technology", "device", "smartphone", "laptop", "code", "programming", "website", "digital"],
            "science": ["science", "physics", "chemistry", "biology", "astronomy", "planet", "star", "atom", "molecule", "experiment", "theory", "research", "discovery"],
            "entertainment": ["movie", "film", "tv", "television", "show", "music", "song", "artist", "actor", "actress", "celebrity", "game", "play", "book", "novel"],
            "history": ["history", "past", "ancient", "century", "year", "historical", "war", "revolution", "king", "queen", "president", "empire", "civilization"]
        }
        
        # Count keyword matches for each topic
        topic_scores = {topic: 0 for topic in topic_keywords}
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topic_scores[topic] += 1
        
        # Get the topic with the highest score
        max_score = max(topic_scores.values()) if topic_scores else 0
        if max_score > 0:
            for topic, score in topic_scores.items():
                if score == max_score:
                    return topic
        
        # Default topic if no keywords match
        return "general"
    
    def _handle_health_topic(self, query):
        """Handle health-related queries with general information"""
        health_responses = {
            "exercise": "Regular physical activity has many benefits including improved cardiovascular health, stronger muscles and bones, and better mental health. Aim for at least 150 minutes of moderate activity per week.",
            "diet": "A balanced diet typically includes fruits, vegetables, whole grains, lean proteins, and healthy fats. It's recommended to limit processed foods, added sugars, and excessive salt.",
            "sleep": "Quality sleep is essential for health. Most adults need 7-9 hours per night. Consistent sleep schedules and a comfortable sleep environment can help improve sleep quality.",
            "stress": "Chronic stress can negatively impact health. Stress management techniques include deep breathing, meditation, regular exercise, and maintaining social connections.",
            "hydration": "Staying hydrated is important for overall health. The recommended water intake varies by individual, but a common guideline is about 8 cups (64 ounces) of water daily.",
            "vitamins": "Vitamins and minerals are essential nutrients that support various bodily functions. A balanced diet usually provides adequate amounts, but some people may benefit from supplements.",
            "headache": "Headaches can be caused by various factors including stress, dehydration, lack of sleep, or eye strain. For persistent or severe headaches, it's advisable to consult a healthcare provider."
        }
        
        # Check for specific health topics
        for topic, response in health_responses.items():
            if topic in query:
                return response
        
        # If no specific health topic is detected
        return "For specific health concerns or symptom analysis, I recommend switching to the health assistant mode which uses our specialized medical model for more accurate information."
    
    def _handle_technology_topic(self, query):
        """Handle technology-related queries"""
        return "I have limited knowledge about technology topics. I'm a simple rule-based assistant without external API access, so I can't provide detailed or current information about technology."
    
    def _handle_science_topic(self, query):
        """Handle science-related queries"""
        return "Science is a vast field that uses observation and experimentation to understand the natural world. Without external API access, I can only provide basic information on scientific topics."
    
    def _handle_entertainment_topic(self, query):
        """Handle entertainment-related queries"""
        return "I don't have access to current entertainment information or databases of movies, music, or other media. For up-to-date entertainment information, you might want to check a dedicated entertainment website or service."
    
    def _handle_history_topic(self, query):
        """Handle history-related queries"""
        return "History is the study of past events. I have limited knowledge about historical events and figures without access to external references or databases."


# Factory function to create LLM handler
def create_llm_handler():
    """Create an instance of the rule-based LLM handler"""
    return LLMHandler() 