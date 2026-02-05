import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///chatbot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Health model configuration 
    # Default to biobert_finetuned directory where trained model is saved
    HEALTH_MODEL_DIR = os.environ.get('HEALTH_MODEL_DIR', 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                     'app', 'models', 'biobert_finetuned'))
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'biobert')  # Default to BioBERT
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Use biobert model in development by default
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'biobert')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Use biobert model for testing
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'biobert')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # This should be set in production
    
    # Use biobert in production by default
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'biobert')