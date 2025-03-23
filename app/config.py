import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///chatbot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Health model configuration 
    HEALTH_MODEL_DIR = os.environ.get('HEALTH_MODEL_DIR', None)  # Set via environment or None to use default
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'bio_clinical_bert')  # Default to Bio_ClinicalBERT
    
    # Legacy model configuration (kept for backward compatibility)
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'app', 'models', 'biobert_model')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Use bio_clinical_bert model in development by default
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'bio_clinical_bert')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Use simpler model for testing to speed up tests
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'biobert')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # This should be set in production
    
    # Use best model in production by default
    HEALTH_MODEL_TYPE = os.environ.get('HEALTH_MODEL_TYPE', 'bio_clinical_bert')