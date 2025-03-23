import os
import logging
from flask import Flask
from app.config import Config, DevelopmentConfig, ProductionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_class=None):
    """Create and configure the Flask application"""
    app = Flask(__name__, static_folder='../static', template_folder='../templates')
    
    # Load appropriate config
    if not config_class:
        env = os.environ.get('FLASK_ENV', 'development')
        if env == 'production':
            config_class = ProductionConfig
        else:
            config_class = DevelopmentConfig
    
    app.config.from_object(config_class)
    
    # Add HuggingFace API key to config
    app.config['HUGGINGFACE_API_KEY'] = os.environ.get('HUGGINGFACE_API_KEY')
    if not app.config['HUGGINGFACE_API_KEY']:
        logger.warning("HuggingFace API key not found in environment variables")
    else:
        logger.info("HuggingFace API key loaded from environment variables")
    
    # Initialize report analyzer if possible
    with app.app_context():
        try:
            from app.utils.report_analyzer import create_report_analyzer
            app.report_analyzer = create_report_analyzer()
            logger.info("Report analyzer initialized successfully")
        except Exception as e:
            app.report_analyzer = None
            logger.warning(f"Could not initialize report analyzer: {str(e)}")
    
    # Initialize health model if possible with the improved model
    try:
        from app.utils.health_model import create_health_model_handler
        
        # Get model directory - first check for configuration
        model_dir = app.config.get('HEALTH_MODEL_DIR')
        
        # If not in config, use default location
        if not model_dir:
            # Default is to use bio_clinical_bert model directory
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'app', 'models', 'bio_clinical_bert')
            
            # If bio_clinical_bert doesn't exist, fall back to original biobert_model
            if not os.path.exists(model_dir):
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       'app', 'models', 'biobert_model')
        
        # Get model type - first check for configuration
        model_type = app.config.get('HEALTH_MODEL_TYPE', 'bio_clinical_bert')
        
        # Create the health model handler with the better model
        app.health_model = create_health_model_handler(model_dir=model_dir, model_type=model_type)
        
        if hasattr(app.health_model, 'is_model_loaded') and app.health_model.is_model_loaded():
            logger.info(f"Full health model ({model_type}) initialized successfully")
        else:
            logger.info("Simple fallback health model initialized")
    except Exception as e:
        app.health_model = None
        logger.warning(f"Could not initialize health model: {str(e)}. Health classification will not be available.")
    
    # Register blueprints
    from app.routes.main_routes import main_bp
    from app.routes.chat_routes import chat_bp
    from app.routes.auth_routes import auth_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(auth_bp)
    
    return app