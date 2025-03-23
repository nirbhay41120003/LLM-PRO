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
    
    # Initialize report analyzer if possible
    try:
        from app.utils.report_analyzer import create_report_analyzer
        app.report_analyzer = create_report_analyzer()
        logger.info("Report analyzer initialized successfully")
    except Exception as e:
        app.report_analyzer = None
        logger.warning(f"Could not initialize report analyzer: {str(e)}")
    
    # Initialize health model if possible
    try:
        from app.utils.health_model import create_health_model_handler
        app.health_model = create_health_model_handler()
        if hasattr(app.health_model, 'is_model_loaded') and app.health_model.is_model_loaded():
            logger.info("Full health model initialized successfully")
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