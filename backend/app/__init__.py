import os
import logging
from flask import Flask
from flask_cors import CORS
from app.config import Config, DevelopmentConfig, ProductionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_class=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for frontend requests
    CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','),
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization'])
    
    # Load appropriate config
    if not config_class:
        env = os.environ.get('FLASK_ENV', 'development')
        if env == 'production':
            config_class = ProductionConfig
        else:
            config_class = DevelopmentConfig
    
    app.config.from_object(config_class)
    
    # Initialize the new integrated health model
    with app.app_context():
        try:
            from app.utils.health_model_v2 import IntegratedHealthModel
            
            # Get model configuration
            model_type = app.config.get('HEALTH_MODEL_TYPE', 'biobert')
            model_dir = app.config.get('HEALTH_MODEL_DIR')
            
            # Create the integrated health model
            app.health_model = IntegratedHealthModel(
                model_dir=model_dir,
                model_type=model_type
            )
            
            # Initialize the model
            if app.health_model.initialize():
                logger.info(f"Integrated Health Model ({model_type}) initialized successfully")
            else:
                logger.warning("Health model initialized but classifier not loaded. Run training first.")
                
        except Exception as e:
            logger.error(f"Could not initialize health model: {str(e)}")
            # Fallback to simple model
            try:
                from app.utils.health_model import create_health_model_handler
                app.health_model = create_health_model_handler()
                logger.info("Fallback health model initialized")
            except:
                app.health_model = None
                logger.error("No health model available")
    
    # Initialize RAG system
    with app.app_context():
        try:
            from app.utils.rag_system import RAGSystem
            app.rag_system = RAGSystem()
            logger.info("RAG system initialized")
        except Exception as e:
            app.rag_system = None
            logger.warning(f"Could not initialize RAG system: {str(e)}")
    
    # Initialize report analyzer
    with app.app_context():
        try:
            from app.utils.report_analyzer import create_report_analyzer
            app.report_analyzer = create_report_analyzer()
            logger.info("Report analyzer initialized")
        except Exception as e:
            app.report_analyzer = None
            logger.warning(f"Could not initialize report analyzer: {str(e)}")
    
    # Register blueprints
    from app.routes.main_routes import main_bp
    from app.routes.chat_routes import chat_bp
    from app.routes.auth_routes import auth_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(auth_bp)
    
    return app