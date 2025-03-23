import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from app import create_app

def setup_logging(app):
    """Setup logging configuration"""
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/health_chatbot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Health Chatbot startup')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the Flask application')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), 
                       help='Port to run the app on')
    parser.add_argument('--host', type=str, default=os.environ.get('HOST', '0.0.0.0'), 
                       help='Host to run the app on')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    app = create_app()
    
    # Setup logging
    setup_logging(app)
    
    # Production settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(
        debug=debug_mode,
        host=args.host,
        port=args.port,
        threaded=True
    )