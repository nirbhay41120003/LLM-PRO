from flask import Blueprint, render_template, jsonify, current_app

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'health_model_loaded': hasattr(current_app, 'health_model') and current_app.health_model.is_model_loaded()
    }) 