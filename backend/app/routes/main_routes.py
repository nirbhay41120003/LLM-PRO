from flask import Blueprint, jsonify, current_app

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'Health Assistant API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': {
            'health': 'POST /health - Analyze symptoms',
            'analyze': 'POST /analyze - Get disease prediction',
            'rag_search': 'POST /rag-search - Search medical knowledge',
            'general': 'POST /general - General health queries',
            'status': 'GET /api/status - Check API status'
        }
    })

@main_bp.route('/api/status')
def api_status():
    """API status check endpoint"""
    model_loaded = False
    model_type = 'unknown'
    
    try:
        if hasattr(current_app, 'health_model') and current_app.health_model:
            model_loaded = current_app.health_model.is_model_loaded()
            model_type = getattr(current_app.health_model, 'model_type', 'biobert')
    except:
        pass
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_type': model_type,
        'rag_available': hasattr(current_app, 'rag_system') and current_app.rag_system is not None
    }) 