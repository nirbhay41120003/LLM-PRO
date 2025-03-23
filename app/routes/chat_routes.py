from flask import Blueprint, request, jsonify, current_app
from app.models.database import db, Chat, Message
import datetime
import logging
import json
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/health', methods=['POST'])
def health_chat():
    """Handle health-related queries"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Get health model response
        health_model = current_app.health_model
        response_data = health_model.get_health_response(user_message)
        
        # Save to database if chat_id is provided
        if chat_id:
            save_messages_to_db(chat_id, user_message, response_data)
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in health chat: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/general', methods=['POST'])
def general_chat():
    """Handle general queries using an LLM"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Get chat history for context if chat_id is provided
        chat_history = []
        if chat_id:
            chat_history = get_chat_history(chat_id)
        
        # Import LLM handler
        from app.utils.llm_handler import create_llm_handler
        llm_handler = create_llm_handler()
        
        # Get LLM response
        response = llm_handler.get_response(user_message, chat_history)
        
        # Save to database if chat_id is provided
        if chat_id:
            save_to_db(chat_id, 'general', user_message, response)
        
        return jsonify({
            'response': response
        })
    except Exception as e:
        logger.error(f"Error in general chat: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/history', methods=['GET'])
def get_history():
    """Get chat history"""
    limit = request.args.get('limit', 10, type=int)
    
    try:
        # Get most recent chats
        chats = Chat.query.order_by(Chat.updated_at.desc()).limit(limit).all()
        return jsonify({
            'chats': [chat.to_dict() for chat in chats]
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/history/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    """Get messages for a specific chat"""
    try:
        chat = Chat.query.get(chat_id)
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp).all()
        return jsonify({
            'chat': chat.to_dict(),
            'messages': [message.to_dict() for message in messages]
        })
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/history/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat and its messages"""
    try:
        chat = Chat.query.get(chat_id)
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        db.session.delete(chat)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Chat {chat_id} deleted successfully'
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/report_analysis', methods=['POST'])
def report_analysis():
    """Handle health report analysis using RAG"""
    try:
        logger.info("Health report analysis request received")
        
        # Check if report analyzer is available
        if not hasattr(current_app, 'report_analyzer_available') or not current_app.report_analyzer_available:
            logger.warning("Report analyzer is not available due to dependency issues")
            return jsonify({
                'error': 'The health report analysis feature is currently unavailable due to dependency issues. Please make sure all required packages are installed correctly.',
                'report_analysis': False
            }), 503
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if it's a PDF
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Get query, use default if not provided
        query = request.form.get('query', 'Analyze this health report and provide insights.')
        chat_id = request.form.get('chat_id')
        
        # Get report analyzer
        from app.utils.report_analyzer import create_report_analyzer
        report_analyzer = create_report_analyzer()
        
        # Process report and generate response
        response_data = report_analyzer.analyze_report(file, query, chat_id)
        
        # Save to database if chat_id is provided
        if chat_id:
            save_report_analysis_to_db(chat_id, query, response_data, file.filename)
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in report analysis: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate responses"""
    logger.info("Received chat request")
    
    data = request.get_json()
    message = data.get('message', '')
    chat_id = data.get('chat_id')
    
    # Create a new chat ID if none provided
    if not chat_id:
        chat_id = str(uuid.uuid4())
    
    # Get current time for timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if this is a health classification request
    health_classification = False
    
    if "#health" in message.lower():
        health_classification = True
        # Remove the #health tag from the message for processing
        clean_message = message.lower().replace("#health", "").strip()
        
        # Check if health model is available
        if current_app.health_model is not None:
            # Get health classification
            classification = current_app.health_model.classify_message(clean_message)
            
            response_data = {
                'response': f"Health Classification: {classification['label']} (Confidence: {classification['score']:.2f})\n\n{classification['explanation']}",
                'chat_id': chat_id,
                'timestamp': timestamp,
                'health_classification': True,
                'classification': classification
            }
        else:
            logger.warning("Health model not available but #health tag was used")
            response_data = {
                'response': "I'm sorry, but the health classification feature is currently unavailable due to missing dependencies.\n\n**Why this happens**: The application needs the `transformers` library to run health classification models.\n\n**What you can try instead**:\n1. Use the 'Health Report Analysis' feature to upload medical reports\n2. Ask general health questions without the #health tag\n3. Contact the administrator to install the required dependencies",
                'chat_id': chat_id,
                'timestamp': timestamp,
                'health_classification': False
            }
    else:
        # Default response for regular chat
        response_data = {
            'response': f"Thank you for your message: '{message}'\n\nThis is a simple response as the full chat model is not currently integrated. Here are some things you can try:\n\n1. **Use the Health Report Analysis** feature to upload and analyze medical reports\n2. **Try a health query** by prefixing your message with #health (note: requires additional dependencies)\n3. **Ask specific questions** about the capabilities of this system",
            'chat_id': chat_id,
            'timestamp': timestamp
        }
    
    logger.info(f"Generated response for chat {chat_id}")
    return jsonify(response_data)


@chat_bp.route('/api/upload-report', methods=['POST'])
def upload_report():
    """Handle health report upload and analysis"""
    logger.info("Received health report upload")
    
    # Check if report analyzer is available
    if not hasattr(current_app, 'report_analyzer') or current_app.report_analyzer is None:
        logger.warning("Report analyzer not available but upload was attempted")
        return jsonify({
            'response': "I'm sorry, but the health report analysis feature is currently unavailable due to missing dependencies. Please try again later.",
            'error': True
        }), 503
    
    # Get form data
    chat_id = request.form.get('chat_id')
    query = request.form.get('query', 'Can you summarize this health report?')
    
    # Check for file
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'response': 'No file uploaded', 'error': True}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'response': 'No file selected', 'error': True}), 400
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        logger.error("Invalid file type")
        return jsonify({'response': 'Only PDF files are supported', 'error': True}), 400
    
    try:
        # Process the file with report analyzer
        result = current_app.report_analyzer.analyze_report(file, query, chat_id)
        logger.info("Report analysis completed successfully")
        
        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['timestamp'] = timestamp
        result['chat_id'] = chat_id
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing report: {str(e)}")
        return jsonify({
            'response': f"Error analyzing report: {str(e)}",
            'error': True
        }), 500


# Helper functions
def save_messages_to_db(chat_id, user_message, response_data):
    """Save health messages to database"""
    try:
        # Check if chat exists, create if not
        chat = Chat.query.get(chat_id)
        if not chat:
            chat = Chat(
                id=chat_id,
                title=user_message[:30] + ('...' if len(user_message) > 30 else ''),
                mode='health',
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow()
            )
            db.session.add(chat)
        else:
            chat.updated_at = datetime.datetime.utcnow()
        
        # Add user message
        user_msg = Message(
            chat_id=chat_id,
            role='user',
            content=user_message,
            content_type='text',
            timestamp=datetime.datetime.utcnow()
        )
        db.session.add(user_msg)
        
        # Add bot response
        bot_msg = Message(
            chat_id=chat_id,
            role='bot',
            content=response_data.get('response', ''),
            content_type='text',
            timestamp=datetime.datetime.utcnow()
        )
        
        # If there's a diagnosis, add metadata
        if 'disease' in response_data and 'confidence' in response_data:
            bot_msg.content_type = 'diagnosis'
            bot_msg.meta_data = jsonify({
                'disease': response_data['disease'],
                'confidence': response_data['confidence']
            }).data.decode('utf-8')
        
        db.session.add(bot_msg)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving messages to DB: {str(e)}")
        raise


def save_to_db(chat_id, mode, user_message, bot_response):
    """Save general chat messages to database"""
    try:
        # Check if chat exists, create if not
        chat = Chat.query.get(chat_id)
        if not chat:
            chat = Chat(
                id=chat_id,
                title=user_message[:30] + ('...' if len(user_message) > 30 else ''),
                mode=mode,
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow()
            )
            db.session.add(chat)
        else:
            chat.updated_at = datetime.datetime.utcnow()
        
        # Add user message
        user_msg = Message(
            chat_id=chat_id,
            role='user',
            content=user_message,
            content_type='text',
            timestamp=datetime.datetime.utcnow()
        )
        db.session.add(user_msg)
        
        # Add bot response
        bot_msg = Message(
            chat_id=chat_id,
            role='bot',
            content=bot_response,
            content_type='text',
            timestamp=datetime.datetime.utcnow()
        )
        db.session.add(bot_msg)
        
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving to DB: {str(e)}")
        raise


def get_chat_history(chat_id, limit=10):
    """Get chat history for a specific chat"""
    try:
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp).limit(limit).all()
        return [message.to_dict() for message in messages]
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return []


def save_report_analysis_to_db(chat_id, user_message, response_data, filename):
    """Save report analysis to database"""
    try:
        # Check if chat exists, create if not
        chat = Chat.query.get(chat_id)
        if not chat:
            chat = Chat(
                id=chat_id,
                title=f"Report Analysis: {filename}",
                mode='report_analysis',
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow()
            )
            db.session.add(chat)
        else:
            chat.updated_at = datetime.datetime.utcnow()
            chat.mode = 'report_analysis'
        
        # Add user message
        user_msg = Message(
            chat_id=chat_id,
            role='user',
            content=user_message,
            content_type='report_query',
            meta_data=jsonify({'filename': filename}).data.decode('utf-8'),
            timestamp=datetime.datetime.utcnow()
        )
        db.session.add(user_msg)
        
        # Add bot response
        bot_msg = Message(
            chat_id=chat_id,
            role='bot',
            content=response_data.get('response', ''),
            content_type='report_analysis',
            timestamp=datetime.datetime.utcnow()
        )
        
        # Add source documents as metadata
        if 'sources' in response_data:
            bot_msg.meta_data = jsonify({
                'sources': response_data['sources']
            }).data.decode('utf-8')
        
        db.session.add(bot_msg)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving report analysis to DB: {str(e)}")
        raise 