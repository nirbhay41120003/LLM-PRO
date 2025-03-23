from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Chat(db.Model):
    """Model for storing chat sessions"""
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(100), nullable=False, default="New Chat")
    mode = db.Column(db.String(20), nullable=False, default="health")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with messages
    messages = db.relationship('Message', backref='chat', lazy=True, cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert chat to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'mode': self.mode,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages': [message.to_dict() for message in self.messages]
        }
    
    @staticmethod
    def from_dict(chat_dict):
        """Create a chat from a dictionary"""
        chat = Chat(
            id=chat_dict.get('id'),
            title=chat_dict.get('title', 'New Chat'),
            mode=chat_dict.get('mode', 'health')
        )
        
        if 'created_at' in chat_dict:
            try:
                chat.created_at = datetime.fromisoformat(chat_dict['created_at'])
            except (ValueError, TypeError):
                chat.created_at = datetime.utcnow()
        
        return chat


class Message(db.Model):
    """Model for storing chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.String(50), db.ForeignKey('chat.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'bot'
    content = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(20), default="text")  # 'text', 'diagnosis', etc.
    meta_data = db.Column(db.Text, nullable=True)  # JSON string for additional data
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert message to dictionary for JSON serialization"""
        data = {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'content_type': self.content_type,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.meta_data:
            try:
                data['metadata'] = json.loads(self.meta_data)
            except json.JSONDecodeError:
                data['metadata'] = {}
        
        return data
    
    @staticmethod
    def from_dict(message_dict, chat_id):
        """Create a message from a dictionary"""
        message = Message(
            chat_id=chat_id,
            role=message_dict.get('role', 'user'),
            content=message_dict.get('content', ''),
            content_type=message_dict.get('content_type', 'text')
        )
        
        metadata = message_dict.get('metadata')
        if metadata:
            if isinstance(metadata, dict):
                message.meta_data = json.dumps(metadata)
            elif isinstance(metadata, str):
                message.meta_data = metadata
        
        if 'timestamp' in message_dict:
            try:
                message.timestamp = datetime.fromisoformat(message_dict['timestamp'])
            except (ValueError, TypeError):
                message.timestamp = datetime.utcnow()
        
        return message


def init_db(app):
    """Initialize the database"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all() 