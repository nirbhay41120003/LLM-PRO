"""
RAG (Retrieval-Augmented Generation) System for Medical Knowledge
Uses FAISS vector store with sentence-transformers for semantic search
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. RAG functionality will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Using fallback search.")


class MedicalKnowledgeBase:
    """Medical knowledge base with semantic search capabilities"""
    
    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'medical_knowledge'
        )
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        self.documents: List[Dict] = []
        self.embeddings = None
        self.index = None
        self.encoder = None
        
        self._initialize_encoder()
        self._load_knowledge_base()
    
    def _initialize_encoder(self):
        """Initialize the sentence encoder for embeddings"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a medical domain-specific model if available, else use general model
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.encoder = SentenceTransformer(model_name)
                logger.info(f"Initialized sentence encoder: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize encoder: {e}")
                self.encoder = None
        else:
            self.encoder = None
    
    def _load_knowledge_base(self):
        """Load the knowledge base from files"""
        # Load disease info
        disease_info_path = os.path.join(
            os.path.dirname(self.knowledge_dir), 'processed', 'disease_info.json'
        )
        
        if os.path.exists(disease_info_path):
            with open(disease_info_path, 'r') as f:
                disease_info = json.load(f)
            
            for disease, info in disease_info.items():
                # Create document for each disease
                doc_text = self._create_disease_document(disease, info)
                self.documents.append({
                    'id': f"disease_{disease}",
                    'type': 'disease_info',
                    'disease': disease,
                    'content': doc_text,
                    'metadata': info
                })
        
        # Load any additional medical documents
        docs_dir = os.path.join(self.knowledge_dir, 'documents')
        if os.path.exists(docs_dir):
            for filename in os.listdir(docs_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(docs_dir, filename), 'r') as f:
                        doc = json.load(f)
                        self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents into knowledge base")
        
        # Build index if we have documents
        if self.documents and self.encoder:
            self._build_index()
    
    def _create_disease_document(self, disease: str, info: Dict) -> str:
        """Create a searchable document from disease info"""
        parts = [f"Disease: {disease}"]
        
        if info.get('description'):
            parts.append(f"Description: {info['description']}")
        
        if info.get('symptoms'):
            symptoms_text = ", ".join(info['symptoms'])
            parts.append(f"Symptoms: {symptoms_text}")
        
        if info.get('causes'):
            causes_text = ", ".join(info['causes'])
            parts.append(f"Causes: {causes_text}")
        
        if info.get('risk_factors'):
            risk_text = ", ".join(info['risk_factors'])
            parts.append(f"Risk factors: {risk_text}")
        
        if info.get('treatment'):
            if isinstance(info['treatment'], list):
                treatment_text = "; ".join(info['treatment'])
            else:
                treatment_text = info['treatment']
            parts.append(f"Treatment: {treatment_text}")
        
        if info.get('when_to_see_doctor'):
            parts.append(f"When to see a doctor: {info['when_to_see_doctor']}")
        
        return "\n".join(parts)
    
    def _build_index(self):
        """Build FAISS index for semantic search"""
        if not FAISS_AVAILABLE or not self.encoder:
            logger.warning("Cannot build index: FAISS or encoder not available")
            return
        
        try:
            # Get embeddings for all documents
            texts = [doc['content'] for doc in self.documents]
            self.embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            logger.info(f"Built FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            self.index = None
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the knowledge base for relevant documents"""
        if not self.documents:
            return []
        
        # If we have FAISS index, use semantic search
        if self.index is not None and self.encoder is not None:
            return self._semantic_search(query, top_k)
        
        # Fallback to keyword search
        return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic search using FAISS"""
        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(score)
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search"""
        query_terms = query.lower().split()
        
        scored_docs = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                doc_copy = doc.copy()
                doc_copy['score'] = score
                scored_docs.append(doc_copy)
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]
    
    def get_disease_info(self, disease: str) -> Optional[Dict]:
        """Get information about a specific disease"""
        for doc in self.documents:
            if doc.get('type') == 'disease_info' and doc.get('disease', '').lower() == disease.lower():
                return doc.get('metadata', {})
        return None
    
    def add_document(self, doc_id: str, content: str, doc_type: str = 'general', metadata: Dict = None):
        """Add a new document to the knowledge base"""
        doc = {
            'id': doc_id,
            'type': doc_type,
            'content': content,
            'metadata': metadata or {}
        }
        self.documents.append(doc)
        
        # Rebuild index
        if self.encoder:
            self._build_index()


class RAGSystem:
    """Retrieval-Augmented Generation system for medical queries"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase = None):
        self.knowledge_base = knowledge_base or MedicalKnowledgeBase()
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        results = self.knowledge_base.search(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}]\n{result['content']}")
        
        return "\n\n".join(context_parts)
    
    def augmented_query(self, query: str, prediction: Dict = None) -> Dict:
        """
        Create an augmented response using retrieved context
        
        Args:
            query: User's original query
            prediction: Model prediction (disease, confidence, etc.)
            
        Returns:
            Dict with augmented response
        """
        # Retrieve relevant context
        context = self.retrieve_context(query, top_k=3)
        
        response_data = {
            'query': query,
            'context_retrieved': bool(context),
            'sources': []
        }
        
        # If we have a prediction, get specific disease info
        if prediction and prediction.get('disease'):
            disease = prediction['disease']
            disease_info = self.knowledge_base.get_disease_info(disease)
            
            if disease_info:
                response_data['disease_info'] = disease_info
                response_data['sources'].append({
                    'type': 'disease_database',
                    'disease': disease
                })
        
        # Add retrieved context
        if context:
            response_data['retrieved_context'] = context
        
        return response_data
    
    def generate_comprehensive_response(self, query: str, prediction: Dict) -> str:
        """Generate a comprehensive response using RAG"""
        augmented = self.augmented_query(query, prediction)
        
        response_parts = []
        
        # Main prediction
        if prediction.get('disease'):
            disease = prediction['disease']
            confidence = prediction.get('confidence', 0)
            
            response_parts.append(
                f"Based on your symptoms, the most likely condition is **{disease}** "
                f"(confidence: {confidence:.1f}%)."
            )
        
        # Add disease information from RAG
        disease_info = augmented.get('disease_info', {})
        
        if disease_info.get('description'):
            response_parts.append(f"\n**About {prediction.get('disease', 'this condition')}:**")
            response_parts.append(disease_info['description'])
        
        if disease_info.get('symptoms'):
            symptoms_list = disease_info['symptoms']
            if symptoms_list:
                response_parts.append("\n**Common symptoms:**")
                for symptom in symptoms_list[:5]:
                    response_parts.append(f"• {symptom}")
        
        if disease_info.get('treatment'):
            treatments = disease_info['treatment']
            if treatments:
                response_parts.append("\n**Recommended actions:**")
                if isinstance(treatments, list):
                    for treatment in treatments[:4]:
                        response_parts.append(f"• {treatment}")
                else:
                    response_parts.append(f"• {treatments}")
        
        if disease_info.get('when_to_see_doctor'):
            response_parts.append(f"\n**When to see a doctor:** {disease_info['when_to_see_doctor']}")
        
        # Add alternatives if present
        alternatives = prediction.get('alternatives', [])
        if alternatives and prediction.get('confidence', 100) < 80:
            response_parts.append("\n**Other possibilities to consider:**")
            for alt in alternatives[:2]:
                response_parts.append(f"• {alt['disease']} ({alt['confidence']:.1f}%)")
        
        # Disclaimer
        response_parts.append(
            "\n\n⚠️ **Disclaimer:** This is an AI-generated assessment and should not replace "
            "professional medical advice. Please consult a healthcare provider for proper diagnosis."
        )
        
        return "\n".join(response_parts)


def create_rag_system() -> RAGSystem:
    """Factory function to create RAG system"""
    return RAGSystem()


if __name__ == "__main__":
    # Test the RAG system
    rag = create_rag_system()
    
    test_query = "I have fever, headache, and body pain"
    context = rag.retrieve_context(test_query)
    print("Retrieved context:")
    print(context[:500] if context else "No context retrieved")
