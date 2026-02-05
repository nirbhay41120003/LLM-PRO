import os
import logging
import tempfile
import re
import json
import time
from PyPDF2 import PdfReader
import torch
from flask import current_app, jsonify
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthReportAnalyzer:
    """Class for analyzing health reports using a simpler approach with direct API calls"""
    
    def __init__(self):
        """Initialize the health report analyzer"""
        logger.info("Initializing Health Report Analyzer")
        
        # Check if HuggingFace API key is available
        self.hf_api_key = current_app.config.get('HUGGINGFACE_API_KEY') or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.hf_api_key:
            logger.warning("HuggingFace API key not found. Some features may not work properly.")
        else:
            logger.info("HuggingFace API key loaded successfully")
        
        # Initialize document store
        self.document_store = {}
        
        logger.info("Health Report Analyzer initialized successfully")

    def process_pdf(self, pdf_file):
        """
        Process a PDF file and extract text
        
        Args:
            pdf_file: A file-like object containing the PDF
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            logger.info("Processing PDF file")
            
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                pdf_file.save(temp_file)
                temp_file_path = temp_file.name
            
            # Extract text from PDF
            reader = PdfReader(temp_file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean up
            os.unlink(temp_file_path)
            
            logger.info(f"PDF processing complete. Extracted {len(text)} characters.")
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def split_text(self, text, chunk_size=1000, overlap=200):
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len and end - start == chunk_size:
                # Find the last period or newline to make better chunks
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                split_point = max(last_period, last_newline)
                if split_point > start + chunk_size // 2:  # Only use if it's far enough
                    end = split_point + 1
            
            chunks.append(text[start:end])
            start = end - overlap if end < text_len else text_len
        
        return chunks
    
    def find_relevant_chunks(self, chunks, query):
        """Find chunks most relevant to the query using simple keyword matching"""
        query_terms = query.lower().split()
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0
            for term in query_terms:
                score += chunk_lower.count(term)
            chunk_scores.append((i, score))
        
        # Sort by score and get top 3
        relevant_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:3]
        return [chunks[i] for i, _ in relevant_chunks if _ > 0]
    
    def query_huggingface_model(self, query, context, max_retries=3, backoff_factor=1):
        """Query the HuggingFace model directly using API with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                if not self.hf_api_key:
                    return "Error: HuggingFace API key not found. Please set the HUGGINGFACE_API_KEY environment variable."
                
                # Combine context and query
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                
                # Send API request to HuggingFace
                headers = {
                    "Authorization": f"Bearer {self.hf_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 500,
                        "temperature": 0.7
                    }
                }
                
                # Use Flan-T5-Large model
                api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                response = requests.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "").strip()
                    return "No valid response received from the model."
                elif response.status_code == 503:
                    # Service unavailable - likely model still loading or server overloaded
                    logger.warning(f"HuggingFace API returned 503 (Service Unavailable). Retry {retries+1}/{max_retries}")
                    error_message = response.text
                    
                    # If this is the last retry, provide a fallback response
                    if retries == max_retries - 1:
                        # Try to extract any estimated time if available in the response
                        try:
                            error_json = response.json()
                            estimated_time = error_json.get('estimated_time', 'unknown')
                            return f"The HuggingFace model is currently loading and unavailable (estimated time: {estimated_time} seconds). In the meantime, here's a summary of the relevant text:\n\n{context[:500]}...\n\nPlease try again in a few moments."
                        except:
                            # Fallback to a generic message
                            return "The HuggingFace API is currently unavailable (Error 503). This usually means the model is still loading or the servers are busy. Here are the relevant sections I found in your document: \n\n" + context[:500] + "...\n\nPlease try again in a few moments."
                    
                    # Exponential backoff
                    wait_time = backoff_factor * (2 ** retries)
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"Error from HuggingFace API: {response.status_code} - {response.text}")
                    return f"Error from HuggingFace API: {response.status_code}. The service might be experiencing issues. Please try again later."
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error querying HuggingFace model: {str(e)}")
                # If this is the last retry, return an error message
                if retries == max_retries - 1:
                    return f"Network error communicating with HuggingFace API: {str(e)}. Please check your internet connection and try again."
                
                # Exponential backoff
                wait_time = backoff_factor * (2 ** retries)
                time.sleep(wait_time)
                retries += 1
            except Exception as e:
                logger.error(f"Error querying HuggingFace model: {str(e)}")
                return f"Error: {str(e)}"
        
        # If we've exhausted all retries
        return "Failed to get a response from HuggingFace API after multiple attempts. The service might be temporarily unavailable."
    
    def fallback_analysis(self, text, query):
        """Provide a simple text-based analysis when the API is unavailable"""
        chunks = self.split_text(text)
        relevant_chunks = self.find_relevant_chunks(chunks, query)
        
        # Join relevant chunks for context
        context = "\n\n".join(relevant_chunks) if relevant_chunks else text[:1000]
        
        # Create a simple summary
        keyword_counts = {}
        for word in re.findall(r'\b\w+\b', context.lower()):
            if len(word) > 3:  # Only count words with more than 3 letters
                keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords_str = ", ".join([word for word, _ in top_keywords])
        
        # Generate simple response
        response = f"Here's a simple analysis of your health report (HuggingFace API is currently unavailable):\n\n"
        response += f"**Key terms found**: {keywords_str}\n\n"
        response += f"**Report length**: {len(text)} characters, approximately {len(text.split())} words\n\n"
        response += f"I found these sections that might be relevant to your query:\n\n"
        
        return response, relevant_chunks
    
    def analyze_report(self, pdf_file, query, chat_id=None):
        """
        Analyze a health report using a direct API approach
        
        Args:
            pdf_file: A file-like object containing the PDF
            query: User's question about the report
            chat_id: Optional chat ID for caching document store
            
        Returns:
            dict: Analysis results
        """
        try:
            # Get or extract text from PDF
            if chat_id and chat_id in self.document_store:
                logger.info(f"Using cached document for chat {chat_id}")
                text = self.document_store[chat_id]
            else:
                # Process PDF
                text = self.process_pdf(pdf_file)
                # Cache for future queries
                if chat_id:
                    self.document_store[chat_id] = text
            
            # Split text into chunks
            chunks = self.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(chunks, query)
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found, using first chunk")
                relevant_chunks = [chunks[0]] if chunks else []
            
            # Use all relevant chunks as context
            context = "\n".join(relevant_chunks)
            
            try:
                # Try to generate answer using HuggingFace model
                answer = self.query_huggingface_model(query, context)
                
                # Check if we got an error message about API unavailability
                if "Error from HuggingFace API: 503" in answer or "HuggingFace API is currently unavailable" in answer:
                    logger.warning("HuggingFace API unavailable, falling back to simple analysis")
                    fallback_response, fallback_chunks = self.fallback_analysis(text, query)
                    
                    # Format response with fallback
                    response = {
                        "response": fallback_response,
                        "sources": fallback_chunks or relevant_chunks,
                        "report_analysis": True,
                        "api_status": "unavailable"
                    }
                else:
                    # Format response with API results
                    response = {
                        "response": answer,
                        "sources": relevant_chunks,
                        "report_analysis": True,
                        "api_status": "ok"
                    }
            except Exception as e:
                # If API call fails, use fallback analysis
                logger.error(f"API call failed, using fallback analysis: {str(e)}")
                fallback_response, fallback_chunks = self.fallback_analysis(text, query)
                
                response = {
                    "response": fallback_response,
                    "sources": fallback_chunks or relevant_chunks,
                    "report_analysis": True,
                    "api_status": "error",
                    "error_details": str(e)
                }
            
            logger.info("Report analysis completed")
            return response
        except Exception as e:
            logger.error(f"Error analyzing report: {str(e)}")
            raise

def create_report_analyzer():
    """Create and return a health report analyzer instance"""
    return HealthReportAnalyzer() 