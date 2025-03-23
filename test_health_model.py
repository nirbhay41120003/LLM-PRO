#!/usr/bin/env python
"""
Test script for the health model
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.utils.health_model import HealthModelHandler

def main():
    """Main test function"""
    print("Initializing health model...")
    # Use the bio_clinical_bert model directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'app', 'models', 'bio_clinical_bert')
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found at: {model_dir}")
        print("Please make sure you've trained the model first.")
        return

    model = HealthModelHandler(model_dir=model_dir)
    
    # Test with a greeting
    print("\n--- Testing with 'hi' ---")
    result = model.classify_message("hi")
    print(f"Label: {result.get('label')}")
    print(f"Score: {result.get('score')}")
    print(f"Explanation: {result.get('explanation')}")
    
    # Test with urine infection
    print("\n--- Testing with 'I have a urine infection' ---")
    result = model.predict("I have a urine infection")
    print(f"Disease: {result.get('disease')}")
    print(f"Confidence: {result.get('confidence')}%")
    print(f"Manual prediction: {result.get('manual_prediction', False)}")
    
    # Test the classify message with urine infection
    print("\n--- Testing classify_message with 'urine infection' ---")
    result = model.classify_message("I think I have a urine infection")
    print(f"Label: {result.get('label')}")
    print(f"Score: {result.get('score')}")
    print(f"Explanation length: {len(result.get('explanation', ''))}")
    
    # Test with a non-medical query
    print("\n--- Testing with 'what can you do' ---")
    result = model.classify_message("what can you do")
    print(f"Label: {result.get('label')}")
    print(f"Score: {result.get('score')}")
    print(f"Explanation: {result.get('explanation')}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main() 