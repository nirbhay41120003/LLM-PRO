#!/usr/bin/env python
"""
Additional test script for the health model focusing on UTI detection
"""

from app.utils.health_model import create_health_model_handler

def main():
    """Main test function"""
    print("Initializing health model...")
    model = create_health_model_handler(model_dir='app/models/bio_clinical_bert')
    
    # Test UTI detection
    print("\n--- Testing UTI detection phrases ---")
    uti_phrases = [
        "I have a urine infection",
        "I think I have a urinary tract infection",
        "burning when I pee",
        "painful urination",
        "bladder infection symptoms"
    ]
    
    for phrase in uti_phrases:
        print(f"\nTesting: '{phrase}'")
        result = model.predict(phrase)
        print(f"Disease: {result.get('disease')}")
        print(f"Confidence: {result.get('confidence')}%")
        print(f"Manual prediction: {result.get('manual_prediction', False)}")
    
    # Test non-health queries
    print("\n--- Testing non-health queries ---")
    non_health_phrases = [
        "hello",
        "how are you doing",
        "what can you help me with",
        "tell me a joke"
    ]
    
    for phrase in non_health_phrases:
        print(f"\nTesting: '{phrase}'")
        result = model.classify_message(phrase)
        print(f"Label: {result.get('label')}")
        print(f"Score: {result.get('score')}")
    
    # Test vague symptoms
    print("\n--- Testing vague symptoms ---")
    vague_symptoms = [
        "I feel a bit off",
        "not feeling well",
        "something's wrong"
    ]
    
    for symptom in vague_symptoms:
        print(f"\nTesting: '{symptom}'")
        result = model.predict(symptom)
        if 'error' in result:
            print(f"Error: {result.get('error')}")
            print(f"Confidence: {result.get('confidence', 0)}%")
        else:
            print(f"Disease: {result.get('disease')}")
            print(f"Confidence: {result.get('confidence')}%")

    print("\nTesting complete!")

if __name__ == "__main__":
    main() 