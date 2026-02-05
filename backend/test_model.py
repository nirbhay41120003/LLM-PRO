#!/usr/bin/env python
"""
Test script for the BioBERT + RAG Health Model
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_classifier():
    """Test the BioBERT classifier"""
    print("\n" + "="*60)
    print("TESTING BIOBERT CLASSIFIER")
    print("="*60)
    
    from app.utils.biobert_classifier import BioBERTClassifier
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'app', 'models', 'biobert_finetuned')
    
    if not os.path.exists(model_dir):
        print(f"❌ Model not found at: {model_dir}")
        print("   Run training first: python -m app.utils.train_pipeline")
        return False
    
    classifier = BioBERTClassifier(model_dir=model_dir)
    
    test_symptoms = [
        "I have fever, headache, and body pain",
        "burning sensation while urinating",
        "skin rash with itching",
        "persistent cough with phlegm",
        "joint pain and swelling"
    ]
    
    print("\nRunning predictions:")
    for symptom in test_symptoms:
        result = classifier.predict(symptom, top_k=3)
        print(f"\n📋 Query: '{symptom}'")
        if result and len(result) > 0:
            print(f"   Top prediction: {result[0]['disease']} ({result[0]['confidence']:.1f}%)")
            if len(result) > 1:
                print(f"   Alternatives: {', '.join([f'{p['disease']} ({p['confidence']:.1f}%)' for p in result[1:3]])}")
        else:
            print("   No prediction available")
    
    print("\n✅ Classifier test complete!")
    return True


def test_rag_system():
    """Test the RAG system"""
    print("\n" + "="*60)
    print("TESTING RAG SYSTEM")
    print("="*60)
    
    from app.utils.rag_system import RAGSystem
    
    rag = RAGSystem()
    
    test_queries = [
        "What causes diabetes?",
        "How to prevent heart disease?",
        "Symptoms of flu"
    ]
    
    print("\nRunning knowledge retrieval:")
    for query in test_queries:
        context = rag.retrieve_context(query, top_k=2)
        print(f"\n📋 Query: '{query}'")
        if context:
            # Show first 100 chars of context
            print(f"   Context retrieved: {context[:100]}...")
        else:
            print("   No context retrieved")
    
    print("\n✅ RAG system test complete!")
    return True


def test_integrated_model():
    """Test the integrated health model"""
    print("\n" + "="*60)
    print("TESTING INTEGRATED HEALTH MODEL")
    print("="*60)
    
    from app.utils.health_model_v2 import IntegratedHealthModel
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'app', 'models', 'biobert_finetuned')
    
    model = IntegratedHealthModel(model_dir=model_dir)
    if not model.initialize():
        print("❌ Model initialization failed")
        return False
    
    test_symptoms = [
        "I have been having fever and headache for 3 days",
        "burning when urinating and frequent urination",
        "I have itchy skin rash on my arms"
    ]
    
    print("\nRunning full analysis:")
    for symptom in test_symptoms:
        print(f"\n📋 Query: '{symptom}'")
        result = model.predict(symptom)
        print(f"   Disease: {result.get('disease', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1f}%")
        if result.get('alternatives'):
            print(f"   Alternatives: {', '.join([a['disease'] for a in result['alternatives'][:2]])}")
    
    print("\n✅ Integrated model test complete!")
    return True


def test_api_endpoints():
    """Test the Flask API endpoints"""
    print("\n" + "="*60)
    print("TESTING API ENDPOINTS")
    print("="*60)
    
    import json
    from app import create_app
    
    app = create_app()
    
    with app.test_client() as client:
        # Test health endpoint
        print("\n📡 Testing /health endpoint...")
        response = client.post('/health', 
                              data=json.dumps({'message': 'fever and headache'}),
                              content_type='application/json')
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✅ Response: {data.get('disease', data.get('response', 'Unknown')[:50])}")
        else:
            data = response.get_json()
            print(f"   ⚠️ Status {response.status_code}: {data.get('error', 'Unknown error')}")
        
        # Test RAG search endpoint
        print("\n📡 Testing /rag-search endpoint...")
        response = client.post('/rag-search',
                              data=json.dumps({'query': 'diabetes symptoms'}),
                              content_type='application/json')
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✅ Found {len(data.get('results', []))} results")
        else:
            print(f"   ❌ Error: {response.status_code}")
        
        # Test general endpoint
        print("\n📡 Testing /general endpoint...")
        response = client.post('/general',
                              data=json.dumps({'message': 'What is a healthy diet?'}),
                              content_type='application/json')
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✅ Got response ({len(data.get('response', ''))} chars)")
        else:
            print(f"   ❌ Error: {response.status_code}")
    
    print("\n✅ API endpoint tests complete!")
    return True


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("HEALTH CHATBOT - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test classifier
    try:
        results['classifier'] = test_classifier()
    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        results['classifier'] = False
    
    # Test RAG system
    try:
        results['rag'] = test_rag_system()
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        results['rag'] = False
    
    # Test integrated model
    try:
        results['integrated'] = test_integrated_model()
    except Exception as e:
        print(f"❌ Integrated model test failed: {e}")
        results['integrated'] = False
    
    # Test API endpoints
    try:
        results['api'] = test_api_endpoints()
    except Exception as e:
        print(f"❌ API test failed: {e}")
        results['api'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
