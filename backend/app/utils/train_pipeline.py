"""
Complete Training Pipeline for Health Assistant
Prepares data, trains BioBERT classifier, and sets up RAG system
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data():
    """Step 1: Prepare and combine all datasets"""
    logger.info("=" * 50)
    logger.info("STEP 1: Preparing Training Data")
    logger.info("=" * 50)
    
    from app.utils.data_loader import MedicalDataLoader
    
    loader = MedicalDataLoader()
    stats = loader.save_processed_data()
    
    logger.info(f"Data preparation complete!")
    logger.info(f"  - Total samples: {stats['samples']}")
    logger.info(f"  - Disease classes: {stats['diseases']}")
    
    return stats


def train_classifier(model_type: str = "biobert", epochs: int = 3, batch_size: int = 16):
    """Step 2: Train BioBERT classifier"""
    logger.info("=" * 50)
    logger.info(f"STEP 2: Training {model_type.upper()} Classifier")
    logger.info("=" * 50)
    
    from app.utils.biobert_classifier import train_biobert_classifier
    
    data_path = os.path.join(project_root, 'data', 'processed', 'combined_symptoms.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        logger.info("Please run data preparation first (Step 1)")
        return None
    
    classifier = train_biobert_classifier(
        data_path=data_path,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size
    )
    
    logger.info("Classifier training complete!")
    return classifier


def setup_rag():
    """Step 3: Setup RAG system"""
    logger.info("=" * 50)
    logger.info("STEP 3: Setting up RAG System")
    logger.info("=" * 50)
    
    from app.utils.rag_system import MedicalKnowledgeBase
    
    kb = MedicalKnowledgeBase()
    
    logger.info(f"RAG system initialized with {len(kb.documents)} documents")
    
    return kb


def test_model():
    """Step 4: Test the trained model"""
    logger.info("=" * 50)
    logger.info("STEP 4: Testing Model")
    logger.info("=" * 50)
    
    from app.utils.health_model_v2 import IntegratedHealthModel
    
    model = IntegratedHealthModel(model_type="biobert")
    
    if not model.initialize():
        logger.error("Failed to initialize model!")
        return False
    
    # Test cases
    test_queries = [
        "I have fever, headache, and body pain for the last two days",
        "I've been experiencing skin rash with itching and scaling",
        "I have burning sensation while urinating and frequent urination",
        "My joints are painful and swollen, especially in the morning",
        "I have a persistent cough with phlegm and difficulty breathing"
    ]
    
    logger.info("Running test predictions...")
    for query in test_queries:
        result = model.get_health_response(query, include_disclaimer=False)
        disease = result.get('disease', 'Unknown')
        confidence = result.get('confidence', 0)
        logger.info(f"\nQuery: '{query[:50]}...'")
        logger.info(f"Prediction: {disease} ({confidence:.1f}%)")
    
    return True


def full_pipeline(model_type: str = "biobert", epochs: int = 3, batch_size: int = 16, skip_training: bool = False):
    """Run the complete training pipeline"""
    logger.info("=" * 60)
    logger.info("HEALTH ASSISTANT - FULL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Prepare data
    stats = prepare_data()
    
    if not skip_training:
        # Step 2: Train classifier
        classifier = train_classifier(model_type, epochs, batch_size)
        
        if classifier is None:
            logger.error("Training failed!")
            return False
    else:
        logger.info("Skipping training (--skip-training flag set)")
    
    # Step 3: Setup RAG
    kb = setup_rag()
    
    # Step 4: Test model
    success = test_model()
    
    if success:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info("The model is ready to use. Start the server with:")
        logger.info("  python app.py")
    else:
        logger.warning("Pipeline completed with warnings. Check logs above.")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Health Assistant Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings
  python -m app.utils.train_pipeline
  
  # Train with PubMedBERT for more epochs
  python -m app.utils.train_pipeline --model pubmedbert --epochs 5
  
  # Only prepare data and test (skip training)
  python -m app.utils.train_pipeline --skip-training
  
  # Just prepare data
  python -m app.utils.train_pipeline --step prepare
  
  # Just train
  python -m app.utils.train_pipeline --step train
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="biobert",
        choices=["biobert", "pubmedbert", "bio_clinical_bert", "scibert"],
        help="Model type to use (default: biobert)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["prepare", "train", "rag", "test", "all"],
        default="all",
        help="Which step to run (default: all)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training step (use existing model)"
    )
    
    args = parser.parse_args()
    
    if args.step == "all":
        full_pipeline(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_training=args.skip_training
        )
    elif args.step == "prepare":
        prepare_data()
    elif args.step == "train":
        train_classifier(args.model, args.epochs, args.batch_size)
    elif args.step == "rag":
        setup_rag()
    elif args.step == "test":
        test_model()


if __name__ == "__main__":
    main()
