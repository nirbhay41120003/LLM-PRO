#!/usr/bin/env python3
"""
Script to train health models and prepare disease information

This script helps users to:
1. Train the health classification model with improved models
2. Generate disease information templates
3. Populate the disease database with comprehensive information

Usage:
    python scripts/train_and_prepare_model.py --help
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

def train_model(model_type, epochs=5, batch_size=16):
    """Train the health classification model with specified parameters"""
    try:
        from app.models.train_model import train_model
        
        logger.info(f"Starting training of {model_type} model with {epochs} epochs and batch size {batch_size}")
        model, tokenizer = train_model(model_type=model_type)
        
        if model is not None and tokenizer is not None:
            logger.info("Model training completed successfully")
            return True
        else:
            logger.error("Model training failed")
            return False
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def prepare_disease_info(overwrite=False):
    """
    Prepare disease information database
    
    Args:
        overwrite: Whether to overwrite existing entries
        
    Returns:
        bool: Success status
    """
    try:
        # Get path to disease info file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        disease_info_path = os.path.join(base_dir, 'data', 'processed', 'disease_info.json')
        disease_classes_path = os.path.join(base_dir, 'data', 'processed', 'disease_classes.txt')
        
        # Ensure the directories exist
        os.makedirs(os.path.dirname(disease_info_path), exist_ok=True)
        
        if not os.path.exists(disease_classes_path):
            logger.error(f"Disease classes file not found at {disease_classes_path}")
            logger.error("You need to train the model first or manually create a disease_classes.txt file")
            return False
        
        # Load disease classes
        with open(disease_classes_path, 'r') as f:
            disease_classes = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Found {len(disease_classes)} disease classes")
        
        # Load existing disease info if available
        disease_info = {}
        if os.path.exists(disease_info_path) and not overwrite:
            try:
                with open(disease_info_path, 'r') as f:
                    disease_info = json.load(f)
                logger.info(f"Loaded existing disease info for {len(disease_info)} diseases")
            except Exception as e:
                logger.warning(f"Could not load existing disease info: {str(e)}")
        
        # Template for disease information
        template = {
            "description": "",
            "symptoms": [],
            "causes": [],
            "risk_factors": [],
            "treatment": [],
            "when_to_see_doctor": ""
        }
        
        # Update disease info
        updated = False
        for disease in disease_classes:
            if disease and (disease not in disease_info or overwrite):
                disease_info[disease] = template.copy()
                updated = True
                logger.info(f"Added template for {disease}")
        
        # Save the updated disease info
        if updated or overwrite:
            with open(disease_info_path, 'w') as f:
                json.dump(disease_info, f, indent=2)
            logger.info(f"Disease information saved to {disease_info_path}")
        else:
            logger.info("No updates needed for disease information")
        
        return True
    
    except Exception as e:
        logger.error(f"Error preparing disease info: {str(e)}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train health models and prepare disease information")
    
    # Model training options
    parser.add_argument('--train', action='store_true', help='Train the health classification model')
    parser.add_argument('--model-type', type=str, default='bio_clinical_bert',
                      choices=['bio_clinical_bert', 'pubmed_bert', 'biogpt', 'biobert'],
                      help='Type of model to use for training')
    
    # Disease info options
    parser.add_argument('--prepare-info', action='store_true', help='Prepare disease information database')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing disease information')
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not (args.train or args.prepare_info):
        parser.print_help()
        return
    
    # Train model if requested
    if args.train:
        success = train_model(args.model_type)
        if not success:
            logger.error("Model training failed")
            return
    
    # Prepare disease info if requested
    if args.prepare_info:
        success = prepare_disease_info(args.overwrite)
        if not success:
            logger.error("Disease information preparation failed")
            return
    
    logger.info("All operations completed successfully")

if __name__ == "__main__":
    main() 