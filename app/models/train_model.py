import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import random
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable CUDA and MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add path to the parent directory to import from app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_preprocessing import preprocess_data
from app.utils.tokenizer import tokenize_input

# Set device to CPU explicitly to avoid MPS issues
device = torch.device("cpu")
print(f"Using device: {device}")

class SymptomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def augment_symptoms(symptom_text: str, num_augmentations: int = 2) -> List[str]:
    """
    Apply data augmentation to symptom descriptions
    
    Args:
        symptom_text: Original symptom text
        num_augmentations: Number of augmented versions to create
        
    Returns:
        List of augmented symptom texts
    """
    augmented_texts = [symptom_text]  # Start with the original
    
    # If the text is too short, don't augment
    if len(symptom_text.split()) < 5:
        return augmented_texts
    
    # Common symptoms and medical terms
    medical_terms = [
        "pain", "ache", "fever", "cough", "headache", "nausea", 
        "fatigue", "tired", "sore", "swelling", "rash", "dizzy", 
        "vomiting", "diarrhea", "shortness of breath", "congestion"
    ]
    
    # Synonym replacements for common words
    synonyms = {
        "pain": ["discomfort", "ache", "soreness", "tenderness"],
        "severe": ["intense", "extreme", "unbearable", "terrible"],
        "mild": ["slight", "minor", "light", "gentle"],
        "persistent": ["continuous", "constant", "ongoing", "relentless"],
        "intermittent": ["occasional", "sporadic", "periodic", "on and off"],
        "swelling": ["inflammation", "edema", "puffiness", "bloating"],
        "redness": ["erythema", "inflammation", "flushing", "reddening"],
        "fatigue": ["exhaustion", "tiredness", "weariness", "lethargy"],
        "nausea": ["queasiness", "sickness", "upset stomach", "stomach discomfort"],
        "headache": ["head pain", "cephalgia", "migraine", "head discomfort"],
        "fever": ["elevated temperature", "pyrexia", "febrile", "high temperature"],
        "cough": ["hack", "throat clearing", "coughing", "respiratory irritation"]
    }
    
    # Techniques to try
    techniques = [
        "synonym_replacement",
        "random_deletion",
        "random_reordering",
        "add_descriptors"
    ]
    
    for _ in range(num_augmentations):
        # Choose a random technique
        technique = random.choice(techniques)
        text_copy = symptom_text
        
        if technique == "synonym_replacement":
            # Replace 1-3 words with synonyms
            words = text_copy.split()
            num_to_replace = min(random.randint(1, 3), len(words))
            
            for _ in range(num_to_replace):
                replace_idx = random.randint(0, len(words) - 1)
                word = words[replace_idx].lower()
                
                # Clean word of punctuation for matching
                clean_word = re.sub(r'[^\w\s]', '', word)
                
                if clean_word in synonyms:
                    # Replace with a synonym
                    replacement = random.choice(synonyms[clean_word])
                    # Preserve capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[replace_idx] = replacement
            
            augmented_text = " ".join(words)
            
        elif technique == "random_deletion":
            # Randomly delete 1-2 words that are not medical terms
            words = text_copy.split()
            if len(words) <= 5:  # Don't delete if text is already short
                augmented_text = text_copy
            else:
                num_to_delete = min(random.randint(1, 2), len(words) - 3)  # Keep at least 3 words
                
                for _ in range(num_to_delete):
                    delete_candidates = []
                    for i, word in enumerate(words):
                        clean_word = re.sub(r'[^\w\s]', '', word.lower())
                        if clean_word not in medical_terms:
                            delete_candidates.append(i)
                    
                    if delete_candidates:
                        delete_idx = random.choice(delete_candidates)
                        words.pop(delete_idx)
                
                augmented_text = " ".join(words)
                
        elif technique == "random_reordering":
            # Reorder clauses (sentences or comma-separated parts)
            parts = re.split(r'[.,;]', text_copy)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) <= 1:
                augmented_text = text_copy
            else:
                random.shuffle(parts)
                augmented_text = ". ".join(parts) + "."
                
        elif technique == "add_descriptors":
            # Add a descriptor to a symptom
            descriptors = ["mild", "severe", "persistent", "intermittent", "recurring", "occasional", "constant"]
            words = text_copy.split()
            
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if clean_word in medical_terms and random.random() < 0.7:  # 70% chance to add a descriptor
                    words.insert(i, random.choice(descriptors))
                    break
            
            augmented_text = " ".join(words)
        
        # Only add if it's different from the original and not already in the list
        if augmented_text != symptom_text and augmented_text not in augmented_texts:
            augmented_texts.append(augmented_text)
    
    return augmented_texts

def find_additional_datasets() -> List[str]:
    """
    Find additional datasets in the data/raw directory
    
    Returns:
        List of dataset file paths
    """
    dataset_files = []
    
    # Get path to data/raw directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'data', 'raw')
    
    # Look for CSV files
    try:
        for file in os.listdir(data_dir):
            if file.endswith('.csv') and 'symptom' in file.lower():
                dataset_files.append(os.path.join(data_dir, file))
        
        logger.info(f"Found {len(dataset_files)} additional dataset files")
    except Exception as e:
        logger.error(f"Error finding additional datasets: {str(e)}")
    
    return dataset_files

def train_model(model_type="bio_clinical_bert"):
    """
    Train a symptom-disease classification model
    
    Args:
        model_type: Type of model to use ('bio_clinical_bert', 'pubmed_bert', 'biogpt', 'biobert')
        
    Returns:
        Tuple of model and tokenizer
    """
    # Load and preprocess main data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'data', 'raw', 'Symptom2Disease.csv')
    main_df = preprocess_data(data_path)
    
    # Find and load additional datasets
    all_dataframes = [main_df]
    dataset_files = find_additional_datasets()
    
    for file_path in dataset_files:
        if file_path != data_path:  # Skip the main dataset we already loaded
            try:
                additional_df = preprocess_data(file_path)
                if len(additional_df) > 0:
                    logger.info(f"Loaded additional dataset from {file_path}: {len(additional_df)} samples")
                    all_dataframes.append(additional_df)
            except Exception as e:
                logger.error(f"Error loading additional dataset {file_path}: {str(e)}")
    
    # Combine all dataframes
    if len(all_dataframes) > 1:
        df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined {len(all_dataframes)} datasets with a total of {len(df)} samples")
    else:
        df = main_df
    
    # Check if we have data
    if len(df) == 0:
        logger.error("No data available for training. Please check your dataset.")
        return None, None
    
    # Apply data augmentation for low-frequency diseases
    disease_counts = df['disease'].value_counts()
    min_samples = 10  # Minimum number of samples per disease
    augmented_samples = []
    
    for disease, count in disease_counts.items():
        if count < min_samples:
            # Find all samples with this disease
            disease_samples = df[df['disease'] == disease]
            
            # Determine how many augmentations to create per sample
            augmentations_per_sample = max(1, min(5, (min_samples - count) // len(disease_samples) + 1))
            
            # Augment each sample
            for _, row in disease_samples.iterrows():
                augmented_texts = augment_symptoms(row['symptoms'], augmentations_per_sample)
                
                # Add new augmented samples (skip the first one which is the original)
                for aug_text in augmented_texts[1:]:
                    augmented_samples.append({
                        'symptoms': aug_text,
                        'disease': disease,
                        'disease_idx': row['disease_idx']
                    })
    
    # Add augmented samples to the dataframe
    if augmented_samples:
        aug_df = pd.DataFrame(augmented_samples)
        df = pd.concat([df, aug_df], ignore_index=True)
        logger.info(f"Added {len(augmented_samples)} augmented samples, new total: {len(df)}")
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['symptoms'].tolist(), df['disease_idx'].tolist(), test_size=0.2, random_state=42, stratify=df['disease_idx']
    )
    
    logger.info(f"Training with {len(train_texts)} samples, validating with {len(val_texts)} samples")
    
    # Select model based on model_type
    if model_type == "bio_clinical_bert":
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        logger.info(f"Using Bio_ClinicalBERT model")
    elif model_type == "pubmed_bert":
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        logger.info(f"Using PubMedBERT model")
    elif model_type == "biogpt":
        model_name = "microsoft/biogpt"
        logger.info(f"Using BioGPT model")
    else:
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        logger.info(f"Using BioBERT model")
    
    # Load tokenizer and tokenize data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    # Create datasets
    train_dataset = SymptomDataset(train_encodings, train_labels)
    val_dataset = SymptomDataset(val_encodings, val_labels)
    
    # Load model
    num_labels = len(df['disease'].unique())
    logger.info(f"Number of unique diseases: {num_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Move model to CPU explicitly
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        # Disable CUDA/GPU usage
        no_cuda=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_type)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save disease classes for later reference
    if hasattr(df, 'attrs') and 'disease_classes' in df.attrs:
        disease_classes = df.attrs['disease_classes']
        disease_classes_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', 'disease_classes.txt'
        )
        with open(disease_classes_path, 'w') as f:
            for disease in disease_classes:
                f.write(f"{disease}\n")
        logger.info(f"Disease classes saved to {disease_classes_path}")
    else:
        # If disease classes are not in attrs, save unique diseases
        unique_diseases = df['disease'].unique()
        disease_classes_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', 'disease_classes.txt'
        )
        with open(disease_classes_path, 'w') as f:
            for disease in unique_diseases:
                f.write(f"{disease}\n")
        logger.info(f"Disease classes saved to {disease_classes_path}")
    
    logger.info(f"Model saved to {model_save_path}")
    return model, tokenizer


def download_disease_info(output_path=None):
    """
    Download or enhance disease information based on the disease classes
    
    Args:
        output_path: Path to save the disease information JSON file
        
    Returns:
        bool: Success status
    """
    # Set default output path if not provided
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', 'disease_info.json'
        )
    
    # Check if disease classes file exists
    disease_classes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data', 'processed', 'disease_classes.txt'
    )
    
    # Load existing disease info if available
    existing_info = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_info = json.load(f)
            logger.info(f"Loaded existing disease info for {len(existing_info)} diseases")
        except Exception as e:
            logger.error(f"Error loading existing disease info: {str(e)}")
    
    if not os.path.exists(disease_classes_path):
        logger.error(f"Disease classes file not found at {disease_classes_path}")
        return False
    
    try:
        # Load disease classes
        with open(disease_classes_path, 'r') as f:
            disease_classes = [line.strip() for line in f.readlines()]
        
        logger.info(f"Processing information for {len(disease_classes)} diseases")
        
        # Template for basic disease information
        template = {
            "description": "",
            "symptoms": [],
            "causes": [],
            "risk_factors": [],
            "treatment": [],
            "when_to_see_doctor": ""
        }
        
        # Add basic info for each disease
        disease_info = {}
        for disease in disease_classes:
            if disease and disease not in existing_info:
                # Create a basic entry for this disease
                disease_info[disease] = template.copy()
            elif disease:
                # Keep existing info
                disease_info[disease] = existing_info[disease]
        
        # Save the combined information
        with open(output_path, 'w') as f:
            json.dump(disease_info, f, indent=2)
        
        logger.info(f"Disease information template saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating disease info: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the health classification model')
    parser.add_argument('--model_type', type=str, default='bio_clinical_bert',
                        choices=['bio_clinical_bert', 'pubmed_bert', 'biogpt', 'biobert'],
                        help='Type of model to use')
    parser.add_argument('--download_info_only', action='store_true',
                        help='Only download/update disease information without training')
    args = parser.parse_args()
    
    if args.download_info_only:
        download_disease_info()
    else:
        model, tokenizer = train_model(model_type=args.model_type)
        # After training, update disease info templates
        download_disease_info()