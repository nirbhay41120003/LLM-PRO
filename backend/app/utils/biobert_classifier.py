"""
BioBERT Fine-Tuning Module for Medical Question Answering and Symptom Classification
Supports multiple medical BERT variants: BioBERT, PubMedBERT, Bio_ClinicalBERT
"""

import os
import torch
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("transformers library not available")


# Supported model configurations
MODEL_CONFIGS = {
    "biobert": {
        "model_name": "dmis-lab/biobert-v1.1",
        "description": "BioBERT v1.1 - Pre-trained on PubMed abstracts"
    },
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "description": "PubMedBERT - Pre-trained on PubMed abstracts and full-text"
    },
    "bio_clinical_bert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "Bio_ClinicalBERT - Pre-trained on clinical notes"
    },
    "scibert": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "description": "SciBERT - Pre-trained on scientific papers"
    }
}


class SymptomDataset(Dataset):
    """Dataset class for symptom-disease classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BioBERTClassifier:
    """BioBERT-based classifier for symptom-disease classification"""
    
    def __init__(
        self,
        model_type: str = "biobert",
        num_classes: int = None,
        model_dir: str = None,
        max_length: int = 256
    ):
        self.model_type = model_type
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.label_encoder = {}
        self.label_decoder = {}
        
        # Determine model directory
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', f'{model_type}_finetuned'
            )
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_pretrained(self):
        """Load pre-trained BioBERT model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for BioBERT")
        
        config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["biobert"])
        model_name = config["model_name"]
        
        logger.info(f"Loading pre-trained model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.num_classes:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.to(device)
        logger.info(f"Model loaded successfully: {config['description']}")
    
    def load_finetuned(self) -> bool:
        """Load fine-tuned model from disk"""
        if not os.path.exists(os.path.join(self.model_dir, 'config.json')):
            logger.warning(f"No fine-tuned model found at {self.model_dir}")
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(device)
            self.model.eval()
            
            # Load label mappings
            label_map_path = os.path.join(self.model_dir, 'label_mapping.json')
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    mappings = json.load(f)
                    self.label_encoder = mappings.get('encoder', {})
                    self.label_decoder = {int(k): v for k, v in mappings.get('decoder', {}).items()}
            
            logger.info(f"Loaded fine-tuned model from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Prepare data for training"""
        texts = df['symptoms'].tolist()
        
        # Create label encoding
        unique_labels = sorted(df['disease'].unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        self.num_classes = len(unique_labels)
        
        labels = [self.label_encoder[label] for label in df['disease']]
        
        return texts, labels
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500
    ):
        """Fine-tune the model on symptom-disease data"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for training")
        
        # Prepare data
        train_texts, train_labels = self.prepare_data(train_df)
        
        if val_df is not None:
            val_texts = val_df['symptoms'].tolist()
            val_labels = [self.label_encoder.get(label, 0) for label in val_df['disease']]
        else:
            # Split training data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.1, random_state=42
            )
        
        # Load pre-trained model
        self.load_pretrained()
        
        # Update model for correct number of classes
        if self.model.config.num_labels != self.num_classes:
            config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["biobert"])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["model_name"],
                num_labels=self.num_classes
            )
            self.model.to(device)
        
        # Create datasets
        train_dataset = SymptomDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = SymptomDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.model_dir, 'logs'),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to="none",
            use_cpu=True  # Force CPU
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        logger.info(f"Starting training with {len(train_dataset)} samples...")
        trainer.train()
        
        # Save model
        self.save_model()
        
        # Evaluate
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def save_model(self):
        """Save the fine-tuned model"""
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        
        # Save label mappings
        with open(os.path.join(self.model_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'encoder': self.label_encoder,
                'decoder': {str(k): v for k, v in self.label_decoder.items()}
            }, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def predict(self, text: str, top_k: int = 3) -> List[Dict]:
        """Make prediction for a single text"""
        if self.model is None:
            if not self.load_finetuned():
                raise ValueError("No model loaded. Train or load a model first.")
        
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            disease = self.label_decoder.get(idx.item(), f"Unknown_{idx.item()}")
            predictions.append({
                'disease': disease,
                'confidence': round(prob.item() * 100, 2)
            })
        
        return predictions
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[List[Dict]]:
        """Make predictions for multiple texts"""
        if self.model is None:
            if not self.load_finetuned():
                raise ValueError("No model loaded. Train or load a model first.")
        
        self.model.eval()
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            for probs in probabilities:
                top_probs, top_indices = torch.topk(probs, 3)
                preds = []
                for prob, idx in zip(top_probs, top_indices):
                    disease = self.label_decoder.get(idx.item(), f"Unknown_{idx.item()}")
                    preds.append({
                        'disease': disease,
                        'confidence': round(prob.item() * 100, 2)
                    })
                all_predictions.append(preds)
        
        return all_predictions


def train_biobert_classifier(
    data_path: str = None,
    model_type: str = "biobert",
    epochs: int = 3,
    batch_size: int = 16
) -> BioBERTClassifier:
    """
    Train a BioBERT classifier on symptom-disease data
    
    Args:
        data_path: Path to training CSV (disease, symptoms columns)
        model_type: Type of model to use (biobert, pubmedbert, bio_clinical_bert)
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Trained BioBERTClassifier
    """
    # Load data
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', 'combined_symptoms.csv'
        )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} training samples with {df['disease'].nunique()} diseases")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['disease'])
    
    # Create and train classifier
    classifier = BioBERTClassifier(model_type=model_type)
    classifier.train(train_df, val_df, epochs=epochs, batch_size=batch_size)
    
    return classifier


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BioBERT classifier")
    parser.add_argument("--model", type=str, default="biobert", 
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model type to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    
    args = parser.parse_args()
    
    classifier = train_biobert_classifier(
        data_path=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Test prediction
    test_text = "I have fever, headache, and body pain for the last two days"
    predictions = classifier.predict(test_text)
    print(f"\nTest prediction for: '{test_text}'")
    for pred in predictions:
        print(f"  {pred['disease']}: {pred['confidence']}%")
