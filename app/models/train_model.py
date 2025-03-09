import os
import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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

def train_model():
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'data', 'raw', 'Symptom2Disease.csv')
    df = preprocess_data(data_path)
    
    # Check if we have data
    if len(df) == 0:
        print("No data available for training. Please check your dataset.")
        return None, None
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['symptoms'].tolist(), df['disease_idx'].tolist(), test_size=0.2, random_state=42
    )
    
    print(f"Training with {len(train_texts)} samples, validating with {len(val_texts)} samples")
    
    # Load tokenizer and tokenize data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    # Create datasets
    train_dataset = SymptomDataset(train_encodings, train_labels)
    val_dataset = SymptomDataset(val_encodings, val_labels)
    
    # Load model
    num_labels = len(df['disease'].unique())
    print(f"Number of unique diseases: {num_labels}")
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=num_labels
    )
    
    # Move model to CPU explicitly
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
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
    trainer.train()
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biobert_model')
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
        print(f"Disease classes saved to {disease_classes_path}")
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
        print(f"Disease classes saved to {disease_classes_path}")
    
    print(f"Model saved to {model_save_path}")
    return model, tokenizer

if __name__ == "__main__":
    train_model()