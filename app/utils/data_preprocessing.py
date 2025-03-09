import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):
    """
    Preprocess the symptom-disease dataset.
    
    Args:
        filepath: Path to the CSV file containing symptom-disease data
        
    Returns:
        Preprocessed DataFrame with symptoms and encoded disease labels
    """
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Check if the file has the expected columns
        if 'label' in df.columns and 'text' in df.columns:
            # Rename columns to match our expected format
            df = df.rename(columns={'text': 'symptoms', 'label': 'disease'})
            print("Using 'label' as disease and 'text' as symptoms")
        elif 'symptoms' not in df.columns or 'disease' not in df.columns:
            # If columns are named differently, try to find them
            if 'symptoms' not in df.columns and 'symptom' in df.columns:
                df.rename(columns={'symptom': 'symptoms'}, inplace=True)
            if 'disease' not in df.columns and 'diagnosis' in df.columns:
                df.rename(columns={'diagnosis': 'disease'}, inplace=True)
                
            # Check again after renaming
            missing_columns = [col for col in ['symptoms', 'disease'] if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean symptoms text
        df['symptoms'] = df['symptoms'].str.lower().str.strip()
        
        # Remove rows with missing values
        df = df.dropna(subset=['symptoms', 'disease'])
        
        # Encode disease labels
        label_encoder = LabelEncoder()
        df['disease_idx'] = label_encoder.fit_transform(df['disease'])
        
        # Store label encoder classes for later reference
        df.attrs['disease_classes'] = label_encoder.classes_.tolist()
        
        print(f"Preprocessed data: {len(df)} rows, {df['disease'].nunique()} unique diseases")
        return df
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        # Return an empty DataFrame with the required columns if there's an error
        return pd.DataFrame(columns=['symptoms', 'disease', 'disease_idx'])