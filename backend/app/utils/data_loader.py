"""
Data Loader Module - Combines and processes multiple medical datasets
Creates unified training data for BioBERT fine-tuning
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """Loads and combines medical datasets from multiple sources"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data'
        )
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.external_dir = os.path.join(self.raw_dir, 'external')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # Ensure directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_original_dataset(self) -> pd.DataFrame:
        """Load the original Symptom2Disease dataset"""
        path = os.path.join(self.raw_dir, 'Symptom2Disease.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={'label': 'disease', 'text': 'symptoms'})
            logger.info(f"Loaded original dataset: {len(df)} samples")
            return df[['disease', 'symptoms']]
        return pd.DataFrame(columns=['disease', 'symptoms'])
    
    def load_disease_symptom_dataset(self) -> pd.DataFrame:
        """Load the disease-symptom dataset from external sources"""
        path = os.path.join(self.external_dir, 'dataset.csv')
        if not os.path.exists(path):
            return pd.DataFrame(columns=['disease', 'symptoms'])
        
        df = pd.read_csv(path)
        
        # Convert symptom columns to text descriptions
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
        
        records = []
        for _, row in df.iterrows():
            disease = row['Disease']
            symptoms = []
            for col in symptom_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    symptom = str(row[col]).strip().replace('_', ' ')
                    symptoms.append(symptom)
            
            if symptoms:
                # Create natural language description
                symptom_text = self._create_symptom_description(symptoms)
                records.append({'disease': disease, 'symptoms': symptom_text})
        
        result = pd.DataFrame(records)
        logger.info(f"Loaded disease-symptom dataset: {len(result)} samples")
        return result
    
    def load_symptom_descriptions(self) -> Dict[str, str]:
        """Load symptom descriptions for knowledge base"""
        path = os.path.join(self.external_dir, 'symptom_Description.csv')
        if not os.path.exists(path):
            return {}
        
        df = pd.read_csv(path)
        descriptions = {}
        for _, row in df.iterrows():
            if 'Disease' in df.columns and 'Description' in df.columns:
                descriptions[row['Disease']] = row['Description']
        
        logger.info(f"Loaded {len(descriptions)} disease descriptions")
        return descriptions
    
    def load_symptom_precautions(self) -> Dict[str, List[str]]:
        """Load precautions for each disease"""
        path = os.path.join(self.external_dir, 'symptom_precaution.csv')
        if not os.path.exists(path):
            return {}
        
        df = pd.read_csv(path)
        precautions = {}
        precaution_cols = [col for col in df.columns if col.startswith('Precaution_')]
        
        for _, row in df.iterrows():
            disease = row['Disease']
            disease_precautions = []
            for col in precaution_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    disease_precautions.append(str(row[col]).strip())
            if disease_precautions:
                precautions[disease] = disease_precautions
        
        logger.info(f"Loaded precautions for {len(precautions)} diseases")
        return precautions
    
    def load_symptom_severity(self) -> Dict[str, int]:
        """Load symptom severity weights"""
        path = os.path.join(self.external_dir, 'Symptom-severity.csv')
        if not os.path.exists(path):
            return {}
        
        df = pd.read_csv(path)
        severity = {}
        for _, row in df.iterrows():
            if 'Symptom' in df.columns and 'weight' in df.columns:
                symptom = str(row['Symptom']).strip().replace('_', ' ')
                severity[symptom] = int(row['weight'])
        
        logger.info(f"Loaded severity for {len(severity)} symptoms")
        return severity
    
    def _create_symptom_description(self, symptoms: List[str]) -> str:
        """Create natural language description from symptom list"""
        templates = [
            "I have been experiencing {symptoms}.",
            "I am suffering from {symptoms}.",
            "My symptoms include {symptoms}.",
            "I have {symptoms}.",
            "I've been having {symptoms} for a few days.",
            "I'm experiencing {symptoms} and feeling unwell.",
        ]
        
        if len(symptoms) == 1:
            symptom_text = symptoms[0]
        elif len(symptoms) == 2:
            symptom_text = f"{symptoms[0]} and {symptoms[1]}"
        else:
            symptom_text = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
        
        template = np.random.choice(templates)
        return template.format(symptoms=symptom_text)
    
    def create_combined_dataset(self, augment: bool = True) -> pd.DataFrame:
        """Combine all datasets into one unified dataset"""
        datasets = []
        
        # Load original dataset
        original = self.load_original_dataset()
        if len(original) > 0:
            datasets.append(original)
        
        # Load external disease-symptom dataset
        external = self.load_disease_symptom_dataset()
        if len(external) > 0:
            datasets.append(external)
        
        if not datasets:
            logger.error("No datasets found!")
            return pd.DataFrame(columns=['disease', 'symptoms'])
        
        # Combine datasets
        combined = pd.concat(datasets, ignore_index=True)
        
        # Normalize disease names
        combined['disease'] = combined['disease'].apply(self._normalize_disease_name)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['symptoms'])
        
        # Augment data if requested
        if augment:
            combined = self._augment_dataset(combined)
        
        logger.info(f"Combined dataset: {len(combined)} samples, {combined['disease'].nunique()} diseases")
        return combined
    
    def _normalize_disease_name(self, name: str) -> str:
        """Normalize disease names for consistency"""
        name = str(name).strip()
        # Common normalizations
        mappings = {
            'fungal infection': 'Fungal Infection',
            'allergy': 'Allergy',
            'gerd': 'GERD',
            'chronic cholestasis': 'Chronic Cholestasis',
            'drug reaction': 'Drug Reaction',
            'peptic ulcer diseae': 'Peptic Ulcer Disease',
            'peptic ulcer disease': 'Peptic Ulcer Disease',
            'aids': 'AIDS',
            'diabetes': 'Diabetes',
            'diabetes ': 'Diabetes',
            'gastroenteritis': 'Gastroenteritis',
            'bronchial asthma': 'Bronchial Asthma',
            'hypertension': 'Hypertension',
            'hypertension ': 'Hypertension',
            'migraine': 'Migraine',
            'cervical spondylosis': 'Cervical Spondylosis',
            'paralysis (brain hemorrhage)': 'Paralysis (Brain Hemorrhage)',
            'jaundice': 'Jaundice',
            'malaria': 'Malaria',
            'chicken pox': 'Chicken Pox',
            'chickenpox': 'Chicken Pox',
            'dengue': 'Dengue',
            'typhoid': 'Typhoid',
            'hepatitis a': 'Hepatitis A',
            'hepatitis b': 'Hepatitis B',
            'hepatitis c': 'Hepatitis C',
            'hepatitis d': 'Hepatitis D',
            'hepatitis e': 'Hepatitis E',
            'alcoholic hepatitis': 'Alcoholic Hepatitis',
            'tuberculosis': 'Tuberculosis',
            'common cold': 'Common Cold',
            'pneumonia': 'Pneumonia',
            'dimorphic hemmorhoids(piles)': 'Hemorrhoids (Piles)',
            'heart attack': 'Heart Attack',
            'varicose veins': 'Varicose Veins',
            'hypothyroidism': 'Hypothyroidism',
            'hyperthyroidism': 'Hyperthyroidism',
            'hypoglycemia': 'Hypoglycemia',
            'osteoarthristis': 'Osteoarthritis',
            'osteoarthritis': 'Osteoarthritis',
            'arthritis': 'Arthritis',
            '(vertigo) paroymam positional vertigo': 'Vertigo',
            'vertigo': 'Vertigo',
            'acne': 'Acne',
            'urinary tract infection': 'Urinary Tract Infection',
            'psoriasis': 'Psoriasis',
            'impetigo': 'Impetigo',
        }
        
        name_lower = name.lower().strip()
        if name_lower in mappings:
            return mappings[name_lower]
        
        # Title case for unknown diseases
        return name.title().strip()
    
    def _augment_dataset(self, df: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
        """Augment dataset with variations"""
        augmented_records = []
        
        for _, row in df.iterrows():
            augmented_records.append(row.to_dict())
            
            # Create variations
            symptoms = row['symptoms']
            disease = row['disease']
            
            for _ in range(multiplier - 1):
                varied = self._create_variation(symptoms)
                if varied != symptoms:
                    augmented_records.append({'disease': disease, 'symptoms': varied})
        
        return pd.DataFrame(augmented_records)
    
    def _create_variation(self, text: str) -> str:
        """Create a variation of symptom text"""
        variations = [
            ("I have been experiencing", "I've been having"),
            ("I have been experiencing", "I am suffering from"),
            ("I am suffering from", "I have"),
            ("I have", "I'm experiencing"),
            ("My symptoms include", "I've noticed"),
            ("for a few days", "recently"),
            ("for a few days", "for some time now"),
            ("and feeling unwell", "and I don't feel well"),
            (", and ", " along with "),
        ]
        
        result = text
        for old, new in variations:
            if old in result and np.random.random() < 0.3:
                result = result.replace(old, new, 1)
                break
        
        return result
    
    def create_disease_info_database(self) -> Dict:
        """Create comprehensive disease information database"""
        descriptions = self.load_symptom_descriptions()
        precautions = self.load_symptom_precautions()
        
        # Load existing disease info
        existing_path = os.path.join(self.processed_dir, 'disease_info.json')
        existing_info = {}
        if os.path.exists(existing_path):
            with open(existing_path, 'r') as f:
                existing_info = json.load(f)
        
        # Get all unique diseases
        combined = self.create_combined_dataset(augment=False)
        all_diseases = combined['disease'].unique()
        
        disease_database = {}
        
        for disease in all_diseases:
            disease_key = disease
            
            # Start with existing info if available
            if disease_key in existing_info and existing_info[disease_key].get('description'):
                disease_database[disease_key] = existing_info[disease_key]
            else:
                # Create new entry
                disease_database[disease_key] = {
                    "description": descriptions.get(disease, f"{disease} is a medical condition that requires proper diagnosis and treatment."),
                    "symptoms": [],
                    "causes": [],
                    "risk_factors": [],
                    "treatment": precautions.get(disease, []),
                    "when_to_see_doctor": "Consult a healthcare provider if symptoms persist or worsen."
                }
        
        return disease_database
    
    def save_processed_data(self):
        """Save all processed data to files"""
        # Save combined dataset
        combined = self.create_combined_dataset(augment=True)
        combined.to_csv(os.path.join(self.processed_dir, 'combined_symptoms.csv'), index=False)
        
        # Save disease classes
        diseases = sorted(combined['disease'].unique())
        with open(os.path.join(self.processed_dir, 'disease_classes.txt'), 'w') as f:
            for disease in diseases:
                f.write(f"{disease}\n")
        
        # Save disease info database
        disease_db = self.create_disease_info_database()
        with open(os.path.join(self.processed_dir, 'disease_info.json'), 'w') as f:
            json.dump(disease_db, f, indent=2)
        
        # Save symptom severity
        severity = self.load_symptom_severity()
        with open(os.path.join(self.processed_dir, 'symptom_severity.json'), 'w') as f:
            json.dump(severity, f, indent=2)
        
        logger.info(f"Saved processed data to {self.processed_dir}")
        logger.info(f"  - {len(combined)} training samples")
        logger.info(f"  - {len(diseases)} disease classes")
        logger.info(f"  - {len(disease_db)} disease info entries")
        
        return {
            'samples': len(combined),
            'diseases': len(diseases),
            'disease_list': diseases
        }


def prepare_training_data():
    """Prepare all training data"""
    loader = MedicalDataLoader()
    return loader.save_processed_data()


if __name__ == "__main__":
    prepare_training_data()
