# Health Assistance Chatbot

A medical chatbot that uses natural language processing and deep learning to analyze symptoms and suggest possible medical conditions with comprehensive disease information.

## Overview

This project implements a health assistance chatbot that:
- Takes user symptoms as input
- Uses a fine-tuned medical language model to analyze the symptoms
- Suggests possible medical conditions based on the symptoms
- Provides comprehensive disease information including symptoms, causes, treatments, and when to see a doctor

## Project Structure

```
health-chatbot/                  # Root directory
│
├── app/                         # Backend application
│   ├── __init__.py              # Initializes the Flask app
│   ├── routes/                  # API routes
│   │   ├── main_routes.py       # Main interface routes
│   │   ├── chat_routes.py       # Chat API endpoints
│   │   └── auth_routes.py       # Authentication routes
│   ├── models/                  # Deep learning models
│   │   ├── bio_clinical_bert/   # Saved Bio_ClinicalBERT model
│   │   ├── biobert_model/       # Saved BioBERT model (legacy)
│   │   └── train_model.py       # Script to train the model
│   ├── utils/                   # Utility functions
│   │   ├── data_preprocessing.py # Data cleaning and preprocessing
│   │   ├── health_model.py      # Health model handler
│   │   ├── report_analyzer.py   # Medical report analyzer
│   │   └── tokenizer.py         # Tokenizer functions
│   └── config.py                # Configuration settings
│
├── data/                        # Datasets
│   ├── raw/                     # Raw datasets (e.g., Symptom2Disease.csv)
│   ├── processed/               # Processed datasets
│   │   ├── disease_classes.txt  # List of diseases the model can classify
│   │   └── disease_info.json    # Comprehensive disease information
│   └── README.md                # Dataset documentation
│
├── templates/                   # Frontend HTML templates
│   └── index.html               # Main chatbot interface
│
├── static/                      # Static files (CSS, JS, images)
│   ├── css/
│   │   └── styles.css           # Custom CSS for the frontend
│   └── js/
│       └── script.js            # JavaScript for frontend interactivity
│
├── scripts/                     # Helper scripts
│   └── train_and_prepare_model.py # Script to train models and prepare disease info
│
├── tests/                       # Unit and integration tests
│   ├── test_routes.py           # Test API endpoints
│   └── test_model.py            # Test the deep learning model
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── app.py                       # Main entry point for the Flask app
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/health-chatbot.git
cd health-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset:
   - Place your symptom-disease dataset in the `data/raw/` directory
   - The dataset should have columns for symptoms and diseases

2. Train the model with improved medical language models:
```bash
# Train with Bio_ClinicalBERT (recommended)
python scripts/train_and_prepare_model.py --train --model-type bio_clinical_bert

# Or train with PubMedBERT
python scripts/train_and_prepare_model.py --train --model-type pubmed_bert

# Or use the original BioBERT
python scripts/train_and_prepare_model.py --train --model-type biobert
```

3. Prepare or update disease information:
```bash
# Generate templates for disease information
python scripts/train_and_prepare_model.py --prepare-info

# To overwrite existing disease information
python scripts/train_and_prepare_model.py --prepare-info --overwrite
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## Model

This project supports multiple biomedical language models:

1. **Bio_ClinicalBERT** (default): A BERT model trained on clinical notes from MIMIC-III, providing better understanding of clinical language and symptoms.

2. **PubMedBERT**: A BERT model trained on PubMed abstracts and full-text articles, offering strong performance on biomedical text.

3. **BioBERT** (legacy): The original biomedical language model trained on biomedical text.

The models are fine-tuned on symptom-disease datasets to predict medical conditions based on symptom descriptions.

## Disease Information

The chatbot provides comprehensive information about diseases, including:
- Detailed descriptions
- Common symptoms
- Causes and risk factors
- Treatment options
- When to see a doctor

This information is stored in `data/processed/disease_info.json` and can be edited to include more details for each disease.

## Note

This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License

[MIT License](LICENSE)

## Contributors

- Your Name - Initial work 