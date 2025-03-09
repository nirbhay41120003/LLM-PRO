# Health Assistance Chatbot

A medical chatbot that uses natural language processing and deep learning to analyze symptoms and suggest possible medical conditions.

## Overview

This project implements a health assistance chatbot that:
- Takes user symptoms as input
- Uses a fine-tuned BioBERT model to analyze the symptoms
- Suggests possible medical conditions based on the symptoms

## Project Structure

```
health-chatbot/                  # Root directory
│
├── app/                         # Backend application
│   ├── __init__.py              # Initializes the Flask app
│   ├── routes.py                # Defines API routes
│   ├── models/                  # Deep learning models
│   │   ├── biobert_model/       # Saved BioBERT model
│   │   └── train_model.py       # Script to train the model
│   ├── utils/                   # Utility functions
│   │   ├── data_preprocessing.py # Data cleaning and preprocessing
│   │   └── tokenizer.py         # Tokenizer functions
│   └── config.py                # Configuration settings
│
├── data/                        # Datasets
│   ├── raw/                     # Raw datasets (e.g., Symptom2Disease.csv)
│   ├── processed/               # Processed datasets
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

2. Train the model:
```bash
python app/models/train_model.py
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://127.0.0.1:5000`

## Model

This project uses a BioBERT model fine-tuned on a symptom-disease dataset. BioBERT is a pre-trained biomedical language representation model designed for biomedical text mining tasks.

## Note

This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License

[MIT License](LICENSE)

## Contributors

- Your Name - Initial work 