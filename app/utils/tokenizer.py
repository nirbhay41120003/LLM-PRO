from transformers import AutoTokenizer
import os

# Path to the saved model's tokenizer
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'models', 'biobert_model')

def get_tokenizer():
    """
    Load the tokenizer from the saved model directory if it exists,
    otherwise load the default BERT tokenizer.
    
    Returns:
        A transformers tokenizer
    """
    try:
        if os.path.exists(MODEL_DIR):
            return AutoTokenizer.from_pretrained(MODEL_DIR)
        else:
            return AutoTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        # Fallback to default BERT tokenizer if there's an error
        return AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_input(text, max_length=128):
    """
    Tokenize input text for the model.
    
    Args:
        text: The input text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Tokenized input as a dictionary
    """
    tokenizer = get_tokenizer()
    
    # Clean and preprocess the text
    text = text.lower().strip()
    
    # Tokenize the text
    encoded_input = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'  # Return PyTorch tensors
    )
    
    return encoded_input