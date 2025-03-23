import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the Flask environment
os.environ['FLASK_ENV'] = os.environ.get('FLASK_ENV', 'production')

from app import create_app
application = create_app()

if __name__ == "__main__":
    application.run() 