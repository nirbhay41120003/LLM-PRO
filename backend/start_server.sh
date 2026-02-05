#!/bin/bash
# Backend Server Startup Script
# Run this on your Mac Mini to start the health assistant API

set -e

# Configuration
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-4}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Health Assistant API Server${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if we're in the backend directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found. Please run this script from the backend directory.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt --quiet

# Set environment variables
export FLASK_ENV=production
export TOKENIZERS_PARALLELISM=false
export CORS_ORIGINS="*"

# Check if model is trained
if [ ! -d "app/models/biobert_finetuned" ]; then
    echo -e "${YELLOW}Warning: No trained model found.${NC}"
    echo -e "${YELLOW}Run: python -m app.utils.train_pipeline${NC}"
fi

echo ""
echo -e "${GREEN}Starting server on ${HOST}:${PORT}${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}"
echo ""

# Run with gunicorn for production
if command -v gunicorn &> /dev/null; then
    echo -e "${GREEN}Running with Gunicorn (production mode)${NC}"
    gunicorn --bind ${HOST}:${PORT} --workers ${WORKERS} --timeout 120 "app:create_app()"
else
    echo -e "${YELLOW}Running with Flask development server${NC}"
    echo -e "${YELLOW}For production, install gunicorn: pip install gunicorn${NC}"
    python app.py --host ${HOST} --port ${PORT}
fi
