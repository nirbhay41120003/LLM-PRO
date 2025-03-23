#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/Users/nirbhay/Desktop/Projects/LLM/health-chatbot-new/backups"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Database backup
pg_dump healthchatbot > "$BACKUP_DIR/db_$TIMESTAMP.sql"

# Application backup
tar -czf "$BACKUP_DIR/app_$TIMESTAMP.tar.gz" \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='logs/*' \
    --exclude='backups/*' \
    .

# Keep only last 7 days of backups
find "$BACKUP_DIR" -type f -mtime +7 -delete 