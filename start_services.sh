#!/bin/bash
# Start Health Chatbot Services

# Kill any existing processes
pkill -f "python app.py" 2>/dev/null
pkill -f "cloudflared tunnel run" 2>/dev/null

# Wait a moment
sleep 2

# Start backend
cd /Users/nirbhay/Desktop/LLM-PRO/backend
export TOKENIZERS_PARALLELISM=false
nohup python app.py --port 5001 > /tmp/health_api.log 2>&1 &
echo "Backend started on port 5001"

# Wait for backend to initialize
sleep 5

# Start Cloudflare tunnel
nohup cloudflared tunnel run health-api > /tmp/cloudflared.log 2>&1 &
echo "Cloudflare tunnel started"

# Wait and verify
sleep 5
curl -s https://health-api.nirbhay.engineer/ && echo ""
echo "Services running!"
