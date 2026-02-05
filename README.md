# Health Assistant - AI Medical Chatbot

An AI-powered health assistant using **BioBERT** fine-tuned for medical symptom classification and **RAG (Retrieval-Augmented Generation)** for comprehensive medical responses.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Vercel)                        │
│                    Next.js + Tailwind CSS                        │
│                    https://your-app.vercel.app                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS API Calls
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Mac Mini Server)                     │
│                    Flask + BioBERT + FAISS                       │
│                    http://your-ip:5000                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  BioBERT    │  │  RAG System │  │  Medical Knowledge Base │  │
│  │  Classifier │  │  (FAISS)    │  │  (43 Diseases)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
LLM-PRO/
├── backend/                 # Flask API Server (runs on Mac Mini)
│   ├── app/
│   │   ├── models/          # Trained models
│   │   ├── routes/          # API endpoints
│   │   └── utils/           # BioBERT, RAG, etc.
│   ├── data/                # Training data & knowledge base
│   ├── app.py               # Server entry point
│   ├── requirements.txt     # Python dependencies
│   └── start_server.sh      # Production startup script
│
└── frontend/                # Next.js App (deploys to Vercel)
    ├── src/app/             # App pages
    ├── package.json
    └── vercel.json          # Vercel configuration
```

## 🚀 Quick Start

### 1. Backend Setup (Mac Mini)

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only)
python -m app.utils.train_pipeline --model biobert --epochs 2 --batch-size 16

# Start the server
chmod +x start_server.sh
./start_server.sh
# Or manually:
python app.py --host 0.0.0.0 --port 5000
```

### 2. Frontend Setup (Local Development)

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local with your backend URL

# Run development server
npm run dev
```

### 3. Deploy Frontend to Vercel

```bash
cd frontend

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variable in Vercel dashboard:
# NEXT_PUBLIC_API_URL = https://your-mac-mini-domain.com:5000
```

## 🌐 Exposing Backend to Internet

To allow Vercel frontend to access your Mac Mini backend:

### Option 1: Port Forwarding (Simple)
1. Forward port 5000 on your router to Mac Mini's IP
2. Use your public IP: `http://YOUR_PUBLIC_IP:5000`

### Option 2: Cloudflare Tunnel (Recommended)
```bash
# Install cloudflared
brew install cloudflared

# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create health-api

# Run tunnel
cloudflared tunnel --url http://localhost:5000
```

### Option 3: ngrok (Quick Testing)
```bash
# Install ngrok
brew install ngrok

# Expose port
ngrok http 5000
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/status` | GET | Check API status |
| `/health` | POST | Analyze symptoms (main endpoint) |
| `/analyze` | POST | Get disease prediction |
| `/rag-search` | POST | Search medical knowledge |
| `/general` | POST | General health queries |

### Example Request

```bash
curl -X POST https://your-api.com/health \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and headache for 3 days"}'
```

### Example Response

```json
{
  "disease": "Malaria",
  "confidence": 20.1,
  "response": "Based on your symptoms, the most likely condition is **Malaria**...",
  "alternatives": [
    {"disease": "Dengue", "confidence": 15.2},
    {"disease": "Typhoid", "confidence": 12.8}
  ],
  "severity_analysis": {
    "severity": "moderate",
    "score": 5
  }
}
```

## 🔧 Configuration

### Backend Environment Variables

```bash
PORT=5000                    # Server port
HOST=0.0.0.0                # Server host
CORS_ORIGINS=*              # Allowed origins (comma-separated)
HEALTH_MODEL_TYPE=biobert   # Model type
TOKENIZERS_PARALLELISM=false
```

### Frontend Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:5000  # Backend URL
```

## 🧠 Model Information

- **Base Model**: BioBERT v1.1 (dmis-lab/biobert-v1.1)
- **Training Data**: 3,628 samples across 43 diseases
- **Embedding Model**: all-MiniLM-L6-v2 for RAG
- **Vector Store**: FAISS for fast similarity search

### Supported Models

| Model | Description |
|-------|-------------|
| `biobert` | BioBERT v1.1 - PubMed abstracts |
| `pubmedbert` | PubMedBERT - Full PubMed corpus |
| `bio_clinical_bert` | Clinical notes |
| `scibert` | Scientific papers |

## 🧪 Testing

```bash
cd backend

# Run all tests
python test_model.py

# Test API manually
curl http://localhost:5000/api/status
```

## 📊 Diseases Covered

Allergy, Arthritis, Bronchial Asthma, Cervical Spondylosis, Chicken Pox, Common Cold, Dengue, Diabetes, Drug Reaction, Fungal Infection, Gastroenteritis, GERD, Heart Attack, Hepatitis, Hypertension, Hypoglycemia, Influenza, Jaundice, Malaria, Migraine, Pneumonia, Psoriasis, Tuberculosis, Typhoid, Urinary Tract Infection, and more...

## ⚠️ Disclaimer

This AI assistant is for **informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
