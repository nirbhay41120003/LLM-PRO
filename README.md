# Health Assistant - AI Medical Chatbot

An AI-powered health assistant using **BioBERT** fine-tuned for medical symptom classification and **RAG (Retrieval-Augmented Generation)** for comprehensive medical responses.

## Live Demo

| Service | URL |
|---------|-----|
| **Frontend** | https://www.nirbhay.engineer/ |
| **Backend API** | https://health-api.nirbhay.engineer |

## Features

- **BioBERT Classifier** - Fine-tuned on 3,628 medical samples for 43 disease classes
- **RAG System** - Retrieval-Augmented Generation using FAISS vector search
- **Natural Conversation** - Chat-based symptom analysis interface
- **Severity Analysis** - Assesses symptom severity with recommendations
- **Medical Knowledge Base** - Comprehensive disease information with treatments

## Architecture

```
Frontend (Vercel) - Next.js 14 + Tailwind CSS
        |
        v (HTTPS)
Cloudflare Tunnel - https://health-api.nirbhay.engineer
        |
        v
Backend (Local Server) - Flask + BioBERT + FAISS
```

## Project Structure

```
LLM-PRO/
├── backend/                 # Flask API Server
│   ├── app/
│   │   ├── models/          # Trained BioBERT model
│   │   ├── routes/          # API endpoints
│   │   └── utils/           # BioBERT, RAG, training
│   ├── data/                # Training data
│   ├── app.py               # Server entry point
│   └── requirements.txt
│
├── frontend/                # Next.js App (Vercel)
│   ├── src/app/
│   ├── package.json
│   └── vercel.json
│
└── start_services.sh        # Start backend + tunnel
```

## Quick Start

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the model (first time only)
python -m app.utils.train_pipeline --model biobert --epochs 2 --batch-size 16

# Start the server
python app.py --port 5001
```

### Frontend Setup

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:5001" > .env.local
npm run dev
```

### Deploy to Production

```bash
./start_services.sh
cd frontend && npx vercel --prod
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/status` | GET | Health check |
| `/health` | POST | Analyze symptoms (main) |
| `/analyze` | POST | Disease prediction |
| `/rag-search` | POST | Search knowledge base |



## Model Information

| Component | Details |
|-----------|---------|
| **Base Model** | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| **Training Data** | 3,628 samples, 43 disease classes |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Vector Store** | FAISS |

## Diseases Covered (43 Classes)

Allergy, Arthritis, Bronchial Asthma, Chicken Pox, Common Cold, Dengue, Diabetes, Fungal Infection, GERD, Heart Attack, Hepatitis, Hypertension, Influenza, Jaundice, Malaria, Migraine, Pneumonia, Psoriasis, Tuberculosis, Typhoid, Urinary Tract Infection, and more...

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React 18, Tailwind CSS |
| **Backend** | Flask, Python 3.11 |
| **ML** | PyTorch, Transformers |
| **NLP Model** | BioBERT v1.1 |
| **Vector DB** | FAISS |
| **Deployment** | Vercel, Cloudflare Tunnel |

## Disclaimer

This AI assistant is for **informational purposes only**. It is not a substitute for professional medical advice. Always consult a qualified healthcare provider.

## License

MIT License
