# 🤖 CallCenter AI Router Agent

An intelligent routing agent that automatically classifies customer support tickets and routes them to the appropriate ML model (TF-IDF+SVM or Transformer) based on complexity analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

**Built by:** [Medhedi Maaroufi](https://github.com/medhedimaaroufi) & [Adem Sayadi](https://github.com/AdemSayadi)

---

## 📁 Project Structure

```
MLOps-CallcenterAI/Langchain/
│
├── agent_service.py          # Main FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .env                     # Environment variables (not in git)
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── README.md               # This file
│
├── venv/                   # Virtual environment (local only)
│
└── tests/                  # Test files (optional)
    ├── test_agent.py
    └── test_complexity.py
```

### Key Files

- **`agent_service.py`** - Core routing logic with complexity analyzer, PII scrubber, and model router
- **`requirements.txt`** - Dependencies: FastAPI, httpx, Prometheus client, python-dotenv, etc.
- **`Dockerfile`** - Multi-stage build for optimized container deployment
- **`.env`** - Configuration for service URLs and settings

---

## 🎯 Project Context

### Problem
Call center support tickets vary greatly in complexity:
- Simple: "Reset my password" → Fast classification needed
- Complex: "OAuth integration fails during microservice deployment with TLS errors" → Deep understanding required

### Solution
This **intelligent router agent** analyzes ticket complexity in real-time and automatically routes to:
- **TF-IDF + SVM Model** (fast, simple tickets)
- **Transformer Model** (accurate, complex tickets)

### Architecture

```
Customer Ticket
      ↓
┌─────────────────────┐
│  Router Agent       │
│  ┌──────────────┐   │
│  │ PII Scrubber │   │  ← Removes sensitive data
│  └──────┬───────┘   │
│         ↓           │
│  ┌──────────────┐   │
│  │ Complexity   │   │  ← Analyzes ticket
│  │ Analyzer     │   │     (technical terms, length, etc.)
│  └──────┬───────┘   │
│         ↓           │
│  ┌──────────────┐   │
│  │ Model Router │   │  ← Decides which model
│  └──────┬───────┘   │
└─────────┼───────────┘
          │
    ┌─────┴─────┐
    ↓           ↓
[TF-IDF]    [Transformer]
 (Fast)      (Accurate)
```

### Key Features

✅ **Intelligent Routing** - Complexity-based model selection  
✅ **PII Protection** - Automatically scrubs emails, phones, credit cards  
✅ **Multi-language** - Supports English, French, Arabic  
✅ **Observability** - Prometheus metrics + health checks  
✅ **Production-Ready** - Docker containerized with security best practices

---

## 🚀 Usage

### 1. Local Development

**Setup:**
```bash
# Clone and navigate
cd MLOps-CallcenterAI/Langchain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your service URLs:
# TFIDF_SERVICE_URL=http://localhost:8001
# TRANSFORMER_SERVICE_URL=http://localhost:8002
```

**Run:**
```bash
uvicorn agent_service:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

---

### 2. Docker Deployment

**Build:**
```bash
docker build -t medhedimaaroufi/callcenterai-router-agent:v1.0.0 .
```

**Run:**
```bash
docker run -d \
  --name callcenter-agent \
  -p 8000:8000 \
  -e TFIDF_SERVICE_URL=http://host.docker.internal:8001 \
  -e TRANSFORMER_SERVICE_URL=http://host.docker.internal:8002 \
  medhedimaaroufi/callcenterai-router-agent:v1.0.0
```

**Check Status:**
```bash
# View logs
docker logs -f callcenter-agent

# Check health
curl http://localhost:8000/health
```

---

### 3. API Examples

#### Simple Ticket (→ TF-IDF Model)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I forgot my password and need to reset it",
    "metadata": {"source": "email"}
  }'
```

**Response:**
```json
{
  "predicted_category": "Account Access",
  "confidence": 0.89,
  "model_used": "tfidf",
  "reasoning": "Score de complexité faible (0.32 < 0.50) | Texte court et concis | Présence de mots-clés simples courants",
  "complexity_score": 0.32,
  "processing_time": 0.45,
  "cleaned_text": "I forgot my password and need to reset it"
}
```

#### Complex Ticket (→ Transformer Model)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "After migrating our platform to microservices architecture, the OAuth integration with our identity provider fails intermittently during high-traffic periods. The logs show: ERROR: handshake_timeout at line 452. Configuration: {\"tlsVersion\": \"1.3\", \"cipher\": \"AES256-GCM\"}",
    "metadata": {"priority": "critical", "source": "chat"}
  }'
```

**Response:**
```json
{
  "predicted_category": "Technical Support",
  "confidence": 0.94,
  "model_used": "transformer",
  "reasoning": "Détection de logs d'erreur techniques | Présence de configuration/code JSON | Score de complexité élevé (0.72 ≥ 0.50) | Nombreux termes techniques (8)",
  "complexity_score": 0.72,
  "processing_time": 1.82,
  "cleaned_text": "After migrating our platform to microservices..."
}
```

#### Python Code Example

```python
import requests

# Configure endpoint
API_URL = "http://localhost:8000/predict"

# Prepare ticket
ticket = {
    "text": "The authentication pipeline crashes during deployment",
    "metadata": {
        "source": "webform",
        "priority": "high",
        "customer_id": "12345"
    }
}

# Send request
response = requests.post(API_URL, json=ticket)
result = response.json()

# Process response
print(f"Category: {result['predicted_category']}")
print(f"Model: {result['model_used']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Reasoning: {result['reasoning']}")
print(f"Processing Time: {result['processing_time']:.2f}s")
```

---

### 4. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify ticket and get prediction |
| `/health` | GET | Health check status |
| `/metrics` | GET | Prometheus metrics |
| `/` | GET | Service information |
| `/docs` | GET | Interactive API documentation (Swagger) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required: Service URLs
TFIDF_SERVICE_URL=http://localhost:8001
TRANSFORMER_SERVICE_URL=http://localhost:8002

# Optional: Logging
LOG_LEVEL=INFO

# Optional: Model Tuning
COMPLEXITY_THRESHOLD=0.50  # 0.0 - 1.0 (default: 0.50)
```

### Complexity Threshold

The threshold determines routing behavior:

- **0.40** - More tickets to Transformer (higher accuracy, slower)
- **0.50** - Balanced (default, recommended)
- **0.60** - More tickets to TF-IDF (faster, lower accuracy for complex cases)

---

## 📊 How Complexity Analysis Works

The agent scores tickets based on 5 factors:

| Factor | Weight | Example |
|--------|--------|---------|
| **Length** | 20% | Short (10 words) = 0.2, Long (70+ words) = 0.9 |
| **Technical Terms** | 35% | "authentication", "API", "encryption", "deployment" |
| **Sentence Structure** | 20% | Avg 25+ words/sentence = 0.9 |
| **Language Mixing** | 15% | English + Arabic + French = 0.8 |
| **Special Characters** | 10% | JSON code, error logs, symbols = 0.9 |

**Final Score** = Σ (Factor × Weight)

**Decision:**
- Score < 0.50 → TF-IDF (fast model)
- Score ≥ 0.50 → Transformer (accurate model)

**Force Rules** (override score):
- Has error logs + technical terms → Transformer
- Contains JSON/config code → Transformer
- Multilingual + technical → Transformer

---

## 🔒 Security Features

### PII Scrubbing

Automatically masks sensitive information:

| Type | Pattern | Replacement |
|------|---------|-------------|
| Email | `user@example.com` | `[EMAIL]` |
| Phone | `+1-555-123-4567` | `[PHONE]` |
| Credit Card | `1234-5678-9012-3456` | `[CARD]` |
| SSN | `123-45-6789` | `[SSN]` |
| IP Address | `192.168.1.1` | `[IP]` |

### Container Security

- ✅ Non-root user (UID 1000)
- ✅ Multi-stage build (minimal attack surface)
- ✅ No secrets in image
- ✅ Health checks enabled

---

## 📈 Monitoring

### Prometheus Metrics

Access at: `http://localhost:8000/metrics`

**Key Metrics:**
```prometheus
# Total predictions by model type
agent_predictions_total{model_type="tfidf"}
agent_predictions_total{model_type="transformer"}

# Prediction latency (histogram)
agent_prediction_duration_seconds

# Complexity score distribution (histogram)
agent_complexity_score
```

**Example Queries:**
```promql
# Predictions per minute
rate(agent_predictions_total[1m])

# Average processing time
rate(agent_prediction_duration_seconds_sum[5m]) / 
rate(agent_prediction_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95, agent_prediction_duration_seconds_bucket)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Service Unavailable (503)**
```bash
# Check if backend services are running
curl http://localhost:8001/health  # TF-IDF
curl http://localhost:8002/health  # Transformer

# Verify environment variables
docker exec callcenter-agent env | grep SERVICE_URL
```

**2. KeyError in Complexity Analysis**
```bash
# Fixed in v1.1.0 - Update to latest version
docker pull medhedimaaroufi/callcenterai-router-agent:v1.0.0
```

**3. Container Health Check Failing**
```bash
# Check logs
docker logs callcenter-agent

# Restart container
docker restart callcenter-agent
```

---

## 📝 Dependencies

### Main Libraries

- **FastAPI** (0.104+) - Web framework
- **httpx** - Async HTTP client
- **Pydantic** - Data validation
- **prometheus-client** - Metrics
- **python-dotenv** - Environment management
- **uvicorn** - ASGI server

### Full List

See `requirements.txt` for complete dependency list.

---

## 👥 Authors

- **Medhedi Maaroufi** - MLOps Engineer & Lead Developer - [@medhedimaaroufi](https://github.com/medhedimaaroufi)
- **Adem Sayadi** - ML Engineer & Co-Developer - [@SayadiAdem](https://github.com/AdemSayadi)

