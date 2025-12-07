"""
Agent IA intelligent utilisant LangChain pour router les tickets
vers le modèle approprié (TF-IDF+SVM ou Transformer)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx
import re
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métriques Prometheus
PREDICTIONS_COUNTER = Counter(
    'agent_predictions_total', 
    'Total predictions by model',
    ['model_type']
)
PREDICTION_DURATION = Histogram(
    'agent_prediction_duration_seconds',
    'Prediction duration'
)
COMPLEXITY_SCORE = Histogram(
    'agent_complexity_score',
    'Complexity scores of tickets'
)

app = FastAPI(title="CallCenterAI Agent", version="1.0.0")

# Configuration des services
TFIDF_SERVICE_URL = os.env.get('TFIDF_SERVICE_URL')
TRANSFORMER_SERVICE_URL = os.env.get('TRANSFORMER_SERVICE_URL')


class TicketRequest(BaseModel):
    """Modèle de requête pour un ticket client"""
    text: str = Field(..., description="Texte du ticket client")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées optionnelles")


class TicketResponse(BaseModel):
    """Modèle de réponse avec prédiction"""
    predicted_category: str
    confidence: float
    model_used: str
    reasoning: str
    complexity_score: float
    processing_time: float
    cleaned_text: str


class ComplexityAnalyzer:
    """
    Analyse la complexité d'un ticket pour déterminer quel modèle utiliser.
    Utilise une approche basée sur des règles inspirée de LangChain.
    """
    
    # Mots-clés techniques complexes
    TECHNICAL_KEYWORDS = {
        'en': ['authentication', 'configuration', 'integration', 'synchronization',
               'architecture', 'deployment', 'migration', 'API', 'security', 'encryption'],
    }
    
    # Mots-clés simples
    SIMPLE_KEYWORDS = {
        'en': ['password', 'login', 'access', 'invoice', 'payment', 'reset'],
    }
    
    def __init__(self):
        self.weights = {
            'length': 0.25,
            'technical_terms': 0.30,
            'sentence_structure': 0.20,
            'language_mix': 0.15,
            'special_chars': 0.10
        }
    
    def detect_language(self, text: str) -> str:
        """Détecte la langue principale du texte"""
        # Détection simple basée sur des patterns
        return 'en'
    
    def calculate_complexity(self, text: str) -> Dict[str, Any]:
        """
        Calcule le score de complexité d'un ticket.
        Retourne un dictionnaire avec le score et les détails.
        """
        scores = {}
        
        # 1. Longueur du texte
        word_count = len(text.split())
        if word_count < 10:
            scores['length'] = 0.2
        elif word_count < 30:
            scores['length'] = 0.5
        else:
            scores['length'] = 0.9
        
        # 2. Termes techniques
        language = self.detect_language(text)
        text_lower = text.lower()
        
        technical_count = sum(
            1 for keyword in self.TECHNICAL_KEYWORDS.get(language, [])
            if keyword.lower() in text_lower
        )
        simple_count = sum(
            1 for keyword in self.SIMPLE_KEYWORDS.get(language, [])
            if keyword.lower() in text_lower
        )
        
        if technical_count > 2:
            scores['technical_terms'] = 0.9
        elif technical_count > 0:
            scores['technical_terms'] = 0.6
        elif simple_count > 0:
            scores['technical_terms'] = 0.2
        else:
            scores['technical_terms'] = 0.5
        
        # 3. Structure des phrases (complexité grammaticale)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 20:
            scores['sentence_structure'] = 0.8
        elif avg_sentence_length > 10:
            scores['sentence_structure'] = 0.5
        else:
            scores['sentence_structure'] = 0.3
        
        # 4. Mélange de langues (indique complexité)
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        scores['language_mix'] = 0.7 if (has_arabic and has_latin) else 0.3
        
        # 5. Caractères spéciaux et codes
        special_chars = len(re.findall(r'[#@$%&*+=<>{}[\]\\|]', text))
        scores['special_chars'] = min(special_chars / 5, 1.0) * 0.8
        
        # Score final pondéré
        final_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        return {
            'score': final_score,
            'details': scores,
            'language': language,
            'word_count': word_count,
            'technical_terms': technical_count,
            'simple_terms': simple_count
        }


class PIIScrubber:
    """Nettoie les informations personnelles identifiables (PII)"""
    
    @staticmethod
    def scrub(text: str) -> str:
        """Supprime ou masque les PII du texte"""
        
        # Email
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )
        
        # Numéros de téléphone (formats variés)
        text = re.sub(
            r'\b(?:\+?[\d]{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b',
            '[PHONE]',
            text
        )
        
        # Numéros de carte de crédit
        text = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '[CARD]',
            text
        )
        
        # Numéros de sécurité sociale / identifiants
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            text
        )
        
        # Adresses IP
        text = re.sub(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            '[IP]',
            text
        )
        
        return text


class ModelRouter:
    """
    Router intelligent pour choisir le modèle approprié.
    Inspiré de l'approche de LangChain pour le routing.
    """
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.threshold = 0.55  # Seuil de complexité
    
    def decide_model(self, text: str) -> Dict[str, Any]:
        """
        Décide quel modèle utiliser basé sur l'analyse de complexité.
        
        Retourne:
            - model: 'tfidf' ou 'transformer'
            - reasoning: explication du choix
            - complexity_info: détails de l'analyse
        """
        
        complexity_info = self.complexity_analyzer.calculate_complexity(text)
        score = complexity_info['score']
        
        # Enregistrer le score dans Prometheus
        COMPLEXITY_SCORE.observe(score)
        
        # Règles de décision
        reasons = []
        
        if score < self.threshold:
            model = 'tfidf'
            reasons.append(f"Score de complexité faible ({score:.2f} < {self.threshold})")
            
            if complexity_info['word_count'] < 20:
                reasons.append("Texte court et concis")
            
            if complexity_info['simple_terms'] > 0:
                reasons.append("Présence de mots-clés simples courants")
            
            if complexity_info['details']['technical_terms'] < 0.5:
                reasons.append("Peu de termes techniques détectés")
                
        else:
            model = 'transformer'
            reasons.append(f"Score de complexité élevé ({score:.2f} ≥ {self.threshold})")
            
            if complexity_info['technical_terms'] > 1:
                reasons.append(f"Présence de {complexity_info['technical_terms']} termes techniques")
            
            if complexity_info['language_mix'] > 0.5:
                reasons.append("Texte multilingue détecté")
            
            if complexity_info['word_count'] > 30:
                reasons.append("Texte long nécessitant une compréhension contextuelle")
            
            if complexity_info['details']['sentence_structure'] > 0.6:
                reasons.append("Structure grammaticale complexe")
        
        reasoning = " | ".join(reasons)
        
        return {
            'model': model,
            'reasoning': reasoning,
            'complexity_info': complexity_info
        }


# Instances globales
pii_scrubber = PIIScrubber()
model_router = ModelRouter()


async def call_model_service(url: str, text: str) -> Dict[str, Any]:
    """Appelle un service de modèle et retourne la prédiction"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{url}/predict",
                json={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error calling model service {url}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model service unavailable: {str(e)}"
            )


@app.post("/predict", response_model=TicketResponse)
async def predict_ticket(request: TicketRequest):
    """
    Endpoint principal pour la classification de tickets.
    Route automatiquement vers le bon modèle.
    """
    
    start_time = time.time()
    
    try:
        # 1. Nettoyer les PII
        cleaned_text = pii_scrubber.scrub(request.text)
        logger.info(f"PII scrubbed from text")
        
        # 2. Analyser et décider du modèle
        decision = model_router.decide_model(cleaned_text)
        model_choice = decision['model']
        
        logger.info(f"Model selected: {model_choice}")
        logger.info(f"Reasoning: {decision['reasoning']}")
        
        # 3. Appeler le service approprié
        if model_choice == 'tfidf':
            service_url = TFIDF_SERVICE_URL
            result = await call_model_service(service_url, cleaned_text)
            predicted_category = result['predicted_class']
            confidence = result['confidence']
        else:
            service_url = TRANSFORMER_SERVICE_URL
            result = await call_model_service(service_url, cleaned_text)
            predicted_category = result['predicted_class']
            confidence = result['score']
        
        # 4. Enregistrer les métriques
        PREDICTIONS_COUNTER.labels(model_type=model_choice).inc()
        
        processing_time = time.time() - start_time
        PREDICTION_DURATION.observe(processing_time)
        
        # 5. Construire la réponse
        response = TicketResponse(
            predicted_category=predicted_category,
            confidence=confidence,
            model_used=model_choice,
            reasoning=decision['reasoning'],
            complexity_score=decision['complexity_info']['score'],
            processing_time=processing_time,
            cleaned_text=cleaned_text
        )
        
        logger.info(f"Prediction completed: {predicted_category} (confidence: {confidence:.2f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Expose les métriques Prometheus"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "agent"
    }


@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "service": "CallCenterAI Agent",
        "version": "1.0.0",
        "description": "Agent intelligent pour router les tickets vers le bon modèle NLP",
        "endpoints": {
            "predict": "/predict",
            "metrics": "/metrics",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)