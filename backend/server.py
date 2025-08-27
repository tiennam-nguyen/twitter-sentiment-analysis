"""
Twitter Sentiment Analysis API
A FastAPI server for sentiment analysis using Enhanced BERT model
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our custom modules
from utils import predict_sentiment_internal, initialize_sentiment_model

# ============== MODEL INITIALIZATION ==============
# Initialize model, tokenizer, and device
model, tokenizer, device = initialize_sentiment_model()

# Sentiment mapping
SENTIMENT_LABELS = {
    0: "positive",
    1: "negative", 
    2: "neutral",
    3: "irrelevant"
}

# ============== FASTAPI APP SETUP ==============
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Enhanced BERT model for sentiment analysis with Google Drive integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== REQUEST/RESPONSE MODELS ==============
class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    processed_text: str

# ============== API ENDPOINTS ==============
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Twitter Sentiment Analysis API", 
        "device": str(device),
        "status": "ready",
        "model": "Enhanced BERT with Attention",
        "version": "2.0.0",
        "features": [
            "Google Drive model loading",
            "Advanced text preprocessing", 
            "Multi-head attention",
            "4-class sentiment classification"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": True,
        "device": str(device)
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for input text
    
    Args:
        input_data: TextInput containing the text to analyze
        
    Returns:
        SentimentResponse with prediction results
    """
    try:
        # Get raw prediction from model
        raw_result = predict_sentiment_internal(
            model, input_data.text, tokenizer, device
        )
        
        # Map the raw class to sentiment labels
        predicted_class = raw_result["predicted_class"]
        sentiment = SENTIMENT_LABELS[predicted_class]
        
        # Return structured response
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=round(raw_result["confidence"], 4),
            probabilities={
                "positive": round(raw_result["probabilities"]["class_0"], 4),
                "negative": round(raw_result["probabilities"]["class_1"], 4),
                "neutral": round(raw_result["probabilities"]["class_2"], 4),
                "irrelevant": round(raw_result["probabilities"]["class_3"], 4)
            },
            processed_text=raw_result["processed_text"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

# ============== SERVER STARTUP ==============
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8888,
        log_level="info"
    )
