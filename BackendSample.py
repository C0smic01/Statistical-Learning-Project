from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Classification API")

# Configure static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model configuration
MODEL_PATH = "./emotion_model"
PRETRAINED_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = {
    0: "anger",
    1: "disgust", 
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

def load_model():
    """Load the fine-tuned model or fallback to pretrained"""
    try:
        if os.path.exists(MODEL_PATH):
            logger.info("Loading fine-tuned model...")
            classifier = pipeline(
                "text-classification",
                model=MODEL_PATH,
                tokenizer=MODEL_PATH,
                return_all_scores=True
            )
        else:
            logger.warning("Fine-tuned model not found, using pretrained")
            classifier = pipeline(
                "text-classification",
                model=PRETRAINED_MODEL,
                tokenizer=PRETRAINED_MODEL,
                return_all_scores=True
            )
        return classifier
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Load model at startup
@app.on_event("startup")
async def startup_event():
    app.state.classifier = load_model()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        results = app.state.classifier(text)
        emotions = sorted(
            [{"label": r["label"], "score": round(r["score"], 4)} for r in results[0]],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return {
            "text": text,
            "emotions": emotions,
            "top_emotion": emotions[0]
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)